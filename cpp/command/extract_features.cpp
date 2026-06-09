#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../program/play.h"
#include "../strmodel/dataset.h"
#include "../strmodel/precompute.h"
#include "../command/commandline.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../main.h"
#include <iomanip>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <zip.h>
#include "../core/using.h"

namespace
{

using std::ref;
using namespace StrModel;
using clock = std::chrono::system_clock;
using time_point = std::chrono::time_point<clock>;

// params of this command
struct Parameters {
  string modelFile;
  string listFile; // CSV file listing all SGFs to be fed into the rating system
  string featureDir; // Directory for move feature cache
  Selection selection; // features to extract
  int windowSize; // Extract up to this many recent moves
  int batchSize; // Send this many moves to the GPU at once (per evaluator thread)
  int batchThreads; // Number of concurrent evaluator threads feeding positions to GPU
  int workerThreads; // Number of concurrent CPU workers
  bool recompute; // Overwrite existing ZIPs, do not reuse them
  bool printRecentMoves; // Output information on which moves are recent moves for which game
};

// stores recent moves of one player in one game in a Dataset
struct RecentMoves {
  GameId game;
  Player pla;
  GamesTurns sel;
};

// handles the actual extraction process
class ExtractFeatures {

public:

  ExtractFeatures(const Parameters& params, DatasetFiles& files, Dataset& dataset, NNEvaluator& evaluator) noexcept;

  // find every game in the dataset that we want to compute features for
  vector<RecentMoves> getAllQueries() const;
  static GamesTurns mergeQueries(const vector<RecentMoves>& recentByGame);
  void printRecentMoves(const vector<RecentMoves>& recentMoves) const;

  // thread worker function to construct recent move ZIPs out of available game feature ZIPs
  void buildRecentMoves(
    RecentMoves* workBegin,       // workload for this thread as begin/end pair,
    RecentMoves* workEnd,         // specifies which moves are recent to which games
    std::atomic<size_t>& counter, // shared counter to report common progress
    size_t total,                 // overall number of items, 100% progress
    time_point startTime          // common start time of all workers
  ) const;

  Logger* logger;

private:

  Parameters params;
  DatasetFiles* files;
  Dataset* dataset;
  NNEvaluator* evaluator;

};

Parameters parseArgs(const vector<string>& args, ConfigParser& cfg);
// unique_ptr<LoadedModel, void(*)(LoadedModel*)> loadModel(string file);
unique_ptr<NNEvaluator> createEvaluator(const string& modelFile, int evaluatorThreads, int batchSize);

// takes a subset of the dataset games, feeds them into NN and prepares the results
class Worker {
 public:
  Worker() = default;
  explicit Worker(Precompute&& precompute_);

  static void setWork(GamesTurns& work_); // assign shared workload for all workers
  void operator()(); // to be called as its own thread

  static Logger* logger; // shared logger for all workers
  static Dataset* dataset; // shared dataset for all workers
  static bool recompute; // shared configuration: set true to disregard existing zips
  static Selection selection; // what to compute
  static bool printResultsDebug; // summary of precompute results to stdout

 private:

  static GamesTurns* work;
  static map<GameId, vector<int>>::iterator workIterator; // points to next work item
  static std::mutex workMutex; // synchronizes access to workIterator
  static size_t progress; // shared counter for finished work
  static time_point startTime; // time when work was assigned
  static std::mutex reportMutex; // synchronizes access to logger

  Precompute precompute;

  bool fetchWork(GameId& gameId, vector<int>*& turns);
  static bool featuresPreexist(GameId gameId, const vector<int>& turns);
  void reportProgress(const string& message);
  void reportError(const string& sgfPath, const char* what);

};

}

int MainCmds::extract_features(const vector<string>& args) {
  ConfigParser cfg;
  Parameters params = parseArgs(args, cfg);
  Logger logger(&cfg, false, true);
  logger.write(Version::getKataGoVersionForHelp());

  Board::initHash();
  ScoreValue::initTables();

  DatasetFiles files(params.featureDir);
  Dataset dataset(params.listFile, files);

  auto evaluator = createEvaluator(params.modelFile, params.batchThreads, params.batchSize);

  ExtractFeatures extract(params, files, dataset, *evaluator);
  extract.logger = &logger;

  logger.write("Find recent moves of all train/eval/test games using window size " + Global::intToString(params.windowSize) + "...");

  vector<RecentMoves> recentByGame = extract.getAllQueries();
  if(params.printRecentMoves)
    extract.printRecentMoves(recentByGame);

  size_t total = 0;
  {
    // combine exactly what needs to be precomputed for which game
    GamesTurns recentAll = ExtractFeatures::mergeQueries(recentByGame);
    total = recentAll.bygame.size();
    logger.write(Global::intToString(total) + " games found. Extracting...");
  
    // evaluate all queries using concurrent GPU workers
    Worker::logger = &logger;
    Worker::dataset = &dataset;
    Worker::recompute = params.recompute;
    Worker::selection = params.selection;
    Worker::printResultsDebug = params.printRecentMoves; // clumsy tie of unrelated debug print options
    Worker::setWork(recentAll);
  
    vector<Worker> workers;
    vector<std::thread> threads;
    for(int i = 0; i < params.workerThreads; i++)
      workers.emplace_back(Precompute(*evaluator));
    for(int i = 0; i < params.workerThreads; i++)
      threads.emplace_back(ref(workers[i]));
    for(auto& thread : threads)
      thread.join();
  }

  // all picks are now available as precomputed pick ZIPs;
  // piece them back together into recent move sets and output recent ZIPs
  {
    logger.write(strprintf("Accumulating recent moves using %d threads...", params.workerThreads));
    auto startTime = clock::now();
    total = recentByGame.size();
    std::atomic<size_t> progress(0);
    vector<std::thread> threads;
    for(int i = 0; i < params.workerThreads; i++) {
      RecentMoves* workBegin = &recentByGame[total*i/params.workerThreads];
      RecentMoves* workEnd = &recentByGame[total*(i+1)/params.workerThreads];
      threads.emplace_back(&ExtractFeatures::buildRecentMoves, extract, workBegin, workEnd, ref(progress), total, startTime);
    }
    for(auto& thread : threads)
      thread.join();
  }

  ScoreValue::freeTables();
  logger.write("All cleaned up, quitting");
  return 0;
}

namespace {

ExtractFeatures::ExtractFeatures(const Parameters& params_, DatasetFiles& files_, Dataset& dataset_, NNEvaluator& evaluator_) noexcept
: logger(nullptr), params(params_), files(&files_), dataset(&dataset_), evaluator(&evaluator_)
{}

vector<RecentMoves> ExtractFeatures::getAllQueries() const {
  vector<RecentMoves> recentByGame;
  for(GameId i = 0; i < dataset->games.size(); i++) {
    if(Dataset::Game::none == dataset->games[i].set)
      continue;

    if(!params.recompute) {
      string blackZipPath = files->featurePath(dataset->games[i].sgfPath, P_BLACK, "Recent");
      string whiteZipPath = files->featurePath(dataset->games[i].sgfPath, P_WHITE, "Recent");
      if(FileUtils::exists(blackZipPath) && FileUtils::exists(whiteZipPath))
        continue;
    }

    if(logger)
      logger->write(strprintf("Find recent moves of %s", dataset->games[i].sgfPath.c_str()));
    recentByGame.push_back({i, P_BLACK, dataset->getRecentMoves(P_BLACK, i, params.windowSize)});
    recentByGame.push_back({i, P_WHITE, dataset->getRecentMoves(P_WHITE, i, params.windowSize)});
  }
  return recentByGame;
}

GamesTurns ExtractFeatures::mergeQueries(const vector<RecentMoves>& recentByGame) {
  GamesTurns merged;
  for(const RecentMoves& recent : recentByGame) {
    for(const auto& game_turns : recent.sel.bygame) {
      vector<int> unioned;
      vector<int>& a = merged.bygame[game_turns.first];
      const vector<int>& b = game_turns.second;
      std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(unioned));
      a = move(unioned); // save to merged
    }
  }
  return merged;
}

void ExtractFeatures::printRecentMoves(const vector<RecentMoves>& recentMoves) const {
  for(const RecentMoves& rec : recentMoves) {
    const string& sgfPath = dataset->games[rec.game].sgfPath;
    std::cout << strprintf("Recent moves to %s for %s:\n", sgfPath.c_str(), PlayerIO::playerToString(rec.pla).c_str());
    for(const auto& kv : rec.sel.bygame) {
      const vector<int>& turns = kv.second;
      std::cout << strprintf("  %d from %s:", turns.size(), dataset->games[kv.first].sgfPath.c_str());
      for(int turn : turns)
        std::cout << ' ' << turn;
      std::cout << "\n";
    }
  }
}

void ExtractFeatures::buildRecentMoves(
  RecentMoves* workBegin,       // workload for this thread as begin/end pair,
  RecentMoves* workEnd,         // specifies which moves are recent to which games
  std::atomic<size_t>& counter, // shared counter to report common progress
  size_t total,                 // overall number of items, 100% progress
  time_point startTime          // common start time of all workers
) const {
  auto reportProgress = [&] (const RecentMoves* moves, const char* message = "") {
    size_t p = ++counter;
    auto elapsedTime = clock::now() - startTime;
    auto remainingTime = elapsedTime * (total - p) / p;
    string remainingString = Global::longDurationToString(remainingTime);
    logger->write(strprintf("%d/%d (%s remaining): %s (%s) %s",
      p, total, remainingString.c_str(),
      dataset->games[moves->game].sgfPath.c_str(),
      PlayerIO::playerToString(moves->pla).c_str(),
      message));
  };
  for(RecentMoves* moves = workBegin; moves != workEnd; ++moves) try {
    const Dataset::Game& game = dataset->games.at(moves->game);
    assert(P_BLACK == moves->pla || P_WHITE == moves->pla);
    PlayerId playerId = P_BLACK == moves->pla ? game.black.player : game.white.player;

    // get relevant precomputations from disk
    vector<BoardFeatures> features; // of all recent moves

    for(auto& kv : moves->sel.bygame) {
      Player pla = dataset->playerColor(playerId, kv.first);
      vector<BoardFeatures> gameFeatures = dataset->loadFeatures(kv.first, pla, "Features");
      vector<BoardFeatures> filtered = Precompute::filter(gameFeatures, kv.second); // only keep the actual recent moves
      features.insert(features.end(), filtered.begin(), filtered.end());
    }

    dataset->storeFeatures(features, moves->game, moves->pla, "Recent");
    reportProgress(moves);
  }
  catch(const std::exception& e) {
    reportProgress(moves, e.what());
  }
}

Parameters parseArgs(const vector<string>& args, ConfigParser& cfg) {
  KataGoCommandLine cmd("Precompute move features for all games in the dataset.");
  cmd.addConfigFileArg("","analysis_example.cfg");
  cmd.addModelFileArg();
  cmd.setShortUsageArgLimit();

  TCLAP::ValueArg<string> listArg("l","list","CSV file listing all SGFs to be fed into the rating system.",true,"","FILE",cmd);
  TCLAP::ValueArg<string> featureDirArg("d","featuredir","Directory for move feature cache.",true,"","DIR",cmd);
  TCLAP::ValueArg<int> windowSizeArg("s","window-size","Extract up to this many recent moves.",false,1000,"SIZE",cmd);
  TCLAP::ValueArg<int> batchSizeArg("b","batch-size","Send this many moves to the GPU at once (per evaluator thread).",false,10,"SIZE",cmd);
  TCLAP::ValueArg<int> batchThreadsArg("t","batch-threads","Number of concurrent evaluator threads feeding positions to GPU.",false,4,"COUNT",cmd);
  TCLAP::ValueArg<int> workerThreadsArg("w","worker-threads","Number of concurrent CPU workers.",false,4,"COUNT",cmd);
  TCLAP::SwitchArg withTrunkArg("T","with-trunk","Extract trunk features.",cmd,false);
  TCLAP::SwitchArg withPickArg("P","with-pick","Extract pick features.",cmd,false);
  TCLAP::SwitchArg withHeadArg("H","with-head","Extract head features.",cmd,false);
  TCLAP::SwitchArg recomputeArg("r","recompute","Overwrite existing ZIPs, do not reuse them.",cmd,false);
  TCLAP::SwitchArg printRecentMovesArg("p","print-recent-moves","Output information on which moves are recent moves for which game.",cmd,false);
  cmd.addOverrideConfigArg();
  cmd.parseArgs(args);

  Parameters params;
  params.modelFile = cmd.getModelFile();
  params.listFile = listArg.getValue();
  params.featureDir = featureDirArg.getValue();
  params.selection.trunk = withTrunkArg.getValue();
  params.selection.pick = withPickArg.getValue();
  params.selection.head = withHeadArg.getValue();
  params.windowSize = windowSizeArg.getValue();
  params.batchSize = batchSizeArg.getValue();
  params.batchThreads = batchThreadsArg.getValue();
  params.workerThreads = workerThreadsArg.getValue();
  params.recompute = recomputeArg.getValue();
  params.printRecentMoves = printRecentMovesArg.getValue();
  cmd.getConfig(cfg);

  if(!params.selection.trunk && !params.selection.pick && !params.selection.head) {
    throw StringError("No features selected for extraction.");
  }

  return params;
}

unique_ptr<NNEvaluator> createEvaluator(const string& modelFile, int evaluatorThreads, int batchSize) {
  assert(evaluatorThreads > 0);
  assert(batchSize > 0);
  const int maxConcurrentEvals = evaluatorThreads*2;
  constexpr int nnXLen = 19;
  constexpr int nnYLen = 19;
  vector<int> gpuIdxByServerThread(evaluatorThreads, -1);
  auto evaluator = make_unique<NNEvaluator>(
    modelFile,
    modelFile,
    "", // expectedSha256
    nullptr, // logger
    batchSize,
    maxConcurrentEvals,
    nnXLen,
    nnYLen,
    true, // requireExactNNLen
    false, // inputsUseNHWC
    23, // nnCacheSizePowerOfTwo
    17, // nnMutexPoolSizePowerOfTwo
    false, // debugSkipNeuralNet
    "", // openCLTunerFile
    "", // homeDataDirOverride
    false, // openCLReTunePerBoardSize
    enabled_t::False, // useFP16Mode
    enabled_t::False, // useNHWCMode
    evaluatorThreads, // numNNServerThreadsPerModel
    gpuIdxByServerThread,
    "", // nnRandSeed
    false, // doRandomize (for symmetry)
    0 // defaultSymmetry
  );
  evaluator->spawnServerThreads();
  return evaluator;
}

Logger* Worker::logger;
Dataset* Worker::dataset;
bool Worker::recompute;
Selection Worker::selection;
bool Worker::printResultsDebug;
GamesTurns* Worker::work;
map<GameId, vector<int>>::iterator Worker::workIterator;
std::mutex Worker::workMutex;
size_t Worker::progress;
time_point Worker::startTime;
std::mutex Worker::reportMutex;

Worker::Worker(Precompute&& precompute_)
: precompute(std::move(precompute_))
{}

void Worker::setWork(GamesTurns& work_) {
  work = &work_;
  workIterator = work->bygame.begin();
  progress = 0;
  startTime = clock::now();
}

void printResultToStdout(const string& sgfPath, const vector<BoardFeatures>& features) {
  if(features.empty())
    std::cout << strprintf("Result %s: empty result\n", sgfPath.c_str());
  else
    std::cout << strprintf("Result %s: move %d-%d\n", sgfPath.c_str(), features.front().turn, features.back().turn);
}

void Worker::operator()() try {
  GameId gameId;
  vector<int>* turns;
  while(fetchWork(gameId, turns)) try {
    if(!recompute && featuresPreexist(gameId, *turns))
      continue;

    vector<BoardQuery> query = Precompute::makeQuery(*turns, selection);
    const string& sgfPath = dataset->games.at(gameId).sgfPath;
    auto sgf = unique_ptr<CompactSgf>(CompactSgf::loadFile(sgfPath));
    assert(sgf);
    vector<BoardResult> results = precompute.evaluate(*sgf, query);
    vector<BoardFeatures> features = Precompute::combine(results);
    if(printResultsDebug)
      printResultToStdout(sgfPath, features);

    vector<BoardFeatures> blackFeatures, whiteFeatures;
    std::copy_if(features.begin(), features.end(), std::back_inserter(blackFeatures),
      [](const BoardFeatures& f) { return P_BLACK == f.pla; });
    std::copy_if(features.begin(), features.end(), std::back_inserter(whiteFeatures),
      [](const BoardFeatures& f) { return P_WHITE == f.pla; });
    dataset->storeFeatures(blackFeatures, gameId, P_BLACK, "Features");
    dataset->storeFeatures(whiteFeatures, gameId, P_WHITE, "Features");

    reportProgress(sgfPath);
  }
  catch(const std::exception& e) {
    reportError(dataset->games.at(gameId).sgfPath, e.what());
  }
}
catch(const std::exception& e) {
  logger->write("Unexpected error in worker thread: "s + e.what());
}

bool Worker::fetchWork(GameId& gameId, vector<int>*& turns) {
  if(work->bygame.end() == workIterator)
    return false;

  std::lock_guard<std::mutex> lock(workMutex);
  gameId = workIterator->first;
  turns = &workIterator->second;
  ++workIterator;
  return true;
}

bool Worker::featuresPreexist(GameId gameId, const vector<int>& turns)
try {
  // load existing features to see what we already have
  vector<BoardFeatures> blackFeatures = dataset->loadFeatures(gameId, P_BLACK, "Features");
  vector<BoardFeatures> whiteFeatures = dataset->loadFeatures(gameId, P_WHITE, "Features");

  // we only accept features if they match our selection criteria
  auto hasSelection = [selection=selection](const BoardFeatures& f) {
    return (!selection.trunk || f.trunk)
        && (!selection.pick || f.pick)
        && (!selection.head || f.head);
  };
  auto blackEnd = std::partition(blackFeatures.begin(), blackFeatures.end(), hasSelection);
  auto whiteEnd = std::partition(whiteFeatures.begin(), whiteFeatures.end(), hasSelection);

  // find the turn numbers where we have features and check if they are complete
  set<int> existing;
  auto inserter = std::inserter(existing, existing.end());
  auto getTurn = [](const BoardFeatures& f) { return f.turn; };
  std::transform(blackFeatures.begin(), blackEnd, inserter, getTurn);
  std::transform(whiteFeatures.begin(), whiteEnd, inserter, getTurn);
  return std::includes(existing.begin(), existing.end(), turns.begin(), turns.end());
} catch(const IOError& ) {
  // zip not found or not read; nothing to worry about, just recompute
  return false;
}

void Worker::reportProgress(const string& message) {
  std::lock_guard<std::mutex> lock(reportMutex);
  size_t p = ++progress;
  size_t total = work->bygame.size();
  auto elapsedTime = clock::now() - startTime;
  auto remainingTime = elapsedTime * (total - p) / p;
  string remainingString = Global::longDurationToString(remainingTime);
  logger->write(strprintf("%d/%d (%s remaining): %s", p, total, remainingString.c_str(), message.c_str()));
}

void Worker::reportError(const string& sgfPath, const char* what) {
  reportProgress(strprintf("Error processing %s: %s", sgfPath.c_str(), what));
}

}
