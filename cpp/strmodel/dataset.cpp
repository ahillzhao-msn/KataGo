#include <fstream>
#include <sstream>
#include <memory>
#include <functional>
#include <zip.h>
#include "strmodel/dataset.h"
#include "game/board.h"
#include "dataio/sgf.h"
#include "core/global.h"
#include "core/fileutils.h"
#include "core/using.h"

namespace StrModel {

DatasetFiles::DatasetFiles(const string& featureDir_)
: featureDir(featureDir_) {}

DatasetFiles::~DatasetFiles() noexcept = default;

string DatasetFiles::featurePath(const string& sgfPath, Player pla, const char* title) const {
  if(featureDir.empty())
    throw StringError("Feature directory not avaiable");

  string sgfPathWithoutExt = Global::chopSuffix(sgfPath, ".sgf");
  string playerString = PlayerIO::playerToString(pla);
  return strprintf("%s/%s_%s%s.zip", featureDir.c_str(), sgfPathWithoutExt.c_str(), playerString.c_str(), title);
}

namespace {

void readBoardMemberFromZip(
  zip_file_t& file,
  BoardFeatures& board,
  int BoardFeatures::* member,
  size_t
) {
  int64_t read = zip_fread(&file, &(board.*member), sizeof(int));
  if(sizeof(int) != read)
    throw StringError(strprintf("Error reading zipped file data: %s", zip_file_strerror(&file)));
}

void readBoardMemberFromZip(
  zip_file_t& file,
  BoardFeatures& board,
  shared_ptr<FeatureVector> BoardFeatures::* member,
  size_t elsPerBoard
) {
  shared_ptr<FeatureVector>& storage = board.*member;
  storage.reset(new FeatureVector(elsPerBoard));
  int64_t read = zip_fread(&file, storage->data(), elsPerBoard*sizeof(float));
  if(elsPerBoard*sizeof(float) != read)
    throw StringError(strprintf("Error reading zipped file data: %s", zip_file_strerror(&file)));
}

template<typename DataMemberType>
uint64_t expectedSize(size_t boards, size_t elsPerBoard) = delete;
template<>
uint64_t expectedSize<int>(size_t boards, size_t elsPerBoard) {
  assert(1 == elsPerBoard); // we can only fit one int per move
  (void)elsPerBoard; // unused in Release build
  return boards * sizeof(int);
}
template<>
uint64_t expectedSize<shared_ptr<FeatureVector>>(size_t boards, size_t elsPerBoard) {
  return boards * elsPerBoard * sizeof(float);
}

// extract one file in the zip, run checks, and extract each stored element to its appropriate move with readMoveMemberFromZip
template<typename T>
void loadFeaturesPart(
  zip_t& archive,
  const char* name,
  vector<BoardFeatures>& features,
  T BoardFeatures::* member,
  size_t elsPerBoard
) {
  int64_t index = zip_name_locate(&archive, name, ZIP_FL_ENC_RAW);
  if(index < 0)
      throw IOError(strprintf("File %s not found in archive.", name));

  zip_stat_t stat;
  if(0 != zip_stat_index(&archive, index, 0, &stat))
    throw IOError(strprintf("Error getting %s file information: %s", name, zip_strerror(&archive)));
  uint64_t expected = expectedSize<T>(features.size(), elsPerBoard);
  if(stat.size != expected)
    throw IOError(strprintf("%s data has %d bytes, but expected %d bytes", name, stat.size, expected));

  unique_ptr<zip_file_t, decltype(&zip_fclose)> file{
    zip_fopen_index(&archive, index, ZIP_RDONLY),
    &zip_fclose
  };
  if(!file)
    throw IOError(strprintf("Error opening %s in zip archive: %s", name, zip_strerror(&archive)));

  for(BoardFeatures& board : features)
    readBoardMemberFromZip(*file, board, member, elsPerBoard);
}

constexpr size_t trunkNumChannels = 384; // strength model is currently fixed to this size
constexpr size_t trunkSize = trunkNumChannels * 19 * 19; // strength model is currently fixed to 19x19
constexpr size_t headNumChannels = 6; // proof of concept model is currently fixed to this size

}

vector<BoardFeatures> DatasetFiles::loadFeatures(const string& path) const {
  int err;
  unique_ptr<zip_t, decltype(&zip_discard)> archive{
    zip_open(path.c_str(), ZIP_RDONLY, &err),
    &zip_discard
  };
  if(!archive) {
    zip_error_t error;
    zip_error_init_with_code(&error, err);
    string errstr = zip_error_strerror(&error);
    zip_error_fini(&error);
    throw IOError("Error opening zip archive: "s + errstr);
  }

  // find out how many positions/trunks are present in the archive
  zip_stat_t stat;
  if(0 != zip_stat_index(archive.get(), 0, 0, &stat))
    throw IOError("Error getting stat of first file in archive: "s + zip_strerror(archive.get()));
  uint64_t expectedCount = stat.size / sizeof(int);
  vector<BoardFeatures> features(expectedCount);

  loadFeaturesPart(*archive, "turn.bin", features, &BoardFeatures::turn, 1);
  if(zip_name_locate(archive.get(), "trunk.bin", ZIP_FL_ENC_RAW) >= 0)
    loadFeaturesPart(*archive, "trunk.bin", features, &BoardFeatures::trunk, trunkSize);
  if(zip_name_locate(archive.get(), "pick.bin", ZIP_FL_ENC_RAW) >= 0)
    loadFeaturesPart(*archive, "pick.bin", features, &BoardFeatures::pick, trunkNumChannels);
  if(zip_name_locate(archive.get(), "head.bin", ZIP_FL_ENC_RAW) >= 0)
    loadFeaturesPart(*archive, "head.bin", features, &BoardFeatures::head, headNumChannels);
  loadFeaturesPart(*archive, "movepos.bin", features, &BoardFeatures::pos, 1);

  return features;
}

namespace {

vector<int> getBufferOfBoardMember(
  const vector<BoardFeatures>& boards,
  int BoardFeatures::* member,
  size_t
) {
  vector<int> buffer(boards.size());
  for(size_t i = 0; i < boards.size(); i++) {
    buffer[i] = boards[i].*member;
  }
  return buffer;
}

FeatureVector getBufferOfBoardMember(
  const vector<BoardFeatures>& boards,
  shared_ptr<FeatureVector> BoardFeatures::* member,
  size_t elsPerBoard
) {
  FeatureVector buffer(boards.size() * elsPerBoard);
  bool haveData = false;
  for(size_t i = 0; i < boards.size(); i++) {
    const BoardFeatures& board = boards[i];
    if(board.*member) {
      assert((board.*member)->size() == elsPerBoard);
      std::copy((board.*member)->begin(), (board.*member)->end(), &buffer[i*elsPerBoard]);
      haveData = true;
    }
  }
  if(!haveData)
    buffer.clear();
  return buffer;
}


template<typename T>
void addFileToZip(zip_t& archive, const vector<T>& buffer, const char* name) {
  unique_ptr<zip_source_t, decltype(&zip_source_free)> source{
    zip_source_buffer(&archive, buffer.data(), buffer.size()*sizeof(T), 0),
    &zip_source_free
  };
  if(!source)
    throw IOError("Error creating zip source: "s + zip_strerror(&archive));

  if(zip_add(&archive, name, source.get()) < 0)
    throw IOError(strprintf("Error adding %s to zip archive: %s", name, zip_strerror(&archive)));
  source.release(); // after zip_add, source is managed by libzip
}

}

void DatasetFiles::storeFeatures(const vector<BoardFeatures>& features, const string& path) const {
  string containingDir = FileUtils::dirname(path);
  if(!containingDir.empty() && !FileUtils::create_directories(containingDir))
    throw IOError("Failed to create directory " + containingDir);

  int err;
  unique_ptr<zip_t, decltype(&zip_discard)> archive{
    zip_open(path.c_str(), ZIP_CREATE | ZIP_TRUNCATE, &err),
    &zip_discard
  };
  if(!archive) {
    zip_error_t error;
    zip_error_init_with_code(&error, err);
    string errstr = zip_error_strerror(&error);
    zip_error_fini(&error);
    throw IOError("Error opening zip archive: "s + errstr);
  }

  // merge individual boards data into contiguous buffers
  vector<int> turnBuffer = getBufferOfBoardMember(features, &BoardFeatures::turn, 1);
  vector<int> posBuffer = getBufferOfBoardMember(features, &BoardFeatures::pos, 1);
  FeatureVector trunkBuffer = getBufferOfBoardMember(features, &BoardFeatures::trunk, trunkSize);
  FeatureVector pickBuffer = getBufferOfBoardMember(features, &BoardFeatures::pick, trunkNumChannels);
  FeatureVector headBuffer = getBufferOfBoardMember(features, &BoardFeatures::head, headNumChannels);

  addFileToZip(*archive, turnBuffer, "turn.bin");
  if(!trunkBuffer.empty()) 
    addFileToZip(*archive, trunkBuffer, "trunk.bin");
  if(!pickBuffer.empty()) 
    addFileToZip(*archive, pickBuffer, "pick.bin");
  if(!headBuffer.empty()) 
    addFileToZip(*archive, headBuffer, "head.bin");
  addFileToZip(*archive, posBuffer, "movepos.bin");

  zip_t* archivep = archive.release();
  if(zip_close(archivep) != 0) {
    IOError error("Error writing zip archive: "s + zip_strerror(archivep));
    zip_discard(archivep);
    throw error;
  }
}

Dataset::Dataset(const vector<Sgf*>& sgfs, const DatasetFiles& files_)
: files(&files_) {
  load(sgfs);
}

Dataset::Dataset(std::istream& stream, const DatasetFiles& files_)
: files(&files_) {
  load(stream);
}

Dataset::Dataset(const string& path, const DatasetFiles& files_)
: files(&files_) {
  std::ifstream stream(path);
  if (!stream.is_open())
    throw IOError("Could not read dataset from " + path);
  load(stream);
  stream.close();
}

void Dataset::load(const vector<Sgf*>& sgfs) {
  // clean any previous data
  games.clear();
  players.clear();
  nameIndex.clear();

  for(Sgf* sgf : sgfs) {
    size_t gameIndex = games.size();
    games.emplace_back();
    Game& game = games[gameIndex];
    game.sgfPath = sgf->fileName;
    string blackName = sgf->getPlayerName(P_BLACK);
    game.black.player = getOrInsertNameIndex(blackName);
    string whiteName = sgf->getPlayerName(P_WHITE);
    game.white.player = getOrInsertNameIndex(whiteName);
    ::Player winner = sgf->getSgfWinner();
    if(P_BLACK == winner)
      game.score = 1;
    else if(P_WHITE == winner)
      game.score = 0;
    else
      game.score = 0.5;
    game.white.prevGame = players[game.white.player].lastOccurrence;
    game.black.prevGame = players[game.black.player].lastOccurrence;
    players[game.white.player].lastOccurrence = gameIndex;
    players[game.black.player].lastOccurrence = gameIndex;
  }
}

void Dataset::load(std::istream& stream) {
  string line;
  std::getline(stream, line);
  if(!stream)
    throw IOError("Could not read dataset header");
  line = Global::trim(line);

  // clean any previous data
  games.clear();
  players.clear();
  nameIndex.clear();

  // map known fieldnames to row indexes, wherever they may be
  enum class F { ignore, sgfPath, whiteName, blackName, whiteRating, blackRating, score, predictedScore, set };
  vector<F> fields;
  string field;
  std::istringstream iss(line);
  while(std::getline(iss, field, ',')) {
    if("File" == field) fields.push_back(F::sgfPath);
    else if("Player White" == field) fields.push_back(F::whiteName);
    else if("Player Black" == field) fields.push_back(F::blackName);
    else if("WhiteRating" == field) fields.push_back(F::whiteRating);
    else if("BlackRating" == field) fields.push_back(F::blackRating);
    else if("Winner" == field || "Judgement" == field || "Score" == field) fields.push_back(F::score);
    else if("PredictedScore" == field) fields.push_back(F::predictedScore);
    else if("Set" == field) fields.push_back(F::set);
    else fields.push_back(F::ignore);
  }

  while (std::getline(stream, line)) {
    size_t gameIndex = games.size();
    games.emplace_back();
    Game& game = games[gameIndex];

    line = Global::trim(line);
    iss = std::istringstream(line);
    int fieldIndex = 0;
    while(std::getline(iss, field, ',')) {
      switch(fields[fieldIndex++]) {
      case F::sgfPath:
        game.sgfPath = field;
        break;
      case F::whiteName:
        game.white.player = getOrInsertNameIndex(field);
        break;
      case F::blackName:
        game.black.player = getOrInsertNameIndex(field);
        break;
      case F::whiteRating:
        game.white.rating = Global::stringToFloat(field);
        break;
      case F::blackRating:
        game.black.rating = Global::stringToFloat(field);
        break;
      case F::score:
        if('b' == field[0] || 'B' == field[0])
          game.score = 1;
        else if('w' == field[0] || 'W' == field[0])
          game.score = 0;
        else
          game.score = std::strtof(field.c_str(), nullptr);
        break;
      case F::predictedScore:
        game.prediction.score = std::strtof(field.c_str(), nullptr);
        break;
      case F::set:
        if("-" == field) game.set = Game::none;
        if("t" == field || "T" == field) game.set = Game::training;
        if("v" == field || "V" == field) game.set = Game::validation;
        if("b" == field || "B" == field) game.set = Game::batch;
        if("e" == field || "E" == field) game.set = Game::test;
        if("x" == field || "X" == field) game.set = Game::exhibition;
        break;
      default:
      case F::ignore:
        break;
      }
    }
    if(!stream)
      throw IOError("Error while reading dataset: bad stream");
    game.white.prevGame = players[game.white.player].lastOccurrence;
    game.black.prevGame = players[game.black.player].lastOccurrence;

    players[game.white.player].lastOccurrence = gameIndex;
    players[game.black.player].lastOccurrence = gameIndex;
  }
}

namespace {
  const char* scoreToString(float score) {
    // only 3 values are really allowed, all perfectly representable in float
    if(0 == score)     return "0";
    if(1 == score)     return "1";
    if(0.5 == score)   return "0.5";
    else               return "(score error)";
  }
}

void Dataset::store(const string& path) const {
  std::ofstream ostrm(path);
  if (!ostrm.is_open())
    throw IOError("Could not write SGF list to " + path);

  ostrm << "File,Player White,Player Black,Score,BlackRating,WhiteRating,PredictedScore,PredictedBlackRating,PredictedWhiteRating,Set\n"; // header

  for(const Game& game : games) {
    string blackName = players[game.black.player].name;
    string whiteName = players[game.white.player].name;

    // file output
    size_t bufsize = game.sgfPath.size() + whiteName.size() + blackName.size() + 100;
    unique_ptr<char[]> buffer( new char[ bufsize ] );
    int printed = std::snprintf(buffer.get(), bufsize, "%s,%s,%s,%s,%.2f,%.2f,%.9f,%f,%f,%c\n",
      game.sgfPath.c_str(), whiteName.c_str(), blackName.c_str(),
      scoreToString(game.score), game.black.rating, game.white.rating,
      game.prediction.score, game.prediction.blackRating, game.prediction.whiteRating, "-TVBE"[game.set]);
    if(printed <= 0)
      throw IOError("Error during formatting.");
    ostrm << buffer.get();
  }

  ostrm.close();
}

Player Dataset::playerColor(PlayerId playerId, GameId gameId) const {
  const Game& game = games.at(gameId);
  if(playerId == game.black.player)
    return P_BLACK;
  else if(playerId == game.white.player)
    return P_WHITE;
  else
    throw ValueError(strprintf("Game %s does not contain player %d (name=%s)",
      game.sgfPath.c_str(), playerId, players[playerId].name.c_str()));
}

vector<int> Dataset::findMovesOfColor(GameId gameId, ::Player pla, size_t capacity) const {
  auto sgf = unique_ptr<CompactSgf>(CompactSgf::loadFile(games[gameId].sgfPath));
  const auto& moves = sgf->moves;
  Rules rules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
  Board board;
  BoardHistory history;
  ::Player initialPla;
  sgf->setupInitialBoardAndHist(rules, board, initialPla, history);

  vector<int> found;
  for(int i = 0; i < moves.size(); i++) {
    if(Board::PASS_LOC != moves[i].loc) {
      if(0 == pla || moves[i].pla == pla) // 0 == pla: all moves, no filter
        found.push_back(i);
    }
  }

  size_t excess = found.size() > capacity ? found.size() - capacity : 0;
  found.erase(found.begin(), found.begin() + excess);
  return found;
}

GamesTurns Dataset::getRecentMoves(PlayerId playerId, ::Player color, size_t capacity) const {
  if(games.empty())
    return {};
  GameId lastGame = games.size() - 1;
  if(playerId >= 0)
    lastGame = players[playerId].lastOccurrence;
  return getRecentMovesStartingAt(playerId, color, lastGame, capacity);
}

GamesTurns Dataset::getRecentMoves(::Player pla, GameId gameId, size_t capacity) const {
  const Game& gameData = games[gameId];
  auto& info = P_BLACK == pla ? gameData.black : gameData.white;
  return getRecentMovesStartingAt(info.player, 0, info.prevGame, capacity);
}

PlayerId Dataset::findOmnipresentPlayer() const {
  if(games.empty())
    return -1;

  PlayerId candidate = games[0].black.player;
  PlayerId alt = games[0].white.player;

  for(size_t i = 1; i < games.size(); i++) {
    PlayerId black = games[i].black.player;
    PlayerId white = games[i].white.player;

    if(black != alt && white != alt)
      alt = -1;

    if(black == candidate || white == candidate) {
      continue;
    }
    else if(black == alt || white == alt) {
      candidate = alt;
      alt = -1;
    }
    else {
      return -1;
    }
  }

  if(alt < 0)
    return candidate;
  else
    return -1;
}

void Dataset::randomSplit(Rand& rand, float trainingPart, float validationPart) {
  assert(trainingPart >= 0);
  assert(validationPart >= 0);
  assert(trainingPart + validationPart <= 1);
  size_t N = games.size();
  vector<uint32_t> gameIdxs(N);
  rand.fillShuffledUIntRange(N, gameIdxs.data());
  size_t trainingCount = std::llround(trainingPart * N);
  size_t validationCount = std::llround(validationPart * N);
  for(size_t i = 0; i < trainingCount; i++)
    games[gameIdxs[i]].set = Game::training;
  for(size_t i = trainingCount; i < trainingCount + validationCount && i < N; i++)
    games[gameIdxs[i]].set = Game::validation;
  for(size_t i = trainingCount + validationCount; i < N; i++)
    games[gameIdxs[i]].set = Game::test;
}

void Dataset::randomBatch(Rand& rand, size_t batchSize) {
  vector<size_t> trainingIdxs;
  for(size_t i = 0; i < games.size(); i++)
    if(~games[i].set & 1)
      trainingIdxs.push_back(i);
  batchSize = min(batchSize, trainingIdxs.size());
  vector<uint32_t> batchIdxs(trainingIdxs.size());
  rand.fillShuffledUIntRange(trainingIdxs.size(), batchIdxs.data());
  for(size_t i = 0; i < batchSize; i++)
    games[trainingIdxs[batchIdxs[i]]].set = Game::batch;
  for(size_t i = batchSize; i < batchIdxs.size(); i++)
    games[trainingIdxs[batchIdxs[i]]].set = Game::training;
}

vector<BoardFeatures> Dataset::loadFeatures(GameId gameId, ::Player pla, const char* title) const {
  string path = files->featurePath(games.at(gameId).sgfPath, pla, title);
  return files->loadFeatures(path);
}

void Dataset::storeFeatures(const vector<BoardFeatures>& features, GameId gameId, ::Player pla, const char* title) const {
  string path = files->featurePath(games.at(gameId).sgfPath, pla, title);
  files->storeFeatures(features, path);
}

size_t Dataset::getOrInsertNameIndex(const string& name) {
  auto it = nameIndex.find(name);
  if(nameIndex.end() == it) {
    size_t index = players.size();
    players.push_back({name, -1});
    bool success;
    tie(it, success) = nameIndex.insert({name, index});
  }
  return it->second;
}

GamesTurns Dataset::getRecentMovesStartingAt(PlayerId playerId, ::Player color, GameId gameId, size_t capacity) const {
  GamesTurns gamesTurns;

  while(0 < capacity && gameId >= 0) {
    GameId h = gameId;
    const Dataset::Game& historicGame = games[h];

    if(playerId >= 0) {
      // follow player's individual history
      color = playerColor(playerId, h);
      gameId = P_BLACK == color ? historicGame.black.prevGame : historicGame.white.prevGame;
    }
    else {
      // follow every game in the dataset
      gameId--;
    }

    vector<int> found = findMovesOfColor(h, color, capacity);
    capacity -= found.size();
    gamesTurns.bygame[h] = move(found);
  }

  return gamesTurns;
}

} // end namespace StrModel
