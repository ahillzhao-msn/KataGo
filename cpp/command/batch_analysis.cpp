// batch_analysis.cpp — Batch SGF analysis with per-move + per-game feature extraction
//
// Output formats:
//   File mode:   one combined <sgf-stem>.npz per game (always zlib compressed):
//                [4B B_size][B KAB2 payload][4B W_size][W KAB2 payload]
//   Stream mode: uncompressed per-player frames on stdout:
//                [1B side 'B'/'W'][4B uint32 idLen][game id][4B uint32 size][KAB2 payload]
//                terminated by a single 0x00 byte. Game id = SGF filename stem.
//
// KAB2 payload:
//   NPZHeader (96 bytes, embeds PlayerSummary) |
//   [ scalars(10) + pick(C) + avgTrunk(C) ] × N  [optionally zlib]
//
// Scalars layout — white's perspective throughout:
//   [0] whiteWinProb    [1] whiteLossProb   [2] whiteNoResultProb
//   [3] whiteScoreMean/50  [4] shorttermScoreError/10
//   [5] policyPrior  [6] policyRank/361  [7] isWhite
//   [8] winDelta    (whiteWinProb[t+1]  - whiteWinProb[t],   deferred fill)
//   [9] scoreDelta/50 (whiteScoreMean[t+1] - whiteScoreMean[t], deferred fill)
//
// HumanSL (optional, -human-model):
//   Second pass over each game; tests 3 candidate rank profiles per player
//   (selected from meanLogPrior estimate); stores best-match rank in PlayerSummary.

#include "batch_analysis.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../core/config_parser.h"
#include "../core/fileutils.h"
#include "../core/global.h"
#include "../dataio/sgf.h"
#include "../main.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"   // also pulls in sgfmetadata.h
#include "../program/play.h"
#include "../program/setup.h"
#include <zlib.h>
#include <chrono>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

using namespace std;

namespace MainCmds {

static const int SCALAR_DIM       = 10;
static const int HUMAN_CANDIDATES = 3;   // rank profiles tested per player per game
static const int MAX_GAME_MOVES   = 800; // truncate games longer than this (corrupted SGF guard)
static const int MAX_HUMAN_MOVES  = 400; // cap HumanSL pass moves to prevent runaway

// 29 profiles ordered weakest → strongest (idx 0=20k … idx 28=9d)
static const int NUM_RANK_PROFILES = 29;
static const char* RANK_PROFILES[NUM_RANK_PROFILES] = {
  "rank_20k","rank_19k","rank_18k","rank_17k","rank_16k","rank_15k",
  "rank_14k","rank_13k","rank_12k","rank_11k","rank_10k","rank_9k",
  "rank_8k", "rank_7k", "rank_6k", "rank_5k", "rank_4k", "rank_3k",
  "rank_2k", "rank_1k", "rank_1d", "rank_2d", "rank_3d", "rank_4d",
  "rank_5d", "rank_6d", "rank_7d", "rank_8d", "rank_9d"
};

static string rankIdxToStr(float idx) {
  int i = (int)(idx + 0.5f);
  if(i < 0 || i >= NUM_RANK_PROFILES) return "?";
  return RANK_PROFILES[i];
}

// Map meanLogPrior → 3 neighbouring candidate rank indices.
// Empirical range: -5.9 (20k, idx=0) to -0.5 (9d, idx=28).
// More negative logPrior = weaker player = lower idx.
static vector<int> rankCandidates(float logPrior) {
  float t = std::max(0.0f, std::min(1.0f, (logPrior + 5.9f) / 5.4f));
  int mid = std::min(27, std::max(1, (int)(t * 28.0f + 0.5f)));
  return {mid - 1, mid, mid + 1};
}

// ─── Per-player aggregate statistics (embedded in NPZHeader) ──────────

struct PlayerSummary {
  float accuracy1;        // fraction top-1 match (rank == 0)
  float accuracy3;        // fraction top-3 match (rank <= 2)
  float meanLogPrior;     // mean log(policyPrior) — core strength signal
  float meanWinRate;      // mean player win probability
  float meanScoreLead;    // mean player score lead (points, player's perspective)
  float meanComplexity;   // mean shorttermScoreError
  float scoreVariance;    // variance of player score lead (Welford online)
  float approxScoreDrop;  // mean max(0, -playerWinDelta) — mistake proxy
  float meanWinDelta;     // mean signed win delta (player's perspective)
  float meanScoreDelta;   // mean signed score delta (player's perspective)
  float humanRankIdx;     // best HumanSL rank index (0=20k…28=9d), -1 if not used
  float humanLogPrior;    // meanLogPrior under best HumanSL profile
  float reserved[4];      // pad to 16 floats = 64 bytes
};

#pragma pack(push, 1)
struct NPZHeader {
  char magic[4];         // "KAB2"
  int32_t numMoves;
  int32_t scalarDim;     // SCALAR_DIM = 10
  int32_t trunkDim;      // avg-pooled trunk channels (dynamic)
  int32_t pickDim;       // pick channels (= trunkDim)
  int32_t nnXLen;
  int32_t nnYLen;
  int32_t flags;         // bit0 = zlib compressed
  PlayerSummary summary; // 64 bytes
};
// Total: 4 + 7*4 + 64 = 96 bytes
#pragma pack(pop)

// ─── Accumulator → PlayerSummary ─────────────────────────────────────

struct PlayerAcc {
  int n = 0;
  float sumTop1 = 0, sumTop3 = 0;
  float sumLogPrior = 0;
  float sumWinRate = 0, sumScoreLead = 0, sumComplexity = 0;
  float sumWinDelta = 0, sumScoreDelta = 0, sumDrop = 0;
  float wfMean = 0, wfM2 = 0;

  void addMove(int rank, float policyPrior, float playerWinRate,
               float playerScoreLead, float complexity) {
    n++;
    sumTop1       += (rank == 0) ? 1.0f : 0.0f;
    sumTop3       += (rank <= 2) ? 1.0f : 0.0f;
    sumLogPrior   += std::log(std::max(policyPrior, 1e-10f));
    sumWinRate    += playerWinRate;
    sumScoreLead  += playerScoreLead;
    sumComplexity += complexity;
    float d1 = playerScoreLead - wfMean;
    wfMean += d1 / n;
    wfM2   += d1 * (playerScoreLead - wfMean);
  }

  void addDelta(float playerWinDelta, float playerScoreDelta) {
    sumWinDelta   += playerWinDelta;
    sumScoreDelta += playerScoreDelta;
    sumDrop       += std::max(0.0f, -playerWinDelta);
  }

  PlayerSummary toSummary() const {
    PlayerSummary s = {};
    s.humanRankIdx  = -1.0f;  // not yet computed
    s.humanLogPrior = 0.0f;
    if(n == 0) return s;
    s.accuracy1       = sumTop1 / n;
    s.accuracy3       = sumTop3 / n;
    s.meanLogPrior    = sumLogPrior / n;
    s.meanWinRate     = sumWinRate / n;
    s.meanScoreLead   = sumScoreLead / n;
    s.meanComplexity  = sumComplexity / n;
    s.scoreVariance   = (n > 1) ? wfM2 / (n - 1) : 0.0f;
    s.approxScoreDrop = sumDrop / n;
    s.meanWinDelta    = sumWinDelta / n;
    s.meanScoreDelta  = sumScoreDelta / n;
    return s;
  }
};

// ─── List entry ───────────────────────────────────────────────────────

struct ListEntry {
  string sgfPath;
  string playerBlack, playerWhite;
  double blackElo = 1500, whiteElo = 1500;
  double score = 0.5;
  char set = 'T';
};

// ─── Helpers ─────────────────────────────────────────────────────────

#ifdef _WIN32
static void ensureDir(const string& path) { _mkdir(path.c_str()); }
#else
static void ensureDir(const string& path) { mkdir(path.c_str(), 0755); }
#endif

static vector<ListEntry> parseList(const string& path) {
  vector<ListEntry> entries;
  ifstream in(path);
  if(!in) { cerr << "Cannot open list: " << path << endl; return entries; }
  string line;
  getline(in, line);
  while(getline(in, line)) {
    if(line.empty()) continue;
    stringstream ss(line);
    ListEntry e;
    string f;
    getline(ss, e.sgfPath,     ',');
    getline(ss, e.playerBlack, ',');
    getline(ss, e.playerWhite, ',');
    getline(ss, f, ','); if(!f.empty()) e.score    = stod(f);
    getline(ss, f, ','); if(!f.empty()) e.blackElo = stod(f);
    getline(ss, f, ','); if(!f.empty()) e.whiteElo = stod(f);
    getline(ss, f, ','); e.set = f.empty() ? 'T' : f[0];
    entries.push_back(e);
  }
  return entries;
}

static string sgfToBase(const string& sgfPath) {
  // Extract filename stem from SGF path: strip directory and ".sgf" extension.
  size_t slash = sgfPath.rfind('/');
  if(slash == string::npos) slash = sgfPath.rfind('\\');
  string name = (slash == string::npos) ? sgfPath : sgfPath.substr(slash + 1);
  if(name.size() > 4 && name.substr(name.size() - 4) == ".sgf")
    name = name.substr(0, name.size() - 4);
  return name;
}

static vector<char> zlibCompress(const char* data, size_t bytes) {
  if(bytes == 0) return {};
  uLongf clen = compressBound((uLong)bytes);
  vector<char> out(clen);
  if(compress((Bytef*)out.data(), &clen, (const Bytef*)data, (uLong)bytes) == Z_OK)
    out.resize(clen);
  else
    out.assign(data, data + bytes);
  return out;
}

// Serialise one player's data into an in-memory KAB2 buffer.
// trunkCh == 0 produces scalars-only (lite) mode.
static vector<char> serializePlayer(
  const vector<float>& movesData,
  int numMoves, int trunkCh, int nnXLen, int nnYLen,
  bool doCompress, const PlayerSummary& summary
) {
  if(numMoves < 1) return {};

  NPZHeader hdr;
  memcpy(hdr.magic, "KAB2", 4);
  hdr.numMoves  = (int32_t)numMoves;
  hdr.scalarDim = (int32_t)SCALAR_DIM;
  hdr.trunkDim  = (int32_t)trunkCh;
  hdr.pickDim   = (int32_t)trunkCh;
  hdr.nnXLen    = (int32_t)nnXLen;
  hdr.nnYLen    = (int32_t)nnYLen;
  hdr.flags     = doCompress ? 1 : 0;
  hdr.summary   = summary;

  vector<char> buf(sizeof(hdr));
  memcpy(buf.data(), &hdr, sizeof(hdr));

  const char* raw  = (const char*)movesData.data();
  size_t      rawB = movesData.size() * sizeof(float);
  if(doCompress) {
    vector<char> comp = zlibCompress(raw, rawB);
    int32_t clen = (int32_t)comp.size();
    size_t prev = buf.size();
    buf.resize(prev + 4 + comp.size());
    memcpy(buf.data() + prev, &clen, 4);
    memcpy(buf.data() + prev + 4, comp.data(), comp.size());
  } else {
    buf.insert(buf.end(), raw, raw + rawB);
  }
  return buf;
}

static void writePlayerFile(
  const string& path, const vector<float>& movesData,
  int numMoves, int trunkCh, int nnXLen, int nnYLen,
  bool doCompress, const PlayerSummary& summary
) {
  if(numMoves < 1) return;
  ofstream out(path, ios::binary);
  if(!out) { cerr << "Cannot write: " << path << endl; return; }
  auto buf = serializePlayer(movesData, numMoves, trunkCh, nnXLen, nnYLen, doCompress, summary);
  out.write(buf.data(), buf.size());
}

// Stream one frame to stdout: [side 'B'/'W'][uint32 idLen][game id][uint32 size][payload]
// Game id (SGF filename stem) lets the Python reader pair B/W frames and map
// results back to input games even when one side is missing.
static void streamPlayer(char side, const string& gameId, const vector<char>& payload) {
  uint32_t idLen = (uint32_t)gameId.size();
  uint32_t sz    = (uint32_t)payload.size();
  cout.write(&side, 1);
  cout.write(reinterpret_cast<const char*>(&idLen), 4);
  cout.write(gameId.data(), (streamsize)gameId.size());
  cout.write(reinterpret_cast<const char*>(&sz), 4);
  cout.write(payload.data(), (streamsize)payload.size());
  cout.flush();
}

// Append one move's feature block.
// Pick is extracted directly from trunk (NCHW: trunk[ch*spatial + rowPos])
// rather than relying on the backend's includePick path, which can return
// zeros on the first evaluation due to buffer initialisation ordering.
// Scalars [8,9] = 0 as deferred-fill placeholders for winDelta / scoreDelta.
static void appendMoveRecord(
  vector<float>& buf,
  const NNOutput* nn, Player evalPla,
  float policyPrior, int rank, int rowPos,
  int trunkCh, int nnXLen, int nnYLen
) {
  buf.push_back(nn->whiteWinProb);
  buf.push_back(nn->whiteLossProb);
  buf.push_back(nn->whiteNoResultProb);
  buf.push_back(nn->whiteScoreMean / 50.0f);
  buf.push_back(nn->shorttermScoreError / 10.0f);
  buf.push_back(policyPrior);
  buf.push_back((float)rank / 361.0f);
  buf.push_back(evalPla == P_WHITE ? 1.0f : 0.0f);
  buf.push_back(0.0f);  // [8] winDelta   — deferred
  buf.push_back(0.0f);  // [9] scoreDelta — deferred

  int spatial = nnXLen * nnYLen;

  // Pick: trunk slice at move location (extracted from trunk, NCHW layout)
  if(nn->trunk != nullptr && rowPos >= 0 && rowPos < spatial) {
    for(int ch = 0; ch < trunkCh; ch++)
      buf.push_back(nn->trunk[ch * spatial + rowPos]);
  } else {
    buf.insert(buf.end(), trunkCh, 0.0f);
  }

  // AvgTrunk: spatial mean of full trunk
  if(nn->trunk != nullptr) {
    for(int ch = 0; ch < trunkCh; ch++) {
      float sum = 0.0f;
      const float* p = nn->trunk + ch * spatial;
      for(int i = 0; i < spatial; i++) sum += p[i];
      buf.push_back(sum / (float)spatial);
    }
  } else {
    buf.insert(buf.end(), trunkCh, 0.0f);
  }
}

// ─── Shared game replay ──────────────────────────────────────────────
//
// Single replay engine with all safety guards: consecutive-color truncation,
// move limit, null/pass handling, game-finished detection, tolerant board
// advancement.  Both the main eval pass and HumanSL pass use this, so any
// new guard added here automatically protects all passes.

struct ReplayPosition {
  Player pla;
  int    rowPos;
  size_t turn;
};

enum class ReplayAction { CONTINUE, STOP };

template<typename Visitor>
static bool replayGame(
  const CompactSgf* sgf, const Rules& rules,
  int nnXLen, int nnYLen, int maxMoves,
  Visitor& visitor
) {
  Board board;
  BoardHistory history;
  Player initialPla;
  sgf->setupInitialBoardAndHist(rules, board, initialPla, history);

  Player lastStonePla   = P_BLACK;
  bool   firstStoneSeen = false;
  int    evaluated       = 0;

  const size_t moveLimit = std::min(sgf->moves.size(), (size_t)maxMoves);
  for(size_t turn = 0; turn < moveLimit; turn++) {
    const Move& move = sgf->moves[turn];

    if(move.loc == Board::NULL_LOC) {
      history.makeBoardMoveAssumeLegal(board, move.loc, move.pla, nullptr, true);
      continue;
    }

    // Consecutive same-color = end-of-game annotation (Chinese dead-stone
    // marking, etc.).  Truncate — keep all data collected so far.
    if(firstStoneSeen && move.pla == lastStonePla) {
      cerr << "  [replay] consecutive " << (move.pla == P_BLACK ? "B" : "W")
           << " at turn " << turn << " — truncating." << endl;
      break;
    }
    lastStonePla   = move.pla;
    firstStoneSeen = true;

    if(history.isGameFinished) {
      history.makeBoardMoveAssumeLegal(board, move.loc, move.pla, nullptr, true);
      continue;
    }

    int rowPos = NNPos::locToPos(move.loc, board.x_size, nnXLen, nnYLen);
    if(rowPos < 0 || rowPos >= NNPos::MAX_NN_POLICY_SIZE) {
      if(!history.makeBoardMoveTolerant(board, move.loc, move.pla))
        break;
      continue;
    }

    ReplayPosition pos{move.pla, rowPos, turn};
    ReplayAction action = visitor(board, history, pos);
    if(action == ReplayAction::STOP) break;
    evaluated++;

    if(!history.makeBoardMoveTolerant(board, move.loc, move.pla))
      break;
  }
  return evaluated > 0;
}

// ─── HumanSL second pass ─────────────────────────────────────────────

static void runHumanSLPass(
  const CompactSgf* sgf, const Rules& rules,
  NNEvaluator* humanEval,
  const vector<int>& candB, const vector<int>& candW,
  int nnXLen, int nnYLen,
  PlayerSummary& sb, PlayerSummary& sw
) {
  float sumLogB[HUMAN_CANDIDATES] = {}, sumLogW[HUMAN_CANDIDATES] = {};
  int   cntB  [HUMAN_CANDIDATES] = {}, cntW  [HUMAN_CANDIDATES] = {};

  auto visitor = [&](const Board& board, const BoardHistory& history,
                     const ReplayPosition& pos) -> ReplayAction {
    const vector<int>& cand = (pos.pla == P_BLACK) ? candB : candW;
    float* sumLog = (pos.pla == P_BLACK) ? sumLogB : sumLogW;
    int*   cnt    = (pos.pla == P_BLACK) ? cntB    : cntW;

    for(int k = 0; k < HUMAN_CANDIDATES; k++) {
      SGFMetadata meta = SGFMetadata::getProfile(RANK_PROFILES[cand[k]]);

      NNResultBuf hbuf;
      hbuf.result = std::make_shared<NNOutput>();
      hbuf.result->includeTrunk = false;
      hbuf.result->includePick  = false;

      MiscNNInputParams hparams;
      hparams.symmetry = 0;
      hparams.policyOptimism = 0.0;

      try {
        humanEval->evaluate(board, history, pos.pla, &meta, hparams, hbuf, true, false);
      } catch(const exception& e) {
        cerr << "HumanSL evaluate exception at turn " << pos.turn
             << " candidate " << k << ": " << e.what() << endl;
        return ReplayAction::STOP;
      }

      if(hbuf.result) {
        float prob = hbuf.result->policyProbs[pos.rowPos];
        sumLog[k] += std::log(std::max(prob, 1e-10f));
        cnt[k]++;
      }
    }
    return ReplayAction::CONTINUE;
  };

  replayGame(sgf, rules, nnXLen, nnYLen, MAX_HUMAN_MOVES, visitor);

  auto selectBest = [](const float sl[], const int ct[],
                       const vector<int>& cand) -> pair<int,float> {
    int   bestK    = -1;
    float bestMean = -1e20f;
    for(int k = 0; k < HUMAN_CANDIDATES; k++) {
      if(ct[k] > 0) {
        float m = sl[k] / ct[k];
        if(m > bestMean) { bestMean = m; bestK = k; }
      }
    }
    if(bestK < 0) return {-1, 0.0f};
    return {cand[bestK], bestMean};
  };

  auto [idxB, meanB] = selectBest(sumLogB, cntB, candB);
  auto [idxW, meanW] = selectBest(sumLogW, cntW, candW);

  sb.humanRankIdx  = (float)idxB;
  sb.humanLogPrior = meanB;
  sw.humanRankIdx  = (float)idxW;
  sw.humanLogPrior = meanW;
}

// ═════════════════════════════════════════════════════════════════════
//  batch_analysis — Main entry
// ═════════════════════════════════════════════════════════════════════

int batch_analysis(const vector<string>& args) {
  string modelFile, configFile, humanModelFile;
  string listFile, sgfDir, outputDir = ".";
  int visits = 0, minMoves = 10, maxGames = 0;
  int maxBatchSize = 64;   // NN batch size for NNEvaluator; higher = larger GPU batches
  bool profile     = false; // hidden: print per-game timing breakdown
  bool streamMode = false;   // -stream: write frames to stdout instead of files
  bool noTrunk    = false;   // -no-trunk: scalars only (10 floats/move), skip trunk/pick
  bool daemonMode = false;   // -daemon: persistent process, jobs fed via stdin

  for(size_t i = 1; i < args.size(); i++) {
    if     (args[i] == "-config"      && i+1 < args.size()) configFile     = args[++i];
    else if(args[i] == "-model"       && i+1 < args.size()) modelFile      = args[++i];
    else if(args[i] == "-human-model" && i+1 < args.size()) humanModelFile = args[++i];
    else if(args[i] == "-list"        && i+1 < args.size()) listFile       = args[++i];
    else if(args[i] == "-sgf-dir"     && i+1 < args.size()) sgfDir         = args[++i];
    else if(args[i] == "-output-dir"  && i+1 < args.size()) outputDir      = args[++i];
    else if(args[i] == "-visits"      && i+1 < args.size()) visits         = stoi(args[++i]);
    else if(args[i] == "-min-moves"   && i+1 < args.size()) minMoves       = stoi(args[++i]);
    else if(args[i] == "-max-games"   && i+1 < args.size()) maxGames       = stoi(args[++i]);
    else if(args[i] == "-batch-size"  && i+1 < args.size()) maxBatchSize   = stoi(args[++i]);
    else if(args[i] == "-stream")      streamMode = true;
    else if(args[i] == "-no-trunk")    noTrunk    = true;
    else if(args[i] == "-daemon")      daemonMode = true;
    else if(args[i] == "-profile")     profile    = true;
    else { cerr << "Unknown argument: " << args[i] << endl; return 1; }
  }

  if(modelFile.empty()) {
    cerr << "Usage: katago batch_analysis -model <model.bin.gz> [-config <cfg>]" << endl;
    cerr << "  [-list <games.csv> | -sgf-dir <dir/>] [-output-dir <out/>]" << endl;
    cerr << "  [-visits N] [-min-moves N] [-max-games N]" << endl;
    cerr << "  [-stream] [-no-trunk]" << endl;
    cerr << "  [-batch-size N (default 64)]" << endl;
    cerr << "  [-human-model <human.bin.gz>]" << endl;
    cerr << endl;
    cerr << "  -stream    Write KAB2 frames to stdout instead of files." << endl;
    cerr << "             Protocol: ['B'/'W'][uint32 idLen][game id][uint32 size][KAB2 payload]" << endl;
    cerr << "             per player, terminated by a single 0x00 byte." << endl;
    cerr << "  -no-trunk  Scalars only (10 floats/move). Omits trunk/pick vectors." << endl;
    cerr << "             Useful for fast rank assessment without training features." << endl;
    cerr << "  -daemon    Persistent mode: keep models loaded, read job lines from" << endl;
    cerr << "             stdin (each line = path to a games.csv; 'quit' exits)." << endl;
    cerr << "             Implies -stream; each job's frames end with a 0x01 byte." << endl;
    return 1;
  }

  // Daemon mode delivers results as stream frames; file output is not supported.
  if(daemonMode) streamMode = true;

  vector<ListEntry> entries;
  if(!daemonMode) {
    if(!listFile.empty())
      entries = parseList(listFile);
    if(entries.empty() && !sgfDir.empty()) {
      vector<string> sgfFiles;
      FileUtils::collectFiles(sgfDir,
        [](const string& n){ return n.size() >= 4 && n.substr(n.size()-4) == ".sgf"; },
        sgfFiles);
      for(const auto& f : sgfFiles) { ListEntry e; e.sgfPath = f; entries.push_back(e); }
    }
    if(entries.empty()) { cerr << "No games found (provide -list or -sgf-dir)" << endl; return 1; }
    if(maxGames > 0 && (int)entries.size() > maxGames) entries.resize(maxGames);
  }

  if(!streamMode) ensureDir(outputDir);

  Logger logger;
  Rand seedRand;
  ConfigParser cfg;
  if(!configFile.empty())
    cfg.initialize(configFile);
  Setup::initializeSession(cfg);
  Board::initHash();

  int nnXLen = 19, nnYLen = 19;
  auto nnEval = Setup::initializeNNEvaluator(
    modelFile, modelFile, "", cfg, logger, seedRand, maxBatchSize,
    nnXLen, nnYLen, 64, true, false, Setup::SETUP_FOR_ANALYSIS
  );
  if(!nnEval) { cerr << "Failed to load model: " << modelFile << endl; return 1; }
  nnXLen = nnEval->getNNXLen();
  nnYLen = nnEval->getNNYLen();
  const int trunkChFull = nnEval->getModelTrunkNumChannels();
  const int trunkCh     = noTrunk ? 0 : trunkChFull;  // 0 = scalars-only mode
  const int moveDim     = SCALAR_DIM + 2 * trunkCh;

  // Optional HumanSL evaluator
  NNEvaluator* humanEval = nullptr;
  if(!humanModelFile.empty()) {
    humanEval = Setup::initializeNNEvaluator(
      humanModelFile, humanModelFile, "", cfg, logger, seedRand, maxBatchSize,
      nnXLen, nnYLen, 64, true, false, Setup::SETUP_FOR_ANALYSIS
    );
    if(!humanEval)
      cerr << "Warning: failed to load human model: " << humanModelFile
           << " — HumanSL pass skipped." << endl;
    else if(!humanEval->requiresSGFMetadata())
      cerr << "Warning: -human-model does not appear to be a HumanSL model." << endl;
  }

  (void)visits;  // reserved for future MCTS depth control

#ifdef _WIN32
  // Binary mode for stdout — prevents CRLF conversion corrupting binary frames
  _setmode(_fileno(stdout), _O_BINARY);
#endif

  cerr << "batch-analysis: "
       << (daemonMode ? string("daemon") : std::to_string(entries.size()) + " games")
       << "  trunkCh=" << trunkChFull << "  moveDim=" << (SCALAR_DIM + 2 * trunkChFull)
       << "  headerBytes=" << sizeof(NPZHeader);
  if(humanEval) cerr << "  [+HumanSL x" << HUMAN_CANDIDATES << "]";
  if(streamMode) cerr << "  [stream mode" << (noTrunk ? " lite" : "") << "]";
  if(daemonMode) cerr << "  [daemon]";
  cerr << "  output=" << (streamMode ? "stdout" : outputDir) << endl;

  bool metaHeaderWritten = false;
  int success = 0, failed = 0, skipped = 0;

  // Profiling accumulators
  double totalNNEvalMs = 0.0;
  double totalGameMs   = 0.0;
  int     totalMoves   = 0;
  using Clock = chrono::steady_clock;

  // Process one batch of entries. Captures evaluators, counters, and mode
  // flags by reference so daemon mode can reuse it per job.
  auto processEntries = [&](const vector<ListEntry>& jobEntries) {
  for(size_t gi = 0; gi < jobEntries.size(); gi++) {
    Clock::time_point gameStart = Clock::now();
    double gameNNEvalMs = 0.0;
    const auto& entry = jobEntries[gi];
    const string base = sgfToBase(entry.sgfPath);

    try {
      auto sgf = CompactSgf::loadFile(entry.sgfPath);
      if(!sgf) { failed++; continue; }

      if((int)sgf->moves.size() < minMoves) { skipped++; continue; }

      Rules rules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());

      vector<float> data_B, data_W;
      data_B.reserve(80 * moveDim);
      data_W.reserve(80 * moveDim);
      int nBlack = 0, nWhite = 0;
      PlayerAcc accB, accW;

      // Deferred delta tracking
      float          prevWinProb    = 0.5f;
      float          prevScoreMean  = 0.0f;
      bool           hasPrevEval    = false;
      vector<float>* prevBuf        = nullptr;
      int            prevRecordIdx  = -1;
      Player         prevPla        = P_BLACK;

      auto mainVisitor = [&](const Board& board, const BoardHistory& history,
                             const ReplayPosition& pos) -> ReplayAction {
        NNResultBuf nnbuf;
        nnbuf.result = std::make_shared<NNOutput>();
        nnbuf.result->includeTrunk = !noTrunk;
        nnbuf.result->includePick  = false;

        MiscNNInputParams params;
        params.symmetry = 0;
        params.policyOptimism = 0.0;
        try {
          Clock::time_point t0 = Clock::now();
          nnEval->evaluate(board, history, pos.pla, params, nnbuf, true, false);
          gameNNEvalMs += chrono::duration<double, milli>(Clock::now() - t0).count();
        } catch(const exception& e) {
          cerr << "Evaluate exception at game " << gi << " turn " << pos.turn
               << ": " << e.what() << endl;
          return ReplayAction::STOP;
        }
        NNOutput* output = nnbuf.result.get();
        if(!output) return ReplayAction::CONTINUE;

        // ── 1. Deferred delta fill ──────────────────────────────────────
        if(hasPrevEval && prevBuf && prevRecordIdx >= 0) {
          float wD = output->whiteWinProb   - prevWinProb;
          float sD = output->whiteScoreMean - prevScoreMean;
          int base_ = prevRecordIdx * moveDim;
          (*prevBuf)[base_ + 8] = wD;
          (*prevBuf)[base_ + 9] = sD / 50.0f;
          float pWD = (prevPla == P_BLACK) ? -wD :  wD;
          float pSD = (prevPla == P_BLACK) ? -sD :  sD;
          ((prevPla == P_BLACK) ? accB : accW).addDelta(pWD, pSD);
        }

        // ── 2. Move-level stats ─────────────────────────────────────────
        float policyPrior = output->policyProbs[pos.rowPos];
        int rank = 0;
        for(int i = 0; i < NNPos::MAX_NN_POLICY_SIZE; i++)
          if(output->policyProbs[i] > policyPrior) rank++;

        float playerWinRate   = (pos.pla == P_BLACK)
                                ? 1.0f - output->whiteWinProb : output->whiteWinProb;
        float playerScoreLead = (pos.pla == P_BLACK)
                                ? -output->whiteScoreMean : output->whiteScoreMean;

        // ── 3. Update accumulator ───────────────────────────────────────
        ((pos.pla == P_BLACK) ? accB : accW).addMove(
          rank, policyPrior, playerWinRate, playerScoreLead,
          output->shorttermScoreError
        );

        // ── 4. Append feature record ────────────────────────────────────
        int curIdx = (pos.pla == P_BLACK) ? nBlack : nWhite;
        auto& dataBuf = (pos.pla == P_BLACK) ? data_B : data_W;
        appendMoveRecord(dataBuf, output, pos.pla, policyPrior, rank, pos.rowPos,
                         trunkCh, nnXLen, nnYLen);

        // ── 5. Advance deferred-delta tracking ──────────────────────────
        prevWinProb   = output->whiteWinProb;
        prevScoreMean = output->whiteScoreMean;
        hasPrevEval   = true;
        prevBuf       = &dataBuf;
        prevRecordIdx = curIdx;
        prevPla       = pos.pla;

        if(pos.pla == P_BLACK) nBlack++; else nWhite++;
        return ReplayAction::CONTINUE;
      };

      replayGame(sgf.get(), rules, nnXLen, nnYLen, MAX_GAME_MOVES, mainVisitor);

      // ── PlayerSummary from first pass ────────────────────────────────────
      PlayerSummary sb = accB.toSummary();
      PlayerSummary sw = accW.toSummary();

      // ── HumanSL second pass (optional) ───────────────────────────────────
      if(humanEval && nBlack >= 5 && nWhite >= 5) {
        try {
          vector<int> candB = rankCandidates(sb.meanLogPrior);
          vector<int> candW = rankCandidates(sw.meanLogPrior);
          runHumanSLPass(sgf.get(), rules, humanEval,
                         candB, candW, nnXLen, nnYLen, sb, sw);
        } catch(const exception& e) {
          cerr << "HumanSL pass failed for game " << gi << ": " << e.what() << endl;
          // sb/sw.humanRankIdx stays -1
        }
      }

      // ── Per-game profile ────────────────────────────────────────────────
      if(profile) {
        Clock::time_point gameEnd = Clock::now();
        double gameMs = chrono::duration<double, milli>(gameEnd - gameStart).count();
        totalNNEvalMs += gameNNEvalMs;
        totalGameMs   += gameMs;
        totalMoves    += (nBlack + nWhite);
        cerr << "  [profile] game " << gi << " \"" << base << "\": " << gameMs << " ms  (nnEval " << gameNNEvalMs << " ms / " << (nBlack + nWhite) << " moves = " << (gameNNEvalMs / max(1, nBlack + nWhite)) << " ms/eval)" << (humanEval ? " [+HumanSL]" : "") << endl;
      }

      // ── Output: stream to stdout or write files ───────────────────────────
      if(streamMode) {
        if(nBlack >= 5) {
          auto buf = serializePlayer(data_B, nBlack, trunkCh, nnXLen, nnYLen, false, sb);
          streamPlayer('B', base, buf);
        }
        if(nWhite >= 5) {
          auto buf = serializePlayer(data_W, nWhite, trunkCh, nnXLen, nnYLen, false, sw);
          streamPlayer('W', base, buf);
        }
      } else {
        // ── Combined KAB2 pair file (single .npz, always compressed) ───
        // Format: [4B B_size][B_KAB2_compressed][4B W_size][W_KAB2_compressed]
        auto bufB = (nBlack >= 5) ? serializePlayer(data_B, nBlack, trunkCh, nnXLen, nnYLen, true, sb)
                                  : vector<char>();
        auto bufW = (nWhite >= 5) ? serializePlayer(data_W, nWhite, trunkCh, nnXLen, nnYLen, true, sw)
                                  : vector<char>();
        string outPath = outputDir + "/" + base + ".npz";
        ofstream ofs(outPath, ios::binary);
        if(ofs) {
          uint32_t bSz = (uint32_t)bufB.size();
          uint32_t wSz = (uint32_t)bufW.size();
          ofs.write(reinterpret_cast<const char*>(&bSz), 4);
          if(bSz > 0) ofs.write(bufB.data(), bSz);
          ofs.write(reinterpret_cast<const char*>(&wSz), 4);
          if(wSz > 0) ofs.write(bufW.data(), wSz);
        }
      }

      // ── Meta CSV (file mode only — stream mode must not touch disk) ──────
      if(!streamMode) {
      if(!metaHeaderWritten) {
        ofstream hdr(outputDir + "/_meta.csv");
        if(hdr)
          hdr << "file,black,white,black_elo,white_elo,total_moves,black_moves,white_moves,set,"
                 "B_acc1,B_acc3,B_logPrior,B_winRate,B_scoreLead,B_complexity,B_scoreVar,B_drop,"
                 "B_humanRank,B_humanLogPrior,"
                 "W_acc1,W_acc3,W_logPrior,W_winRate,W_scoreLead,W_complexity,W_scoreVar,W_drop,"
                 "W_humanRank,W_humanLogPrior\n";
        metaHeaderWritten = true;
      }
      {
        ofstream meta(outputDir + "/_meta.csv", ios::app);
        if(meta)
          meta << base << "," << entry.playerBlack << "," << entry.playerWhite << ","
               << entry.blackElo << "," << entry.whiteElo << ","
               << sgf->moves.size() << "," << nBlack << "," << nWhite << "," << entry.set << ","
               << sb.accuracy1 << "," << sb.accuracy3 << "," << sb.meanLogPrior << ","
               << sb.meanWinRate << "," << sb.meanScoreLead << "," << sb.meanComplexity << ","
               << sb.scoreVariance << "," << sb.approxScoreDrop << ","
               << rankIdxToStr(sb.humanRankIdx) << "," << sb.humanLogPrior << ","
               << sw.accuracy1 << "," << sw.accuracy3 << "," << sw.meanLogPrior << ","
               << sw.meanWinRate << "," << sw.meanScoreLead << "," << sw.meanComplexity << ","
               << sw.scoreVariance << "," << sw.approxScoreDrop << ","
               << rankIdxToStr(sw.humanRankIdx) << "," << sw.humanLogPrior << "\n";
      }
      }

      success++;
    } catch(const exception& e) {
      cerr << "Error on game " << gi << " (" << entry.sgfPath << "): " << e.what() << endl;
      failed++;
    } catch(...) {
      cerr << "Unknown error on game " << gi << " (" << entry.sgfPath << ") — skipping." << endl;
      failed++;
    }

    if((gi+1) % 10 == 0 || gi == jobEntries.size()-1) {
      cerr << "  [" << (gi+1) << "/" << jobEntries.size() << "]"
           << "  ok=" << success << " skip=" << skipped << " fail=" << failed << endl;
    }
  }
  };  // processEntries

  if(daemonMode) {
    // Persistent loop: one job per stdin line (path to a games.csv).
    // Each job's frames are followed by a 0x01 marker; a final 0x00
    // terminator is written when the daemon exits.
    cerr << "daemon: ready" << endl;
    string line;
    while(getline(cin, line)) {
      while(!line.empty() && (line.back() == '\r' || line.back() == ' '))
        line.pop_back();
      if(line.empty()) continue;
      if(line == "quit") break;

      // Soft reset: clear NN caches and counters, keep models loaded.
      // Acknowledged with the same 0x01 marker as a job.
      if(line == "reset") {
        nnEval->clearCache();
        if(humanEval) humanEval->clearCache();
        success = failed = skipped = 0;
        totalNNEvalMs = totalGameMs = 0.0;
        totalMoves = 0;
        char jobDone = 1;
        cout.write(&jobDone, 1);
        cout.flush();
        cerr << "daemon: reset done" << endl;
        continue;
      }

      vector<ListEntry> jobEntries = parseList(line);
      if(jobEntries.empty())
        cerr << "daemon: no entries in job list: " << line << endl;
      else
        processEntries(jobEntries);

      char jobDone = 1;
      cout.write(&jobDone, 1);
      cout.flush();
      cerr << "daemon: job done (" << jobEntries.size() << " entries)" << endl;
    }
  } else {
    processEntries(entries);
  }

  if(streamMode) {
    // Terminator: single 0x00 byte signals end-of-stream to Python reader
    char terminator = 0;
    cout.write(&terminator, 1);
    cout.flush();
  }

  cerr << "\n=== batch-analysis complete ===" << endl;
  cerr << "  total=" << (success + skipped + failed)
       << "  ok=" << success << "  skipped(<" << minMoves << "mv)=" << skipped
       << "  failed=" << failed << endl;
  cerr << "  format: KAB2  scalarDim=" << SCALAR_DIM
       << "  trunkCh=" << trunkCh
       << (noTrunk ? " [lite/no-trunk]" : "") << "  moveDim=" << moveDim << endl;
  if(!streamMode) cerr << "  output: " << outputDir << "/" << endl;
  return 0;
}

}  // namespace MainCmds
