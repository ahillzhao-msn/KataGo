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
//
// Evaluation: each pass replays the game into position snapshots, evaluates them
// on -threads client threads (the NNEvaluator batches concurrent rows), then
// folds the results in move order, so output is identical to a serial run.

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
#include <atomic>
#include <chrono>
#include <thread>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

using namespace std;

namespace MainCmds {

static const int SCALAR_DIM       = 10;
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

// ─── HumanSL rank anchor: coarse + refine scan ────────────────────────
//
// The anchor is a fixed procedure, identical for every player and game:
//   1. gate: the main model scores every move; per player, only moves whose point loss
//      lies inside that player's 10–90% quantile band for the game vote (see
//      humanVotingMoves) — the main model is the shared yardstick, HumanSL the ruler;
//   2. total HumanSL log-likelihood of the voting moves under 5 fixed profiles;
//   3. a natural cubic spline over those totals gives a posterior over all 29 profiles
//      (flat prior);
//   4. label = posterior mean rank (continuous), with posterior sd as its spread.
// Work ≈ 14% of an all-moves x 29-profile scan. On 160 players (80 games) its split-half
// stability matched the full scan's (r 0.78 vs 0.78, bootstrap se 0.04); 4 profiles was
// the measured floor, 3 profiles collapsed (r 0.70).
static const int HUMAN_PROFILES[] = {0, 7, 14, 21, 28};  // 20k, 13k, 6k, 2d, 9d
static const double HUMAN_LOSS_BAND_LO = 0.10;   // see humanVotingMoves
static const double HUMAN_LOSS_BAND_HI = 0.90;
static const float HUMAN_ANCHOR_VERSION = 2.0f;  // 1 = legacy 3-profile window

// Natural cubic spline through (xs[i], ys[i]) (xs strictly increasing), evaluated at
// each integer profile index 0..NUM_RANK_PROFILES-1.
static vector<double> splineAtProfiles(const vector<int>& xs, const vector<double>& ys) {
  int n = (int)xs.size();
  vector<double> m(n, 0.0);  // second derivatives; natural: m[0] = m[n-1] = 0
  if(n >= 3) {
    vector<double> c(n, 0.0), d(n, 0.0);
    for(int i = 1; i < n - 1; i++) {
      double h0 = xs[i] - xs[i - 1], h1 = xs[i + 1] - xs[i];
      double a = h0, b = 2.0 * (h0 + h1), cc = h1;
      double r = 6.0 * ((ys[i + 1] - ys[i]) / h1 - (ys[i] - ys[i - 1]) / h0);
      double den = b - a * c[i - 1];
      c[i] = cc / den;
      d[i] = (r - a * d[i - 1]) / den;
    }
    for(int i = n - 2; i >= 1; i--) m[i] = d[i] - c[i] * m[i + 1];
  }
  vector<double> out(NUM_RANK_PROFILES);
  int seg = 0;
  for(int x = 0; x < NUM_RANK_PROFILES; x++) {
    while(seg < n - 2 && x > xs[seg + 1]) seg++;
    double h = xs[seg + 1] - xs[seg];
    double t1 = xs[seg + 1] - x, t0 = x - xs[seg];
    out[x] = m[seg] * t1 * t1 * t1 / (6 * h) + m[seg + 1] * t0 * t0 * t0 / (6 * h)
           + (ys[seg] / h - m[seg] * h / 6) * t1 + (ys[seg + 1] / h - m[seg + 1] * h / 6) * t0;
  }
  return out;
}

// Posterior (flat prior) over profiles from per-profile total log-likelihoods.
static void posteriorMeanSd(const vector<double>& tot, double& mean, double& sd) {
  double mx = *std::max_element(tot.begin(), tot.end());
  double z = 0, s1 = 0, s2 = 0;
  for(int i = 0; i < (int)tot.size(); i++) {
    double w = std::exp(tot[i] - mx);
    z += w; s1 += w * i; s2 += w * i * i;
  }
  mean = s1 / z;
  sd = std::sqrt(std::max(0.0, s2 / z - mean * mean));
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
  float humanRankIdx;     // HumanSL anchor rank, rounded (0=20k…28=9d), -1 if not used
  float humanLogPrior;    // mean HumanSL log-likelihood under the best evaluated profile
  float humanRankExp;     // HumanSL anchor rank, continuous posterior mean (0..28)
  float humanRankSd;      // posterior sd of the anchor (ranks)
  float humanProfiles;    // number of profiles evaluated for this player
  float humanAnchorVer;   // anchor method version (HUMAN_ANCHOR_VERSION); 0 = absent
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

// ─── Position snapshots + parallel evaluation ────────────────────────
//
// Positions within a game are independent NN queries, so each pass first
// replays the game into snapshots, evaluates them concurrently (the
// NNEvaluator forms GPU batches from rows queued by concurrent clients), and
// then folds the results strictly in move order. Folding stops at the first
// position whose evaluation threw, matching the old serial STOP semantics.

struct PositionSnap {
  Board board;
  BoardHistory history;
  ReplayPosition pos;
};

static vector<PositionSnap> snapshotGame(
  const CompactSgf* sgf, const Rules& rules, int nnXLen, int nnYLen, int maxMoves
) {
  vector<PositionSnap> snaps;
  auto collect = [&](const Board& board, const BoardHistory& history,
                     const ReplayPosition& pos) -> ReplayAction {
    snaps.push_back(PositionSnap{board, history, pos});
    return ReplayAction::CONTINUE;
  };
  replayGame(sgf, rules, nnXLen, nnYLen, maxMoves, collect);
  return snaps;
}

// Run fn(i) for i in [0, n) on up to numThreads threads. fn must not throw.
template<typename Fn>
static void parallelFor(int n, int numThreads, Fn&& fn) {
  int nt = std::min(numThreads, n);
  if(nt <= 1) {
    for(int i = 0; i < n; i++) fn(i);
    return;
  }
  std::atomic<int> next(0);
  auto work = [&]() {
    for(int i = next.fetch_add(1); i < n; i = next.fetch_add(1))
      fn(i);
  };
  vector<std::thread> threads;
  threads.reserve(nt - 1);
  for(int t = 0; t < nt - 1; t++) threads.emplace_back(work);
  work();
  for(auto& th : threads) th.join();
}

// ─── HumanSL second pass ─────────────────────────────────────────────

// Total HumanSL log-likelihood of each side's moves under the requested profiles.
// sel: snapshot indices to include (ascending). profB/profW: profile indices to
// evaluate for Black's / White's moves. Returns false (after logging) if any
// evaluation threw.
static bool humanLogLik(
  const vector<PositionSnap>& snaps, const vector<int>& sel,
  NNEvaluator* humanEval,
  const vector<int>& profB, const vector<int>& profW,
  int numThreads,
  vector<double>& totB, vector<double>& totW, int& cntB, int& cntW
) {
  // Work items: (position, k-th profile of that position's side), flattened.
  const int nPos = (int)sel.size();
  vector<size_t> start(nPos + 1, 0);
  for(int i = 0; i < nPos; i++)
    start[i + 1] = start[i] + ((snaps[sel[i]].pos.pla == P_BLACK) ? profB.size() : profW.size());

  struct HumanEval { float prob = 0.0f; bool hasResult = false; string error; };
  vector<HumanEval> evals(start[nPos]);

  parallelFor((int)evals.size(), numThreads, [&](int idx) {
    int i = (int)(std::upper_bound(start.begin(), start.end(), (size_t)idx) - start.begin()) - 1;
    const PositionSnap& snap = snaps[sel[i]];
    const vector<int>& prof = (snap.pos.pla == P_BLACK) ? profB : profW;
    SGFMetadata meta = SGFMetadata::getProfile(RANK_PROFILES[prof[idx - start[i]]]);

    NNResultBuf hbuf;
    hbuf.includeTrunk = false;

    MiscNNInputParams hparams;
    hparams.symmetry = 0;
    hparams.policyOptimism = 0.0;

    HumanEval& out = evals[idx];
    try {
      humanEval->evaluate(snap.board, snap.history, snap.pos.pla, &meta, hparams, hbuf, true, false);
      if(hbuf.result) {
        out.prob = hbuf.result->policyProbs[snap.pos.rowPos];
        out.hasResult = true;
      }
    } catch(const exception& e) {
      out.error = e.what();
      if(out.error.empty()) out.error = "unknown error";
    }
  });

  totB.assign(profB.size(), 0.0);
  totW.assign(profW.size(), 0.0);
  cntB = cntW = 0;
  // Sum in move order so the result does not depend on thread scheduling.
  for(int i = 0; i < nPos; i++) {
    bool black = snaps[sel[i]].pos.pla == P_BLACK;
    vector<double>& tot = black ? totB : totW;
    bool any = false;
    for(size_t k = 0; k < tot.size(); k++) {
      const HumanEval& ev = evals[start[i] + k];
      if(!ev.error.empty()) {
        cerr << "HumanSL evaluate exception at turn " << snaps[sel[i]].pos.turn
             << " profile " << (black ? profB : profW)[k] << ": " << ev.error << endl;
        return false;
      }
      if(!ev.hasResult) continue;
      tot[k] += std::log(std::max((double)ev.prob, 1e-10));
      any = true;
    }
    if(any) (black ? cntB : cntW)++;
  }
  return true;
}

// Linear-interpolated quantile of v (numpy's default method). v non-empty.
static float quantileLinear(vector<float> v, double q) {
  std::sort(v.begin(), v.end());
  double h = (v.size() - 1) * q;
  size_t lo = (size_t)std::floor(h);
  if(lo + 1 >= v.size()) return v.back();
  return (float)(v[lo] + (h - lo) * (v[lo + 1] - v[lo]));
}

// Moves that vote in the HumanSL anchor: within the first MAX_HUMAN_MOVES turns, and,
// per side, whose main-model point loss lies inside that player's
// [HUMAN_LOSS_BAND_LO, HUMAN_LOSS_BAND_HI] quantile band for this game. The main model
// is the shared yardstick: it gates out each player's exceptional best and worst moves,
// so the label reflects the player's typical play.
static vector<int> humanVotingMoves(const vector<PositionSnap>& snaps, const vector<float>& moverLoss) {
  vector<int> idx[2];
  for(int i = 0; i < (int)snaps.size() && snaps[i].pos.turn < (size_t)MAX_HUMAN_MOVES; i++)
    idx[snaps[i].pos.pla == P_BLACK ? 0 : 1].push_back(i);
  vector<int> sel;
  for(auto& side : idx) {
    if(side.empty()) continue;
    vector<float> v;
    for(int i : side) v.push_back(moverLoss[i]);
    float lo = quantileLinear(v, HUMAN_LOSS_BAND_LO), hi = quantileLinear(v, HUMAN_LOSS_BAND_HI);
    for(int i : side)
      if(moverLoss[i] >= lo && moverLoss[i] <= hi) sel.push_back(i);
  }
  std::sort(sel.begin(), sel.end());
  return sel;
}

// HumanSL rank anchor (loss-band gate + fixed profiles + spline posterior) for both players.
// moverLoss[i]: points the mover of snapshot i gave away per the main model (0 if unknown).
static void runHumanSLPass(
  const vector<PositionSnap>& snaps,
  const vector<float>& moverLoss,
  NNEvaluator* humanEval,
  int numThreads,
  PlayerSummary& sb, PlayerSummary& sw
) {
  const vector<int> sel = humanVotingMoves(snaps, moverLoss);
  const vector<int> prof(std::begin(HUMAN_PROFILES), std::end(HUMAN_PROFILES));
  vector<double> tB, tW;
  int nB, nW;
  if(!humanLogLik(snaps, sel, humanEval, prof, prof, numThreads, tB, tW, nB, nW))
    return;

  auto finish = [&](const vector<double>& tot, int cnt, PlayerSummary& s) {
    if(cnt == 0) return;
    double mean, sd;
    posteriorMeanSd(splineAtProfiles(prof, tot), mean, sd);
    s.humanRankIdx   = (float)std::lround(mean);
    s.humanLogPrior  = (float)(*std::max_element(tot.begin(), tot.end()) / cnt);
    s.humanRankExp   = (float)mean;
    s.humanRankSd    = (float)sd;
    s.humanProfiles  = (float)prof.size();
    s.humanAnchorVer = HUMAN_ANCHOR_VERSION;
  };
  finish(tB, nB, sb);
  finish(tW, nW, sw);
}

// ═════════════════════════════════════════════════════════════════════
//  batch_analysis — Main entry
// ═════════════════════════════════════════════════════════════════════

int batch_analysis(const vector<string>& args) {
  string modelFile, configFile, humanModelFile;
  string listFile, sgfDir, outputDir = ".";
  int visits = 0, minMoves = 10, maxGames = 0;
  int maxBatchSize = 64;   // NN batch size for NNEvaluator; higher = larger GPU batches
  int numThreads   = 16;   // client threads evaluating positions concurrently
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
    else if(args[i] == "-threads"     && i+1 < args.size()) numThreads     = std::max(1, stoi(args[++i]));
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
    cerr << "  [-batch-size N (default 64)] [-threads N (default 16)]" << endl;
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
       << "  headerBytes=" << sizeof(NPZHeader) << "  threads=" << numThreads;
  if(humanEval) cerr << "  [+HumanSL x" << (int)(sizeof(HUMAN_PROFILES) / sizeof(int)) << " profiles]";
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
      int            prevSnapIdx    = -1;

      // ── Evaluate every position concurrently ────────────────────────────
      vector<PositionSnap> snaps = snapshotGame(sgf.get(), rules, nnXLen, nnYLen, MAX_GAME_MOVES);
      vector<float> moverLoss(snaps.size(), 0.0f);  // main-model points given away, per move

      struct MainEval {
        bool  hasResult = false;
        string error;
        float whiteWinProb = 0.0f, whiteScoreMean = 0.0f, complexity = 0.0f;
        float policyPrior = 0.0f;
        int   rank = 0;
        vector<float> record;   // one appendMoveRecord block (moveDim floats)
      };
      vector<MainEval> evals(snaps.size());

      Clock::time_point evalStart = Clock::now();
      parallelFor((int)snaps.size(), numThreads, [&](int i) {
        const PositionSnap& snap = snaps[i];
        NNResultBuf nnbuf;
        nnbuf.includeTrunk = !noTrunk;

        MiscNNInputParams params;
        params.symmetry = 0;
        params.policyOptimism = 0.0;

        MainEval& ev = evals[i];
        try {
          nnEval->evaluate(snap.board, snap.history, snap.pos.pla, params, nnbuf, true, false);
        } catch(const exception& e) {
          ev.error = e.what();
          if(ev.error.empty()) ev.error = "unknown error";
          return;
        }
        const NNOutput* output = nnbuf.result.get();
        if(!output) return;

        ev.hasResult      = true;
        ev.whiteWinProb   = output->whiteWinProb;
        ev.whiteScoreMean = output->whiteScoreMean;
        ev.complexity     = output->shorttermScoreError;
        ev.policyPrior    = output->policyProbs[snap.pos.rowPos];
        for(int p = 0; p < NNPos::MAX_NN_POLICY_SIZE; p++)
          if(output->policyProbs[p] > ev.policyPrior) ev.rank++;
        ev.record.reserve(moveDim);
        appendMoveRecord(ev.record, output, snap.pos.pla, ev.policyPrior, ev.rank,
                         snap.pos.rowPos, trunkCh, nnXLen, nnYLen);
      });
      gameNNEvalMs += chrono::duration<double, milli>(Clock::now() - evalStart).count();

      // ── Fold results in move order ──────────────────────────────────────
      for(size_t i = 0; i < snaps.size(); i++) {
        const ReplayPosition& pos = snaps[i].pos;
        const MainEval& ev = evals[i];
        if(!ev.error.empty()) {
          cerr << "Evaluate exception at game " << gi << " turn " << pos.turn
               << ": " << ev.error << endl;
          break;
        }
        if(!ev.hasResult) continue;

        // ── 1. Deferred delta fill ──────────────────────────────────────
        if(hasPrevEval && prevBuf && prevRecordIdx >= 0) {
          float wD = ev.whiteWinProb   - prevWinProb;
          float sD = ev.whiteScoreMean - prevScoreMean;
          int base_ = prevRecordIdx * moveDim;
          (*prevBuf)[base_ + 8] = wD;
          (*prevBuf)[base_ + 9] = sD / 50.0f;
          float pWD = (prevPla == P_BLACK) ? -wD :  wD;
          float pSD = (prevPla == P_BLACK) ? -sD :  sD;
          moverLoss[prevSnapIdx] = -pSD;
          ((prevPla == P_BLACK) ? accB : accW).addDelta(pWD, pSD);
        }

        // ── 2. Move-level stats ─────────────────────────────────────────
        float playerWinRate   = (pos.pla == P_BLACK)
                                ? 1.0f - ev.whiteWinProb : ev.whiteWinProb;
        float playerScoreLead = (pos.pla == P_BLACK)
                                ? -ev.whiteScoreMean : ev.whiteScoreMean;

        // ── 3. Update accumulator ───────────────────────────────────────
        ((pos.pla == P_BLACK) ? accB : accW).addMove(
          ev.rank, ev.policyPrior, playerWinRate, playerScoreLead, ev.complexity
        );

        // ── 4. Append feature record ────────────────────────────────────
        int curIdx = (pos.pla == P_BLACK) ? nBlack : nWhite;
        auto& dataBuf = (pos.pla == P_BLACK) ? data_B : data_W;
        dataBuf.insert(dataBuf.end(), ev.record.begin(), ev.record.end());

        // ── 5. Advance deferred-delta tracking ──────────────────────────
        prevWinProb   = ev.whiteWinProb;
        prevScoreMean = ev.whiteScoreMean;
        hasPrevEval   = true;
        prevBuf       = &dataBuf;
        prevRecordIdx = curIdx;
        prevPla       = pos.pla;
        prevSnapIdx   = (int)i;

        if(pos.pla == P_BLACK) nBlack++; else nWhite++;
      }

      // ── PlayerSummary from first pass ────────────────────────────────────
      PlayerSummary sb = accB.toSummary();
      PlayerSummary sw = accW.toSummary();

      // ── HumanSL second pass (optional) ───────────────────────────────────
      if(humanEval && nBlack >= 5 && nWhite >= 5) {
        try {
          runHumanSLPass(snaps, moverLoss, humanEval, numThreads, sb, sw);
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
                 "W_humanRank,W_humanLogPrior,"
                 "B_humanRankExp,B_humanRankSd,W_humanRankExp,W_humanRankSd,humanAnchorVer\n";
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
               << rankIdxToStr(sw.humanRankIdx) << "," << sw.humanLogPrior << ","
               << sb.humanRankExp << "," << sb.humanRankSd << ","
               << sw.humanRankExp << "," << sw.humanRankSd << ","
               << std::max(sb.humanAnchorVer, sw.humanAnchorVer) << "\n";
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
