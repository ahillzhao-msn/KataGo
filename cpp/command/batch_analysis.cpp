// batch_analysis.cpp — Batch SGF analysis with per-move + per-game feature extraction
//
// Output format KAB2 (per-player files _B.npz / _W.npz):
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
#include <direct.h>
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

using namespace std;

namespace MainCmds {

static const int SCALAR_DIM       = 10;
static const int HUMAN_CANDIDATES = 3;   // rank profiles tested per player per game

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

static void ensureDir(const string& path) { _mkdir(path.c_str()); }

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

static void writePlayerFile(
  const string& path, const vector<float>& movesData,
  int numMoves, int trunkCh, int nnXLen, int nnYLen,
  bool doCompress, const PlayerSummary& summary
) {
  if(numMoves < 1) return;
  ofstream out(path, ios::binary);
  if(!out) { cerr << "Cannot write: " << path << endl; return; }

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
  out.write((const char*)&hdr, sizeof(hdr));

  const char* raw  = (const char*)movesData.data();
  size_t      rawB = movesData.size() * sizeof(float);
  if(doCompress) {
    vector<char> comp = zlibCompress(raw, rawB);
    int32_t clen = (int32_t)comp.size();
    out.write((const char*)&clen, 4);
    out.write(comp.data(), clen);
  } else {
    out.write(raw, rawB);
  }
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

// ─── HumanSL second pass ─────────────────────────────────────────────
//
// Re-replays all moves in the game and queries humanEval with 3 candidate
// rank profiles for each player.  Updates sb.humanRankIdx / humanLogPrior.

static void runHumanSLPass(
  const CompactSgf* sgf, const Rules& rules,
  NNEvaluator* humanEval,
  const vector<int>& candB, const vector<int>& candW,
  int nnXLen, int nnYLen,
  PlayerSummary& sb, PlayerSummary& sw
) {
  Board board;
  BoardHistory history;
  Player initialPla;
  sgf->setupInitialBoardAndHist(rules, board, initialPla, history);

  float sumLogB[HUMAN_CANDIDATES] = {}, sumLogW[HUMAN_CANDIDATES] = {};
  int   cntB  [HUMAN_CANDIDATES] = {}, cntW  [HUMAN_CANDIDATES] = {};

  for(const Move& move : sgf->moves) {
    if(move.loc == Board::NULL_LOC) {
      history.makeBoardMoveAssumeLegal(board, move.loc, move.pla, nullptr, true);
      continue;
    }

    Player evalPla = move.pla;
    int rowPos = NNPos::locToPos(move.loc, board.x_size, nnXLen, nnYLen);

    const vector<int>& cand = (evalPla == P_BLACK) ? candB : candW;
    float* sumLog = (evalPla == P_BLACK) ? sumLogB : sumLogW;
    int*   cnt    = (evalPla == P_BLACK) ? cntB    : cntW;

    for(int k = 0; k < HUMAN_CANDIDATES; k++) {
      SGFMetadata meta = SGFMetadata::getProfile(RANK_PROFILES[cand[k]]);

      NNResultBuf hbuf;
      hbuf.result = std::make_shared<NNOutput>();
      hbuf.result->includeTrunk = false;  // policy head only for HumanSL
      hbuf.result->includePick  = false;

      MiscNNInputParams hparams;
      hparams.symmetry = 0;
      hparams.policyOptimism = 0.0;

      humanEval->evaluate(board, history, evalPla, &meta, hparams, hbuf, true, false);

      if(hbuf.result) {
        float prob = (rowPos >= 0 && rowPos < NNPos::MAX_NN_POLICY_SIZE)
                     ? hbuf.result->policyProbs[rowPos] : 0.0f;
        sumLog[k] += std::log(std::max(prob, 1e-10f));
        cnt[k]++;
      }
    }

    history.makeBoardMoveAssumeLegal(board, move.loc, move.pla, nullptr, true);
  }

  // Pick best candidate for each player
  auto selectBest = [](const float sl[], const int ct[],
                       const vector<int>& cand) -> pair<int,float> {
    int   bestK    = 0;
    float bestMean = -1e20f;
    for(int k = 0; k < HUMAN_CANDIDATES; k++) {
      if(ct[k] > 0) {
        float m = sl[k] / ct[k];
        if(m > bestMean) { bestMean = m; bestK = k; }
      }
    }
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
  bool noCompress = false;

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
    else if(args[i] == "-no-compress") noCompress = true;
    else { cerr << "Unknown argument: " << args[i] << endl; return 1; }
  }

  if(modelFile.empty()) {
    cerr << "Usage: katago batch_analysis -model <model.bin.gz> [-config <cfg>]" << endl;
    cerr << "  [-list <games.csv> | -sgf-dir <dir/>] [-output-dir <out/>]" << endl;
    cerr << "  [-visits N] [-min-moves N] [-max-games N] [-no-compress]" << endl;
    cerr << "  [-human-model <human.bin.gz>]" << endl;
    return 1;
  }

  vector<ListEntry> entries;
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

  ensureDir(outputDir);

  Logger logger;
  Rand seedRand;
  ConfigParser cfg = configFile.empty() ? ConfigParser() : ConfigParser(configFile);
  Setup::initializeSession(cfg);
  Board::initHash();

  int nnXLen = 19, nnYLen = 19;
  auto nnEval = Setup::initializeNNEvaluator(
    modelFile, modelFile, "", cfg, logger, seedRand, 64,
    nnXLen, nnYLen, 64, true, false, Setup::SETUP_FOR_ANALYSIS
  );
  if(!nnEval) { cerr << "Failed to load model: " << modelFile << endl; return 1; }
  nnXLen = nnEval->getNNXLen();
  nnYLen = nnEval->getNNYLen();
  const int trunkCh = nnEval->getModelTrunkNumChannels();
  const int moveDim = SCALAR_DIM + 2 * trunkCh;

  // Optional HumanSL evaluator
  NNEvaluator* humanEval = nullptr;
  if(!humanModelFile.empty()) {
    humanEval = Setup::initializeNNEvaluator(
      humanModelFile, humanModelFile, "", cfg, logger, seedRand, 64,
      nnXLen, nnYLen, 64, true, false, Setup::SETUP_FOR_ANALYSIS
    );
    if(!humanEval)
      cerr << "Warning: failed to load human model: " << humanModelFile
           << " — HumanSL pass skipped." << endl;
    else if(!humanEval->requiresSGFMetadata())
      cerr << "Warning: -human-model does not appear to be a HumanSL model." << endl;
  }

  (void)visits;  // reserved for future MCTS depth control

  cout << "batch-analysis: " << entries.size() << " games"
       << "  trunkCh=" << trunkCh << "  moveDim=" << moveDim
       << "  headerBytes=" << sizeof(NPZHeader);
  if(humanEval) cout << "  [+HumanSL x" << HUMAN_CANDIDATES << "]";
  cout << "  output=" << outputDir << endl;

  bool metaHeaderWritten = false;
  int success = 0, failed = 0, skipped = 0;

  for(size_t gi = 0; gi < entries.size(); gi++) {
    const auto& entry = entries[gi];
    const string base = "game_" + Global::uint64ToHexString(gi);

    try {
      auto sgf = CompactSgf::loadFile(entry.sgfPath);
      if(!sgf) { failed++; continue; }

      auto& moves = sgf->moves;
      if((int)moves.size() < minMoves) { skipped++; continue; }

      Rules rules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
      Board board;
      BoardHistory history;
      Player initialPla;
      sgf->setupInitialBoardAndHist(rules, board, initialPla, history);

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

      for(size_t turn = 0; turn < moves.size(); turn++) {
        const Move& move = moves[turn];
        if(move.loc == Board::NULL_LOC) {
          history.makeBoardMoveAssumeLegal(board, move.loc, move.pla, nullptr, true);
          continue;
        }

        Player evalPla = move.pla;
        int rowPos = NNPos::locToPos(move.loc, board.x_size, nnXLen, nnYLen);

        NNResultBuf nnbuf;
        nnbuf.result = std::make_shared<NNOutput>();
        nnbuf.result->includeTrunk = true;
        nnbuf.result->includePick  = false;  // we extract pick from trunk directly

        MiscNNInputParams params;
        params.symmetry = 0;
        params.policyOptimism = 0.0;
        nnEval->evaluate(board, history, evalPla, params, nnbuf, true, false);
        NNOutput* output = nnbuf.result.get();

        if(output) {
          // ── 1. Fill deferred delta into previous record ──────────────────
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

          // ── 2. Move-level stats (computed once, shared by acc + record) ──
          float policyPrior = (rowPos >= 0 && rowPos < NNPos::MAX_NN_POLICY_SIZE)
                              ? output->policyProbs[rowPos] : 0.0f;
          int rank = 0;
          for(int i = 0; i < NNPos::MAX_NN_POLICY_SIZE; i++)
            if(output->policyProbs[i] > policyPrior) rank++;

          float playerWinRate   = (evalPla == P_BLACK)
                                  ? 1.0f - output->whiteWinProb
                                  : output->whiteWinProb;
          float playerScoreLead = (evalPla == P_BLACK)
                                  ? -output->whiteScoreMean
                                  :  output->whiteScoreMean;

          // ── 3. Update accumulator ─────────────────────────────────────────
          ((evalPla == P_BLACK) ? accB : accW).addMove(
            rank, policyPrior, playerWinRate, playerScoreLead,
            output->shorttermScoreError
          );

          // ── 4. Append record (delta slots pre-zeroed, filled above) ──────
          int curIdx = (evalPla == P_BLACK) ? nBlack : nWhite;
          auto& dataBuf = (evalPla == P_BLACK) ? data_B : data_W;
          appendMoveRecord(dataBuf, output, evalPla, policyPrior, rank, rowPos,
                           trunkCh, nnXLen, nnYLen);

          // ── 5. Advance deferred-delta tracking ───────────────────────────
          prevWinProb   = output->whiteWinProb;
          prevScoreMean = output->whiteScoreMean;
          hasPrevEval   = true;
          prevBuf       = &dataBuf;
          prevRecordIdx = curIdx;
          prevPla       = evalPla;

          if(evalPla == P_BLACK) nBlack++; else nWhite++;
        }

        history.makeBoardMoveAssumeLegal(board, move.loc, move.pla, nullptr, true);
      }

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

      // ── Write per-player NPZ files ────────────────────────────────────────
      if(nBlack >= 5)
        writePlayerFile(outputDir + "/" + base + "_B.npz",
                        data_B, nBlack, trunkCh, nnXLen, nnYLen, !noCompress, sb);
      if(nWhite >= 5)
        writePlayerFile(outputDir + "/" + base + "_W.npz",
                        data_W, nWhite, trunkCh, nnXLen, nnYLen, !noCompress, sw);

      // ── Meta CSV ──────────────────────────────────────────────────────────
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
               << moves.size() << "," << nBlack << "," << nWhite << "," << entry.set << ","
               << sb.accuracy1 << "," << sb.accuracy3 << "," << sb.meanLogPrior << ","
               << sb.meanWinRate << "," << sb.meanScoreLead << "," << sb.meanComplexity << ","
               << sb.scoreVariance << "," << sb.approxScoreDrop << ","
               << rankIdxToStr(sb.humanRankIdx) << "," << sb.humanLogPrior << ","
               << sw.accuracy1 << "," << sw.accuracy3 << "," << sw.meanLogPrior << ","
               << sw.meanWinRate << "," << sw.meanScoreLead << "," << sw.meanComplexity << ","
               << sw.scoreVariance << "," << sw.approxScoreDrop << ","
               << rankIdxToStr(sw.humanRankIdx) << "," << sw.humanLogPrior << "\n";
      }

      success++;
    } catch(const exception& e) {
      cerr << "Error on game " << gi << " (" << entry.sgfPath << "): " << e.what() << endl;
      failed++;
    }

    if((gi+1) % 10 == 0 || gi == entries.size()-1) {
      cout << "  [" << (gi+1) << "/" << entries.size() << "]"
           << "  ok=" << success << " skip=" << skipped << " fail=" << failed << endl;
    }
  }

  cout << "\n=== batch-analysis complete ===" << endl;
  cout << "  total=" << entries.size()
       << "  ok=" << success << "  skipped(<" << minMoves << "mv)=" << skipped
       << "  failed=" << failed << endl;
  cout << "  format: KAB2  scalarDim=" << SCALAR_DIM
       << "  trunkCh=" << trunkCh << "  moveDim=" << moveDim << endl;
  cout << "  output: " << outputDir << "/" << endl;
  return 0;
}

}  // namespace MainCmds
