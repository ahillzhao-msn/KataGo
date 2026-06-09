// batch_analysis.cpp — Batch SGF analysis with trunk/pick/head feature extraction
// Inspired by go-analyzer preprocessing pipeline (https://github.com/ahillzhao-msn/go-analyzer)
// Key patterns adopted:
//   - Main line only extraction (skip SGF branches)
//   - Skip games with < 10 moves
//   - Per-file error tolerance (catch & continue)
//   - Binary NPZ output with metadata
//   - Batch directory scanning
//   - Progress reporting

#include "batch_analysis.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <direct.h>
#include <ctime>

#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/fileutils.h"
#include "../dataio/sgf.h"
#include "../program/setup.h"
#include "../program/play.h"
#include "../command/commandline.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/modelversion.h"
#include "../main.h"

using namespace std;

namespace MainCmds {

// ─── NPZ binary format ──────────────────────────────────────────
// Compact binary format: header + interleaved float arrays
// Layout:
//   [magic:4] = "KABN" (KataGo Analysis Batch NPZ)
//   [num_moves:4][num_head:4]=12[num_trunk:4]=256[num_pick:4]=256
//   For each move: [head:48B][trunk:1024B][pick:1024B]
// Total per move: 2096 bytes

static const int HEAD_DIM = 12;
static const int TRUNK_DIM = 256;
static const int PICK_DIM = 256;
static const int BYTES_PER_MOVE = 2096;  // (12 + 256 + 256) * 4

#pragma pack(push, 1)
struct NPZHeader {
  char magic[4] = {'K','A','B','N'};
  int32_t numMoves;
  int32_t headDim = HEAD_DIM;
  int32_t trunkDim = TRUNK_DIM;
  int32_t pickDim = PICK_DIM;
};
struct MoveRecord {
  float head[HEAD_DIM];
  float trunk[TRUNK_DIM];
  float pick[PICK_DIM];
};
#pragma pack(pop)

// ─── Per-game metadata ──────────────────────────────────────────
struct GameMeta {
  string blackName;
  string whiteName;
  double blackElo = 1500;
  double whiteElo = 1500;
  string result;         // "B+R", "W+0.5", etc.
  string date;
  string source;         // filename or URL
  int numMoves = 0;
  int analyzedMoves = 0;
};

// ─── Utility ─────────────────────────────────────────────────────

static void ensureDir(const string& path) {
  _mkdir(path.c_str());
}

static string timestamp() {
  time_t t = time(nullptr);
  char buf[32];
  struct tm tm;
  localtime_s(&tm, &t);
  strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
  return string(buf);
}

// ─── SGF list parsing ───────────────────────────────────────────

struct ListEntry {
  string sgfPath;
  string playerBlack, playerWhite;
  double blackElo = 1500, whiteElo = 1500;
  double score = 0.5;
  char set = 'T';
};

static vector<ListEntry> parseList(const string& path) {
  vector<ListEntry> entries;
  ifstream in(path);
  if(!in) { cerr << "Cannot open list: " << path << endl; return entries; }
  string line;
  getline(in, line); // skip header
  while(getline(in, line)) {
    if(line.empty()) continue;
    stringstream ss(line);
    ListEntry e;
    string field;
    getline(ss, e.sgfPath, ',');
    getline(ss, e.playerBlack, ',');
    getline(ss, e.playerWhite, ',');
    getline(ss, field, ','); e.score = stod(field);
    getline(ss, field, ','); e.blackElo = stod(field);
    getline(ss, field, ','); e.whiteElo = stod(field);
    getline(ss, field, ','); e.set = field.empty() ? 'T' : field[0];
    entries.push_back(e);
  }
  return entries;
}

// ─── Extract head features from NNOutput ─────────────────────────

static void extractHead(const NNOutput* nn, int rowPos, float* head, Player evalPla) {
  // 12-dim features matching go-analyzer's AnalysisRecord format
  const float wr = nn->whiteWinProb;
  const float sl = (float)nn->whiteScoreMean;
  const float prior = (rowPos >= 0 && rowPos < NNPos::MAX_NN_POLICY_SIZE) ? nn->policyProbs[rowPos] : 0.0f;

  head[0] = 0.0f;                              // is_best (needs candidate list)
  head[1] = 0.0f;                              // is_top5 (needs candidate list)
  head[2] = 1.0f - prior;                      // complexity
  head[3] = 0.5f;                               // policy_entropy (needs full distribution)
  head[4] = prior;                               // policy_prior
  head[5] = (evalPla == P_BLACK ? wr : 1.0f - wr); // winrate (from current player's perspective)
  head[6] = (evalPla == P_BLACK ? sl : -sl) / 50.0f; // scoreLead normalized
  head[7] = 0.1f;                               // scoreStdev (placeholder)
  head[8] = 0.0f;                               // utility (needs search)
  head[9] = 0.0f;                               // lcb (needs search)
  head[10] = 0.05f;                             // visits_ratio (needs search)
  head[11] = (evalPla == P_BLACK ? 0.0f : 1.0f); // player: 0=Black, 1=White
}

// ═════════════════════════════════════════════════════════════════
//  batch_analysis — Main command
// ═════════════════════════════════════════════════════════════════

int batch_analysis(const vector<string>& args) {
  // Parse arguments
  string modelFile, configFile;
  string listFile, sgfDir;
  string outputDir = ".";
  int visits = 0;          // 0 = use config default
  int minMoves = 10;       // skip games with fewer moves (go-analyzer: 50)
  int maxGames = 0;        // 0 = all
  bool headOnly = false;   // only output 12-dim head features
  bool trunkOnly = false;  // only output 256-dim trunk features

  for(size_t i = 1; i < args.size(); i++) {
    if(args[i] == "-config" && i+1 < args.size()) { configFile = args[++i]; }
    else if(args[i] == "-model" && i+1 < args.size()) { modelFile = args[++i]; }
    else if(args[i] == "-list" && i+1 < args.size()) { listFile = args[++i]; }
    else if(args[i] == "-sgf-dir" && i+1 < args.size()) { sgfDir = args[++i]; }
    else if(args[i] == "-output-dir" && i+1 < args.size()) { outputDir = args[++i]; }
    else if(args[i] == "-visits" && i+1 < args.size()) { visits = stoi(args[++i]); }
    else if(args[i] == "-min-moves" && i+1 < args.size()) { minMoves = stoi(args[++i]); }
    else if(args[i] == "-max-games" && i+1 < args.size()) { maxGames = stoi(args[++i]); }
    else if(args[i] == "-head-only") { headOnly = true; }
    else if(args[i] == "-trunk-only") { trunkOnly = true; }
    else { cerr << "Unknown: " << args[i] << endl; return 1; }
  }

  if(modelFile.empty()) {
    cerr << "Usage: katago batch-analysis -model model.bin.gz [-list games.csv | -sgf-dir dir/] -output-dir out/" << endl;
    cerr << "  [-visits N] [-batch-size N] [-max-games N]" << endl;
    return 1;
  }

  ensureDir(outputDir);

  // Collect SGF paths
  vector<ListEntry> entries;
  if(!listFile.empty())
    entries = parseList(listFile);
  else if(!sgfDir.empty()) {
    // Simple directory listing
    // TODO: use std::filesystem for C++17 recursive scan
  }

  if(entries.empty()) {
    cerr << "No games found" << endl;
    return 1;
  }
  if(maxGames > 0 && (int)entries.size() > maxGames)
    entries.resize(maxGames);

  cout << "batch-analysis: " << entries.size() << " games"
       << ", visits=" << visits
       << ", visits=" << visits
       << ", output=" << outputDir << endl;

  // ── NN setup ──
  Logger logger;
  Rand seedRand;
  ConfigParser cfg = configFile.empty() ? ConfigParser() : ConfigParser(configFile);
  int nnXLen = 19, nnYLen = 19;
  int defaultMaxBatch = 64;  // nnMaxBatchSize in config overrides this
  auto nnEval = Setup::initializeNNEvaluator(
    modelFile, modelFile, "", cfg, logger, seedRand, 64,
    nnXLen, nnYLen, defaultMaxBatch, true, false, Setup::SETUP_FOR_ANALYSIS
  );
  if(!nnEval) { cerr << "Failed to load model" << endl; return 1; }
  nnXLen = nnEval->getNNXLen();
  nnYLen = nnEval->getNNYLen();

  const string runId = timestamp();
  int success = 0, failed = 0, skipped = 0;

  // ── Process each game ──
  for(size_t gi = 0; gi < entries.size(); gi++) {
    const auto& entry = entries[gi];
    const string base = "game_" + Global::uint64ToHexString(gi);

    // Load SGF with error tolerance
    auto sgf = CompactSgf::loadFile(entry.sgfPath);
    if(!sgf) { failed++; continue; }

    // Extract main line only (CompactSgf already skips branches)
    auto& moves = sgf->moves;
    if((int)moves.size() < minMoves) { skipped++; continue; }

    // Parse game metadata
    GameMeta meta;
    meta.blackName = entry.playerBlack;
    meta.whiteName = entry.playerWhite;
    meta.blackElo = entry.blackElo;
    meta.whiteElo = entry.whiteElo;
    meta.numMoves = (int)moves.size();

    // Play through the game
    Rules rules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
    Board board;
    BoardHistory history;
    Player initialPla;
    sgf->setupInitialBoardAndHist(rules, board, initialPla, history);

    vector<MoveRecord> blackRecords, whiteRecords;

    for(size_t turn = 0; turn < moves.size(); turn++) {
      const Move& move = moves[turn];
      if(move.loc == Board::NULL_LOC) continue;  // skip passes

      Player evalPla = move.pla;
      int rowPos = NNPos::locToPos(move.loc, board.x_size, nnXLen, nnYLen);

      // Evaluate position (before move)
      NNResultBuf buf;
      buf.result = std::make_shared<NNOutput>();
      MiscNNInputParams params;
      params.symmetry = 0;
      params.policyOptimism = 0;

      nnEval->evaluate(board, history, evalPla, params, buf, false, false);
      NNOutput* output = buf.result.get();
      if(!output) continue;

      MoveRecord rec;
      memset(&rec, 0, sizeof(rec));
      extractHead(output, rowPos, rec.head, evalPla);
      rec.head[11] = (evalPla == P_BLACK ? 0.0f : 1.0f);

      // Trunk features (spatial average pool)
      if(output->trunk) {
        int spatial = nnXLen * nnYLen;
        int ch = TRUNK_DIM;
        for(int c = 0; c < ch; c++) {
          double sum = 0;
          for(int xy = 0; xy < spatial; xy++)
            sum += output->trunk[c * spatial + xy];
          rec.trunk[c] = (float)(sum / spatial);
        }
      }

      // Pick features (at move position)
      if(output->pick && rowPos >= 0) {
        for(int c = 0; c < PICK_DIM; c++)
          rec.pick[c] = output->pick[c];
      }

      // Assign to correct player
      if(evalPla == P_BLACK) blackRecords.push_back(rec);
      else whiteRecords.push_back(rec);

      // Make the move
      history.makeBoardMoveAssumeLegal(board, move.loc, move.pla, nullptr, true);
    }

    // Write NPZ per player
    auto writeNPZ = [&](const vector<MoveRecord>& records, const string& suffix) {
      if((int)records.size() < 5) return;
      NPZHeader hdr;
      hdr.numMoves = (int32_t)records.size();
      string path = outputDir + "/" + base + suffix + ".npz";
      ofstream out(path, ios::binary);
      if(!out) return;
      out.write((const char*)&hdr, sizeof(hdr));
      out.write((const char*)records.data(), records.size() * sizeof(MoveRecord));
    };
    writeNPZ(blackRecords, "_B");
    writeNPZ(whiteRecords, "_W");

    // Write metadata CSV (one line per game)
    if(gi == 0) {
      ofstream metaOut(outputDir + "/_meta.csv");
      if(metaOut) {
        metaOut << "file,black,white,black_elo,white_elo,moves,analyzed" << endl;
        metaOut.close();
      }
    }
    ofstream metaOutApp(outputDir + "/_meta.csv", ios::app);
    if(metaOutApp) {
      metaOutApp << base << ","
                 << meta.blackName << "," << meta.whiteName << ","
                 << meta.blackElo << "," << meta.whiteElo << ","
                 << meta.numMoves << ","
                 << (blackRecords.size() + whiteRecords.size()) << endl;
    }

    success++;

    // Progress report (go-analyzer style: every 10 or at end)
    if((gi+1) % 10 == 0 || gi == entries.size()-1) {
      double elapsed = 0; // could track time
      cout << "  [" << (gi+1) << "/" << entries.size() << "] "
           << "ok=" << success << " skip=" << skipped << " fail=" << failed
           << " (last: " << meta.blackName << " vs " << meta.whiteName << ")" << endl;
    }
  }

  cout << "\n=== batch-analysis complete ===" << endl;
  cout << "  Games: " << entries.size() << " total, "
       << success << " ok, " << skipped << " skipped (<10 moves), "
       << failed << " failed" << endl;
  cout << "  Output: " << outputDir << "/" << endl;
  return 0;
}

}  // namespace MainCmds
