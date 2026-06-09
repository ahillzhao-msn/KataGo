#ifndef STRMODEL_PRECOMPUTE_H
#define STRMODEL_PRECOMPUTE_H

#include <memory>
#include <string>
#include <vector>
#include <queue>
#include "dataio/sgf.h"
#include "neuralnet/nneval.h"
#include "strmodel/dataset.h"

namespace StrModel {

// specifies which features should be extracted for the strength model
struct Selection {
  bool trunk; // raw trunk outputs from the neural net
  bool pick;  // move-position feature vector picked from trunk output
  bool head; // traditional head features like winrate, lead and policy
};

// Specifies that we need the given features from NN output for the board with the given turn index.
struct BoardQuery {
  int turn; // 0-based turn number in the game
  Selection selection; // which features to query for this board
};

// Holds output from the NN for the board with the given turn index.
struct BoardResult {
  int turn; // 0-based turn number in the game
  Player pla;
  int pos; // index into trunk data of move chosen by player
  float whiteWinProb;
  float whiteLossProb; // not necessarily 1-winProb because no result is possible
  float expectedScore; // predicted score at end of game by NN
  float whiteLead; // predicted bonus points to make game fair
  float movePolicy; // policy at move location
  float maxPolicy; // best move policy
  shared_ptr<FeatureVector> trunk; // trunk features
  shared_ptr<FeatureVector> pick; // trunk features at move location
};

class Precompute {

public:

  explicit Precompute(NNEvaluator& evaluator);

  // construct the queries to get the specified selection of features for the specified turns in a game
  static vector<BoardQuery> makeQuery(const vector<int>& turns, Selection selection);
  // query selected features on the given game
  vector<BoardResult> evaluate(const CompactSgf& sgf, const vector<BoardQuery>& query);
  // query selected features on a single board
  BoardResult evaluateBoard(Board& board, const BoardHistory& history, Loc loc, Player pla, Selection sel);
  // combine evaluation results (like winrate every turn) into features (like winrate loss)
  static vector<BoardFeatures> combine(const vector<BoardResult>& results);
  // get features for specific turns (recent moves)
  static vector<BoardFeatures> filter(const vector<BoardFeatures>& features, const vector<int>& turns);

private:

  NNEvaluator* evaluator;

};

} // end namespace StrModel

#endif
