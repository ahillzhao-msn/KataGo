#include "precompute.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <zip.h>
#include "dataio/sgf.h"
#include "dataio/numpywrite.h"
#include "neuralnet/modelversion.h"

#include <iostream>

namespace StrModel {

Precompute::Precompute(NNEvaluator& evaluator_)
: evaluator(&evaluator_)
{
}

vector<BoardQuery> Precompute::makeQuery(const vector<int>& turns, Selection selection) {
  vector<BoardQuery> queries;
  for(size_t i = 0; i < turns.size(); i++) {
    queries.push_back({turns[i], selection});
    if((i+1 >= turns.size() || turns[i+1] != turns[i]+1) && selection.head)
      // for head features, we also need the board after the move to calculate winrate loss etc
      queries.push_back({turns[i]+1, {false, false, selection.head}});
  }
  return queries;
}

vector<BoardResult> Precompute::evaluate(const CompactSgf& sgf, const vector<BoardQuery>& query) {
  const auto& moves = sgf.moves;
  Rules rules = sgf.getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
  Board board;
  BoardHistory history;
  Player initialPla;
  sgf.setupInitialBoardAndHist(rules, board, initialPla, history);
  auto queryIt = query.begin();
  auto queryEnd = query.end();
  vector<BoardResult> results;

  for(int turnIdx = 0; turnIdx < moves.size() && queryIt != queryEnd; turnIdx++) {
    Move move = moves[turnIdx];

    if(turnIdx == queryIt->turn) {
      BoardResult result = evaluateBoard(board, history, move.loc, move.pla, queryIt->selection);
      result.turn = queryIt->turn; // must match for later filter
      results.push_back(result);
      queryIt++;
    }

    history.makeBoardMoveAssumeLegal(board, move.loc, move.pla, NULL, true);
  }
  if(queryIt != queryEnd) { // add final board?
    if(moves.size() != queryIt->turn)
      throw ValueError(strprintf("Query for turn %d in %d-move game", queryIt->turn, moves.size()));
    BoardResult result = evaluateBoard(board, history, Board::NULL_LOC, C_EMPTY, queryIt->selection);
    result.turn = queryIt->turn; // must match for later filter
    results.push_back(result);
    if(++queryIt != queryEnd)
      throw ValueError(strprintf("Query for turn %d in %d-move game", queryIt->turn, moves.size()));
  }
  return results;
}

BoardResult Precompute::evaluateBoard(Board& board, const BoardHistory& history, Loc loc, Player pla, Selection sel) {
  int nnXLen = evaluator->getNNXLen();
  int nnYLen = evaluator->getNNYLen();
  int rowPos = NNPos::locToPos(loc, board.x_size, nnXLen, nnYLen);

  MiscNNInputParams nnInputParams;
  nnInputParams.symmetry = 0;
  nnInputParams.policyOptimism = 0;

  NNResultBuf buf;
  buf.rowPos = rowPos;
  buf.includeTrunk = sel.trunk;
  buf.includePick = sel.pick;
  Player evalPla = C_EMPTY == pla ? history.presumedNextMovePla : pla;
  evaluator->evaluate(board, history, evalPla, nnInputParams, buf, false, false);
  assert(buf.hasResult);
  NNOutput& nnout = *buf.result;

  // interpret NN result
  BoardResult result;
  result.pla = pla;
  result.pos = rowPos;
  int trunkNumChannels = evaluator->getTrunkNumChannels();
  assert(trunkNumChannels > 0); // this is always the case if evaluator is properly set up
  if(nnout.trunk)
    result.trunk = make_shared<FeatureVector>(nnout.trunk, nnout.trunk + trunkNumChannels * nnXLen * nnYLen); // trunk features at move location
  if(nnout.pick)
    result.pick = make_shared<FeatureVector>(nnout.pick, nnout.pick + trunkNumChannels); // trunk features at move location
  if(sel.head) {
    result.whiteWinProb = nnout.whiteWinProb;
    result.whiteLossProb = nnout.whiteLossProb;
    result.expectedScore = nnout.whiteScoreMean;
    result.whiteLead = nnout.whiteLead;
    result.movePolicy = nnout.policyProbs[buf.rowPos];
    result.maxPolicy = *std::max_element(std::begin(nnout.policyProbs), std::end(nnout.policyProbs));
  }
  return result;
}

vector<BoardFeatures> Precompute::combine(const vector<BoardResult>& results) {
  vector<BoardFeatures> features;
  size_t count = results.size();

  for(size_t i = 0; i < count; i++) {
    BoardFeatures feats;
    const BoardResult& result = results[i];
    feats.turn = result.turn;
    feats.pla = result.pla;
    feats.pos = result.pos;
    feats.trunk = result.trunk;
    feats.pick = result.pick;

    if(i+1 < count) { // head features only work for boards showing the move outcome
      const BoardResult& nextresult = results[i+1];
      if(nextresult.turn == result.turn+1) { // we can only determine head features when we have the next move
        FeatureVector head(6);
        head[0] = P_WHITE == result.pla ? nextresult.whiteWinProb : nextresult.whiteLossProb; // post-move winProb
        head[1] = P_WHITE == result.pla ? nextresult.whiteLead : -nextresult.whiteLead; // lead
        head[2] = result.movePolicy; // movePolicy
        head[3] = result.maxPolicy; // maxPolicy
        head[4] = P_WHITE == result.pla // winrateLoss
                         ? result.whiteWinProb - nextresult.whiteWinProb
                         : result.whiteLossProb - nextresult.whiteLossProb;
        head[5] = P_WHITE == result.pla // pointsLoss
                         ? result.whiteLead - nextresult.whiteLead
                         : -(result.whiteLead - nextresult.whiteLead);
        feats.head = make_shared<FeatureVector>(move(head));
      }
    }
    features.push_back(move(feats));
  }

  return features;
}

vector<BoardFeatures> Precompute::filter(const vector<BoardFeatures>& features, const vector<int>& turns) {
  if(features.empty() && !turns.empty())
    throw ValueError(strprintf("Cannot get results for %d turns: precomputed boards are empty", turns.size()));

  vector<BoardFeatures> filtered;
  size_t fIdx = 0;
  size_t tIdx = 0;
  while(fIdx < features.size() && tIdx < turns.size()) {
    // we rely on the invariant that features and turns are always ordered ascending
    int fTurn = features[fIdx].turn;
    int tTurn = turns[tIdx];
    if(fTurn == tTurn) {
      filtered.push_back(features[fIdx]);
      fIdx++;
      tIdx++;
    }
    else if(fTurn < tTurn)
      fIdx++;
    else if(fTurn > tTurn)
      tIdx++;
  }

  if(tIdx < turns.size())
    throw ValueError(strprintf("Move %d not found in %d (%d-%d) precomputed boards",
      turns[tIdx], features.size(), features.front().turn, features.back().turn));

  return filtered;
}

} // end namespace StrModel
