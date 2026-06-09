#ifndef PROGRAM_STRENGTHMODEL_H_
#define PROGRAM_STRENGTHMODEL_H_

#include "core/rand.h"
#include "search/search.h"
#include "dataio/sgf.h"
#include "strmodel/dataset.h"

namespace StrModel {

// The predictor, given a match between two opponents, estimates their ratings and the match score (win probability).
// This is the abstract base class for our predictors:
//   - The StochasticPredictor based on simple statistics
//   - The FullPredictor (to be done!)
class Predictor {

public:

  // The resulting prediction might keep the players' ratings at 0 (no prediction), but it always predicts the score.
  virtual Dataset::Prediction predict(const MoveFeatures* blackFeatures, size_t blackCount, const MoveFeatures* whiteFeatures, size_t whiteCount) = 0;

protected:

  // give an expected score by assuming that the given ratings are Glicko-2 ratings.
  static float glickoScore(float blackRating, float whiteRating);

};

class StochasticPredictor : public Predictor {

public:

  Dataset::Prediction predict(const MoveFeatures* blackFeatures, size_t blackCount, const MoveFeatures* whiteFeatures, size_t whiteCount) override;

};

} // end namespace StrModel

#endif  // PROGRAM_STRENGTHMODEL_H_
