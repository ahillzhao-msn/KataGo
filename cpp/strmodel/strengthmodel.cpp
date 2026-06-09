#include "strengthmodel.h"
#include "core/global.h"
#include "core/fileutils.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include "core/using.h"

namespace StrModel {

using std::sqrt;

float Predictor::glickoScore(float blackRating, float whiteRating) {
  float GLICKO2_SCALE = 173.7178f;
  return 1.0f / (1.0f + exp((whiteRating - blackRating) / GLICKO2_SCALE));
  // Elo Score (for reference):
  // float Qblack = static_cast<float>(std::pow(10, blackRating / 400));
  // float Qwhite = static_cast<float>(std::pow(10, whiteRating / 400));
  // return Qblack / (Qblack + Qwhite);
}

namespace {

float fSum(float a[], size_t N) noexcept {  // sum with slightly better numerical stability
  if(N <= 0)
    return 0;
  for(size_t step = 1; step < N; step *= 2) {
    for(size_t i = 0; i+step < N; i += 2*step)
      a[i] += a[i+step];
  }
  return a[0];
}

float fAvg(float a[], size_t N) noexcept {
  return fSum(a, N) / N;
}

float fVar(float a[], size_t N, float avg) noexcept { // corrected variance
  for(size_t i = 0; i < N; i++)
    a[i] = (a[i]-avg)*(a[i]-avg);
  return fSum(a, N) / (N-1);
}

// Because of float limitations, normcdf(x) maxes out for |x| > 5.347.
// Therefore its value is capped such that the result P as well as
// 1.f-P are in the closed interval (0, 1) under float arithmetic.
float normcdf(float x) noexcept {
  float P = .5f * (1.f + std::erf(x / sqrt(2.f)));
  if(P >= 1) return std::nextafter(1.f, 0.f);     // =0.99999994f, log(0.99999994f): -5.96e-08
  if(P <= 0) return 1 - std::nextafter(1.f, 0.f); // =0.00000006f, log(0.00000006f): -16.63
  else return P;
}

}

Dataset::Prediction StochasticPredictor::predict(const MoveFeatures* blackFeatures, size_t blackCount, const MoveFeatures* whiteFeatures, size_t whiteCount) {
  if(0 == blackCount || 0 == whiteCount)
    return {0, 0, .5}; // no data for prediction
  constexpr float gamelength = 100; // assume 100 moves per player for an average game
  vector<float> buffer(std::max(blackCount, whiteCount));
  for(size_t i = 0; i < whiteCount; i++)
    buffer[i] = whiteFeatures[i].pointsLoss;
  float wplavg = fAvg(vector<float>(buffer).data(), whiteCount);  // average white points loss
  float wplvar = 2 <= whiteCount ? fVar(buffer.data(), whiteCount, wplavg) : 100.f;      // variance of white points loss
  for(size_t i = 0; i < blackCount; i++)
    buffer[i] = blackFeatures[i].pointsLoss;
  float bplavg = fAvg(vector<float>(buffer).data(), blackCount);  // average black points loss
  float bplvar = 2 <= blackCount ? fVar(buffer.data(), blackCount, bplavg) : 100.f;      // variance of black points loss
  const float epsilon = 0.000001f;  // avoid div by 0
  float z = sqrt(gamelength) * (wplavg - bplavg) / sqrt(bplvar + wplvar + epsilon); // white pt advantage in standard normal distribution at move# [2*gamelength]
  return {0, 0, normcdf(z)};
}

} // end namespace StrModel
