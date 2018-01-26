
#include "MonteCarlo.h"


  MonteCarlo::MonteCarlo() { //default constructor
    total_days = 10;
    //historicalData get elsewhere
  }
  MonteCarlo::MonteCarlo(float* historicalData, int daysToGenerate) {
    historical_price_data = historicalData;
    total_days = daysToGenerate;
  }
  float* MonteCarlo::getResults() {
    return results;
  }


  float MonteCarlo::calculateDrift(float avgDailyReturn, float variance){
    //drift = average daily return - (variance ÷ 2)
    return 0;
  }
  float MonteCarlo::calculateAverage() {
    return 0;
  }
  float MonteCarlo::calculateVariance() {
    return 0;
  }
  float MonteCarlo::calculateStandardDeviation() {
    return 0;
  }
  float MonteCarlo::calculateNextDayPrice() {
    return 0;
  }
