
#include "MonteCarlo.h"
#include <cmath>
#include <random>


  MonteCarlo::MonteCarlo() { //default constructor
    total_days = 10;
    results = new float[total_days];
    //historicalData get elsewhere
  }
  MonteCarlo::MonteCarlo(float* historicalData, int lengthOfHD, int daysToGenerate, float seed_val) {
    historical_price_data = historicalData;
    total_days = daysToGenerate;
    sizeOfHPD = lengthOfHD;
    results = new float[total_days];
    seed = seed_val;
  }
  float* MonteCarlo::getResults() {
    return results;
  }

  void MonteCarlo::calculatePeriodicDailyReturn() {
    sizeOfPDR = sizeOfHPD-1;
    periodic_daily_return = new float[sizeOfPDR];
    for(int i = 1; i < sizeOfHPD; i++) {
      periodic_daily_return[i] = log(historical_price_data[i] / historical_price_data[i-1]);
    }
  }

  void MonteCarlo::calculateDrift(){
    //drift = average daily return - (variance ÷ 2)
    drift = average_daily_return - (variance/2.0);
  }
  void MonteCarlo::calculateAverage() {
    float result = 0;
    for (int i = 0; i < sizeOfPDR; i++) {
      result += periodic_daily_return[i];
    }
    average_daily_return = result/sizeOfPDR;
  }
  void MonteCarlo::calculateVariance() {
    float squared_differences[sizeOfPDR];
    float mean = 0;

    for (int i = 0; i < sizeOfPDR; i++) {
      squared_differences[i] = pow((periodic_daily_return[i] - average_daily_return), 2.0); // square
    }
    // Now get the mean of these differences
    for (int j = 0; j < sizeOfPDR; j++) {
      mean += squared_differences[j];
    }

    variance = mean / sizeOfPDR;
  }
  void MonteCarlo::calculateStandardDeviation() {
    std_dev = sqrt(variance);
  }
  float MonteCarlo::getRandomNumber() {
    std::default_random_engine generator (seed); // create generator with seed
    std::normal_distribution<float> dist (0.0,1.0);
    return std_dev*dist(generator); // The generated random number
  }

  void MonteCarlo::calculateFuturePricing() {
    float randomVal = getRandomNumber();
    float currentPrice = historical_price_data[sizeOfHPD-1]; // Get last value in price data to start with.
    for (int i = 0; i < total_days; i++) {
      results[i] = currentPrice*(exp(drift+randomVal));
      currentPrice = results[i];
    }
  }
