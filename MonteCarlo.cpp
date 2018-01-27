
#include "MonteCarlo.h"
#include <cmath>
#include <random>
#include <iostream>


  MonteCarlo::MonteCarlo() { //default constructor
    total_days = 10;
    results = new double[total_days];
    //historicalData get elsewhere
  }
  MonteCarlo::MonteCarlo(double* historicalData, int lengthOfHD, int daysToGenerate, unsigned int seed_val) {
    historical_price_data = historicalData;
    total_days = daysToGenerate;
    sizeOfHPD = lengthOfHD;
    results = new double[total_days];
    seed = seed_val;
    std::cout << seed_val << std::endl;
  }

  void MonteCarlo::calculatePeriodicDailyReturn() {
    sizeOfPDR = sizeOfHPD-1;
    periodic_daily_return = new double[sizeOfPDR];
    for(int i = 1; i < sizeOfHPD; i++) {
      periodic_daily_return[i] = log(historical_price_data[i] / historical_price_data[i-1]);
    }
  }

  double* MonteCarlo::getResults() {
    calculatePeriodicDailyReturn();
    calculateAverage();
    calculateVariance();
    calculateStandardDeviation();
    calculateDrift();
    calculateFuturePricing();

    return results;
  }

  void MonteCarlo::calculateDrift(){
    //drift = average daily return - (variance ÷ 2)
    drift = average_daily_return - (variance/2.0);
    std::cout << "ERAERAER drit " << drift << std::endl;
  }
  void MonteCarlo::calculateAverage() {
    double result = 0;
    for (int i = 0; i < sizeOfPDR; i++) {
      result += periodic_daily_return[i];
    }
    average_daily_return = result/sizeOfPDR;
  }
  void MonteCarlo::calculateVariance() {
    double squared_differences[sizeOfPDR];
    double mean = 0;

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

  void MonteCarlo::calculateFuturePricing() {
    std::default_random_engine generator(seed); // create generator with seed
    std::normal_distribution<double> dist (0.0,1.0);

    double currentPrice = historical_price_data[sizeOfHPD-1]; // Get last value in price data to start with.
    for (int i = 0; i < total_days; i++) {
      results[i] = currentPrice*(exp(drift+std_dev*dist(generator)));
      currentPrice = results[i];
    }
  }
