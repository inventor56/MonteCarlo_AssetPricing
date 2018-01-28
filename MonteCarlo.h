
#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#include <random>

// Think about what we want this class to do....
class MonteCarlo {
public:
  MonteCarlo();
  MonteCarlo(double* historicalData, int lengthOfHD, int daysToGenerate, unsigned int seed_val);
  void calculateResults(bool non_cuda);
  double getResultAt(int index);
  double getDrift();
private:
  // User Inputted Variables
  double* historical_price_data; // The historical price data
  int sizeOfHPD; // Array length of historical price data
  int total_days;
  // Runtime Variables
  double* results; // The results of your calculations
  double* periodic_daily_return;
  int sizeOfPDR; // Array length of PDR
  double drift;
  double std_dev; // Standard deviation
  double average_daily_return; // Average across whoel series
  double variance;
  unsigned int seed;
  //Functions
  void calculatePeriodicDailyReturn(); // periodic daily return = ln (day's price ÷ previous day's price)
  void calculateDrift(); //drift = average daily return - (variance ÷ 2)
  void calculateAverage();
  void calculateVariance();
  void calculateStandardDeviation();
  void calculateFuturePricing();
};

#endif
