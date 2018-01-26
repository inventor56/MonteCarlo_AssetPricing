
#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

// Think about what we want this class to do....
class MonteCarlo {
public:
  MonteCarlo();
  MonteCarlo(float* historicalData, int lengthOfHD, int daysToGenerate);
  float* getResults();
private:
  // User Inputted Variables
  float* historical_price_data; // The historical price data
  int sizeOfHPD; // Array length of historical price data
  int total_days;
  // Runtime Variables
  float* results; // The results of your calculations
  float* periodic_daily_return;
  int sizeOfPDR; // Array length of PDR
  float drift;
  float std_dev; // Standard deviation
  float average_daily_return; // Average across whoel series
  float variance;
  //Functions
  void calculatePeriodicDailyReturn(); // periodic daily return = ln (day's price ÷ previous day's price)
  void calculateDrift(); //drift = average daily return - (variance ÷ 2)
  void calculateAverage();
  void calculateVariance();
  void calculateStandardDeviation();
  void calculateFuturePricing();
  f


};

#endif
