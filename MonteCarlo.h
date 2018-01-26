
#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

// Think about what we want this class to do....
class MonteCarlo {
public:
  MonteCarlo();
  MonteCarlo(float* historicalData, int daysToGenerate);
  float* getResults();
private:
  // User Inputted Variables
  float* historical_price_data; // The historical price data
  int total_days;
  // Runtime Variables
  float* results; // The results of your calculations
  float* random_numbers;
  float* periodic_daily_return; // periodic daily return = ln (day's price ÷ previous day's price)
  float drift;
  float std_dev; // Standard deviation
  //Functions
  float calculateDrift(float avgDailyReturn, float variance); //drift = average daily return - (variance ÷ 2)
  float calculateAverage();
  float calculateVariance();
  float calculateStandardDeviation();
  float calculateNextDayPrice();


};

#endif
