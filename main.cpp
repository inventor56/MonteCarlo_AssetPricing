#include "MonteCarlo.h"
#include <array>
#include <iostream>
#include <fstream>

using namespace std;

int main() {
  int num_of_simulations;
  int days_to_generate = 10;
  // Parse input from file

  int hist_length = 8;
  float* results;
  float* hist_arr = new float[hist_length];
  hist_arr[0] = 78.237083;
  hist_arr[1] = 78.36644;
  hist_arr[2] = 83.391197;
  hist_arr[3] = 83.470802;
  hist_arr[4] = 82.764351;
  hist_arr[5] = 82.764351;
  hist_arr[6] = 83.630005;
  hist_arr[7] = 83.719551;

  MonteCarlo testing = MonteCarlo(hist_arr, hist_length, days_to_generate, time(nullptr));
  results = testing.getResults();

  for (int i = 0; i < days_to_generate; i++) {
    cout << results[i] << " : ";
  }





  // Run Serial Version
  // Run Parallel Version

  // Write results to new file (different columns?)
  /*
  string fileName;
  // open file, generate data
  cin >> filename;
  ofstream statsFile(filename);
  if(statsFile.is_open()){
    statsFile << "Day\t";
    for (int j = 0; j < num_of_simulations; j++) {
      statsFile << "Simulation" << j << ":Price"
      for(int i = 1; i <= days_to_generate; i++) {
          rng_grid grid_i = rng_grid(threads, i, i); // threads and grid of size i X i
          grid_i.test1(0);
          grid_i.test3(0);
          statsFile << i << "\t" << grid_i.computeSpeedup() << "\n";
      }
    }
    statsFile.close();
  }
  else cout << "Unable to Open File" << endl; */

  return 0;
}
