#include "MonteCarlo.h"
#include <array>
#include <iostream>
#include <fstream>

using namespace std;

int main() {
  int num_of_simulations;
  int hist_length = 0;
  int days_to_generate = 10;
  double * hist_array;
  double * results;
  // Parse input from file
  
  vector<double> temp_storage;
  string val, trash;
  ifstream csvFile("DataSets/MSthreeMonth.csv");
  getline(csvFile, trash); // Skip first line
  while (true) {
  	for (int b = 0; b < 5; b++) {// Set up specifically for Yahoo Finance data (as of the current version)
  		getline(csvFile, trash, ',');
  		//cout << "These should be erased: " << val << endl;
  	}
  	if (csvFile.eof())
  		break;
  	getline(csvFile, val, ',');
  	cout << "This is the val: " << val << endl;
  	temp_storage.push_back(stof(val));
  	hist_length++; // Increment total amount of historical data
  }
  cout << "We have " << hist_length << "numbers" << endl;
  csvFile.close();
  
  // Convert vector to array (for now)
  hist_array = new double[hist_length];
  copy(temp_storage.begin(), temp_storage.end(), hist_array);
  
  MonteCarlo testing = MonteCarlo(hist_array, hist_length, days_to_generate, time(nullptr));
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
