#include "MonteCarlo.h"
#include <array>
#include <iostream>
#include <fstream>
#include "kernel_cuda.h"
#include <chrono>

using namespace std;


void input_verifier() {
    if (cin.fail()) {
        cout << "Invalid Input." << endl;
        exit(1);
    }
}

int main() {
  int num_of_simulations = 5;
  int hist_length = 0;
  int days_to_generate = 10;
  double * hist_array;
  double ** results;


  // Parse input from file
  string fileInput;
  vector<double> temp_storage;
  string val, trash;
  int run_type = 2;


  // Welcome output, recieve filename
  cout << "Welcome to the Asset Price Fluctuation Monte Carlo simulation\n"
    << "Please enter the path name for your dataset file. (data sets can be downloaded from finance.yahoo.com/quote)\n "
    << "Some sample data sets are included in the DataSets folder.\n"
    << "Type 'DataSets/DIS.csv' for instance, in order to use the Disney stock data set:"
    << endl;
  cin >> fileInput;
  input_verifier();
  // Get the option the user would like to perform for generations:
  // 1. Serialized method
  // 2. Parallel method on CUDA cores
  cout << "Please enter the type of execution:\n"
    << "1. Serialized method\n"
    << "2. Parallel method on CUDA cores\n"
    << endl;
  cin >> run_type;
  input_verifier();
  // Get the number of days the user would like to generate
  cout << "Please enter the degree of accuracy (i.e. number of simulations and days to generate) for generated predictions: " << endl;
  cin >> days_to_generate;
  input_verifier();
  num_of_simulations = days_to_generate;

  //Quick Checks
  if (num_of_simulations > 10000) {  // If simulation size greater than what CUDA can handle
    cout << "We're sorry, but that number of simulations is too large for CUDA to handle!" << endl;
    exit(1);
  }

  ifstream csvFile(fileInput);
  getline(csvFile, trash); // Skip first line
  while (true) {
  	for (int b = 0; b < 5; b++) {// Set up specifically for Yahoo Finance data (as of the current version)
  		getline(csvFile, trash, ',');
  	}
  	if (csvFile.eof())
  		break;
  	getline(csvFile, val, ',');
  	temp_storage.push_back(stof(val));
  	hist_length++; // Increment total amount of historical data
  }
  csvFile.close();

  // Convert vector to array (for now)
  hist_array = new double[hist_length];
  copy(temp_storage.begin(), temp_storage.end(), hist_array);


  ///////////////////////////////
  // Serialized version!
  ///////////////////////////////

  // Here, we need to set up way to store data for multiple simulations
  if (run_type == 1) {
    auto begin_time = std::chrono::high_resolution_clock::now(); // For keeping track of when the chrono clock begins

    results = new double*[num_of_simulations];
    for (int r = 0; r < num_of_simulations; r++) {
      results[r] = new double[days_to_generate];
    }

    for (int k = 0; k < num_of_simulations; k++) {
      // Creatre monte carlo with time + k as a seed_val
      MonteCarlo* obj = new MonteCarlo(hist_array, hist_length, days_to_generate, time(nullptr)+k);
      // Calculate calculateResults
      obj->calculateResults(true); // non-cuda
      // add results to array

      for (int aa = 0; aa < days_to_generate; aa++){
        results[k][aa] = obj->getResultAt(aa);
      }
    }
    auto end_time = std::chrono::high_resolution_clock::now(); // For keeping track of when the chrono clock ends
    std::chrono::duration<double, std::milli> duration = end_time - begin_time;
    double duration_serial = duration.count();
    cout << "Serially computing " << num_of_simulations << " simulations, with results stretching out "
      << days_to_generate << " days took " << duration_serial << " ms." << endl;
  }
  ////////////////////////////////////////////
  // Parallelized Version (CUDA Technology)!
  ////////////////////////////////////////////

  else if (run_type == 2){
    auto begin_time = std::chrono::high_resolution_clock::now(); // For keeping track of when the chrono clock begins

    results = cuda_run(hist_array, hist_length, days_to_generate, num_of_simulations);

    auto end_time = std::chrono::high_resolution_clock::now(); // For keeping track of when the chrono clock ends
    std::chrono::duration<double, std::milli> duration = end_time - begin_time;
    double duration_cuda = duration.count();
    cout << "Parallelly computing " << num_of_simulations << " simulations, with results stretching out "
      << days_to_generate << " days took " << duration_cuda << " ms." << endl;

  }

  // Offer to write data out, if results aren't too massive for Excel entry
  if (num_of_simulations <= 100){
    // Write out results to a filename
    // Write out with each row representing the simulation #
    // Write out with each column representing the day #
    string writeOutFile;
    cout << "Please enter in the name of the text file name you would like to save to" << endl;
    input_verifier(); // Verify input

    // Open a file to write Excel readable data into
    cin >> writeOutFile;
    ofstream statsFile(writeOutFile);
    if(statsFile.is_open()){
        statsFile << "\t";
        for (int x = 1; x <= days_to_generate; x++) {
          statsFile << "Day #" << x << "\t";
        }
        statsFile << "\n";
        for(int h = 0; h < num_of_simulations; h++) {
          statsFile << "Sim #" << h+1 << "\t";
          for (int l = 0; l < days_to_generate; l++) {
            statsFile << results[h][l] << "\t";
          }
          statsFile << "\n";
        }
        statsFile.close();
    }
    else cout << "Unable to Open File" << endl;
  }

  return 0;
}
