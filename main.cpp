#include "MonteCarlo.h"
#include <array>
#include <iostream>
#include <fstream>
#include "kernel_cuda.h"

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


  // Welcome and days to generate
  cout << "Welcome to the Asset Price Fluctuation Monte Carlo simulation"
    << "Please enter the path name for your dataset file. (data sets can be downloaded from finance.yahoo.com/quote): "
    << endl;
  cin >> fileInput;
  input_verifier();
  // Get the number of days the user would like to generate
  cout << "Please enter the number of days you will need future predictions for: " << endl;
  cin >> days_to_generate;
  input_verifier();
  // Get the number of simulations the user would like to perform
  cout << "Please enter the number of simulations you would like to run: " << endl;
  cin >> num_of_simulations;
  input_verifier();
  // Get the option the user would like to perform for generations:
  // 1. Serialized method
  // 2. Parallel method on CUDA cores
  // 3. Perform both and then compare results

  ifstream csvFile(fileInput);
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
  cout << "We have " << hist_length << " numbers" << endl;
  csvFile.close();

  // Convert vector to array (for now)
  hist_array = new double[hist_length];
  copy(temp_storage.begin(), temp_storage.end(), hist_array);


  ///////////////////////////////
  // Serialized version!
  ///////////////////////////////

  // Here, we need to set up way to store data for multiple simulations
  if (run_type == 1) {
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


    /*
    for (int i = 0; i < num_of_simulations; i++) {
      cout << "Simulation " << i << ":";
      for (int c = 0; c < days_to_generate; c++) {
        cout << "day "<< c << ": "<< results[i][c] << " . ";
      }
      cout << endl;
    } */
  }
  ////////////////////////////////////////////
  // Parallelized Version (CUDA Technology)
  ////////////////////////////////////////////

  else if (run_type == 2){
    results = cuda_run(hist_array, hist_length, days_to_generate, num_of_simulations);
  }



  // Write out results to a filename
  // Write out with each row representing the simulation #
  // Write out with each column representing the day #
  string writeOutFile;

  cout << "Please enter in the name of the text file name you would like to save to" << endl;
  input_verifier(); // Verify input

  // open file, generate data
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

  return 0;
}
