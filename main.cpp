#include "MonteCarlo.h"
#include <array>
#include <iostream>
#include <fstream>

using namespace std;

int main() {
  int num_of_simulations = 5;
  int hist_length = 0;
  int days_to_generate = 10;
  double * hist_array;
  double ** results;
  // Parse input from file

  vector<double> temp_storage;
  string val, trash;
  ifstream csvFile("DataSets/DIS.csv");
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


  results = new double*[num_of_simulations];
  for (int r = 0; r < num_of_simulations; r++) {
    results[r] = new double[days_to_generate];
  }
  // Here, we need to set up way to store data for multiple simulations


  for (int k = 0; k < num_of_simulations; k++) {
    // Creatre monte carlo with time + k as a seed_val
    MonteCarlo* obj = new MonteCarlo(hist_array, hist_length, days_to_generate, time(nullptr)+k);
    // Calculate calculateResults
    obj->calculateResults();
    // add results to array

    for (int aa = 0; aa < days_to_generate; aa++){
      results[k][aa] = obj->getResultAt(aa);
    } 
  }

  for (int i = 0; i < num_of_simulations; i++) {
    cout << "Simulation " << i << ":";
    for (int c = 0; c < days_to_generate; c++) {
      cout << "day "<< c << ": "<< results[i][c] << " . ";
    }
    cout << endl;
  }
  /*
MonteCarlo obj = MonteCarlo(hist_array, hist_length, days_to_generate, time(nullptr));
obj.calculateResults();
for (int c = 0; c < days_to_generate; c++) {
  cout << "day "<< c << ": "<< obj.getResultAt(c) << " . ";
}
*/

  /*
  string writeOutFile;
  int iterations;

  // Recieve user input
  cout << "Please enter the max x number of x by x grids you'd like to generate"
       << "(i.e x = 100, we'll generate 1x1, 2x2, 3x3, ..., 99x99, 100x100)" << endl;
  cin >> iterations;
  input_verifier(); // Verify input
  cout << "Please enter in the name of the text file name you would like to save to" << endl;
  input_verifier(); // Verify input

  // open file, generate data
  cin >> filename;
  ofstream statsFile(filename);
  if(statsFile.is_open()){
      statsFile << "Data Size\tSpeedup\n";
      for(int i = 1; i <= iterations; i++) {
          rng_grid grid_i = rng_grid(threads, i, i); // threads and grid of size i X i
          grid_i.test1(0);
          grid_i.test3(0);
          statsFile << i << "\t" << grid_i.computeSpeedup() << "\n";
      }
      statsFile.close();
  }
  else cout << "Unable to Open File" << endl;
  */

  return 0;
}
