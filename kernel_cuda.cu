#include "kernel_cuda.h"
#include "book.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "MonteCarlo.h"

/*
__global__ void kernel(curandState_t* frames) { // Invoking the CUDA Kernel, delegating each thread to save it's ID
  curand(&frames[blockIdx.x]) % 1000;
}

__global__ void init(unsigned int seed, curandState_t* states) {
  curand_init(seed,
              blockIdx.x,
              0,
              &states[blockIdx.x]);
}
*/

__global__ void kernel(double drift, double init_price, int days, float* normals, double* result) {
  int index = blockIdx.x*blockDim.x;
  double currentPrice = init_price; // Get last value in price data to start with.
  for (int i = 0; i < days; i++) {
    result[index+i] = currentPrice*(exp(drift)+normals[blockIdx.x*blockDim.x+i]);
    currentPrice = result[index+i];
  }
}

double** cuda_run(double* histArr, int histLength, int daysToGenerate, int simulationsToRun) {
  int n_size = daysToGenerate*simulationsToRun;
  double drift_amt;


  // Allocate memory for...
  // Random Numbers
  float* rand_normals; //= new float[n_size];
  // Array (one-dimensional) in which we will store our reuslts in intially, before converting
  double* initial_cuda_results;
  HANDLE_ERROR( cudaMallocManaged(&rand_normals, n_size * sizeof(float)));
  HANDLE_ERROR( cudaMallocManaged(&initial_cuda_results, n_size * sizeof(double)));

  // Array in which we will store our final results in
  double ** cuda_results = new double*[simulationsToRun];
  for (int r = 0; r < simulationsToRun; r++) {
    cuda_results[r] = new double[daysToGenerate];
  }

  // Calculate drift
  MonteCarlo* createDrift = new MonteCarlo(histArr, histLength, daysToGenerate, simulationsToRun);
  createDrift->calculateResults(false); // cuda version, just calculateto get the drift
  drift_amt = createDrift->getDrift();

  // Create all random numbers for the normal distribution, using currand parallelism
  curandGenerator_t curGen;
  curandCreateGenerator(&curGen, CURAND_RNG_PSEUDO_MTGP32); // use the mersenne twister algorithm for CURAND_RNG_PSEUDO_MTGP32
  curandSetPseudoRandomGeneratorSeed(curGen, 1234ULL); // generate using a big unsigned int as your seed
  curandGenerateNormal(curGen, rand_normals, n_size, 0.0f, 1.0f); // generate

  // Determine how many blocks & internal threads will need to be utilized


  // Now we can run our parallel program
  kernel<<<simulationsToRun,1>>>(drift_amt, histArr[histLength-1], daysToGenerate, rand_normals, initial_cuda_results);

  cudaDeviceSynchronize();

  // Free up memory on the GPU
  HANDLE_ERROR( cudaFree(rand_normals));
  HANDLE_ERROR (cudaFree(initial_cuda_results));

  //Return results
  return cuda_results;
/*
// Figure this out add in array and grab indices fromt here
  int dimensions = columns * rows;
  int * gen_nums
  int

  //Version with manual memory in array
  curandState_t* frames;
  int blocks = 200;


  HANDLE_ERROR( cudaMalloc((void**) &frames, dimensions * sizeof(curandState_t)));

  init<<<blocks, 1>>>(time(nullptr), frames);

  kernel<<<blocks, 1>>>(columns, frames);

  HANDLE_ERROR( cudaFree( frames)); // Free memory

  return true;
  */
}
