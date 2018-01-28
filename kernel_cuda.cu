#include "kernel_cuda.h"
#include "book.h"
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
double** cuda_run(double* histArr, int histLength, int daysToGenerate, int simulationsToRun) {
  double drift_amt;
  // Array in which we will store our results in
  double ** cuda_results = new double*[simulationsToRun];
  for (int r = 0; r < simulationsToRun; r++) {
    cuda_results[r] = new double[daysToGenerate];
  }

  MonteCarlo* createDrift = new MonteCarlo(histArr, histLength, daysToGenerate, simulationsToRun);
  createDrift->calculateResults(false); // cuda version, just calculateto get the drift
  drift_amt = createDrift->getDrift();

  // Now we can run our parallel program

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
