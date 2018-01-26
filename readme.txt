- CPSC 445
- Assignment: CUDA Mini Project
- Code Author: Josh Smith

- To run executable in Linux terminal, navigate to the Executable directory and type ./MersenneTwister
- To compile from the source code, be sure to compile with C++11 and Cuda
	- nvcc -std=c++11 main.cpp mst.cpp rng_grid.cpp kernel_cuda.cu -o MersenneTwister.c



- PROGRAM INSTRUCTIONS
	- Herbie Goes to Monte Carlo

- NOTES
	- You may experience some higher performance than normal on initial runs
		- i.e. efficiency may exceed 1, speedup may exeed number of cores
	- This seems to be fixed be rerunning the program and trying the same values again


- Tools Used
	- CLion IDE by JetBrains
	- Ubuntu - Linux Distribution
- References
	- https://www.investopedia.com/terms/e/efficientmarkethypothesis.asp
		- Holds the Monte Carlo formula, implemented in this programs
	- M. Matsumoto and T. Nishimura, "Mersenne Twister: A 623-dimensionally equidistributed uniform pseudorandom number generator", 
		ACM Trans. on Modeling and Computer Simulation Vol. 8, No. 1, January pp. 3-30 (1998)
	- Makoto Matsumoto and Takuji Nishimura, "Dynamic Creation of Pseudorandom Number Generators", 
		Monte Carlo and Quasi-Monte Carlo Methods 1998, Springer, 2000, pp 56--69.
	- https://stackoverflow.com/questions/16070019/how-to-keep-kernel-code-inside-separate-cu-file-other-than-the-main-cpp
	- https://stackoverflow.com/questions/15855090/create-2d-array-with-cuda
	- https://stackoverflow.com/questions/16599501/cudamemcpy-function-usage/16616738#16616738
	- https://stackoverflow.com/questions/14119088/passing-a-class-object-to-a-kernel
	- https://stackoverflow.com/questions/6978643/cuda-and-classes
	- https://developer.download.nvidia.com/compute/DevZone/docs/html/CUDALibraries/doc/CURAND_Library.pdf
	

