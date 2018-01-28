- To run executable in Linux terminal, navigate to the Executable directory and type ./MersenneTwister
- To compile from the source code, be sure to compile with C++11 and CUDA
	- nvcc main.cpp MonteCarlo.cpp -std=c++11
	- nvcc main.cpp MonteCarlo.cpp kernel_cuda.cu -std=c++11
	- nvcc main.cpp MonteCarlo.cpp kernel_cuda.cu -std=c++11 -lcurand
	- Using C+11 support for NVCC as of CUDA version 7.0



- PROGRAM INSTRUCTIONS
	- Herbie Goes to Monte Carlo

- NOTES
	- You can get data from https://finance.yahoo.com/
		- I.E. https://finance.yahoo.com/quote/ge/history/
		- Click on "Download Data" above the Volume


- Tools Used
	- Ubuntu - Linux Distribution
	- CIDA - NVIDIA Parallel Processing
- References
	- https://www.investopedia.com/terms/e/efficientmarkethypothesis.asp
	- https://www.investopedia.com/terms/m/montecarlosimulation.asp
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
