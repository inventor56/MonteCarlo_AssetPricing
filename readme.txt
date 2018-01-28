- MONTE CARLO SIMULATION UTILIZING MERSENNE TWISTER PRNG
	- This program is intended to predict fluctuations in asset prices, utilizing random numbers as part of the calculations.
		- Runs a full Monte Carlo simulation, for however many simulations the user desires
	- At the end, if the number of simulations does not exceed 100, the user can export a document readable by Excel
		- We restrict to under 100 due to how big the resulting file would be otherwise

- COMPILATION INSTRUCTIONS
	- To compile from the source code, be sure to compile with C++11, CUDA, and CURAND (for parallel RNG generation)
		- Type
			- nvcc main.cpp MonteCarlo.cpp kernel_cuda.cu -std=c++11 -lcurand
		- Using C+11 support for NVCC as of CUDA version 7.0

- PROGRAM INSTRUCTIONS
	- Type in the name of the dataset you'd like to use for historical data
		- These can be obtained from the Yahoo Finance website, look below for more details
	- Type in 1 to run the simulations serially
		- Will take more time for larger data sets
	- Type in 2 to run the simulations parallelly
		- Will generate random numbers and perform the simulations on CUDA enable Processing
		- Will take more time on very small data sets

- DATA NOTES
	- You can get data from https://finance.yahoo.com/
		- I.E. https://finance.yahoo.com/quote/ge/history/
		- Click on "Download Data" above the Volume


- Tools Used
	- Ubuntu - Linux Distribution
	- CUDA Toolkit - NVIDIA Parallel Processing
	- CLion - IntelliJ IDE

- References
	- https://www.investopedia.com/terms/e/efficientmarkethypothesis.asp
	- https://www.investopedia.com/terms/m/montecarlosimulation.asp
		- Holds the Monte Carlo formula, implemented in this program
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
