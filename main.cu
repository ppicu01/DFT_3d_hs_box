/*
 *  A classical DFT program that calculates the density profile of hard spheres in a hard box in three dimensions
 *  on graphics cards using C/C++ and the CUDA programming language.
 *  The program employs the original Rosenfeld functional without any correction (such as q3 or tensorial extensions).
 *  It serves as a accompanying information for the paper "Massively parallel GPU-accelerated minimization of classical density functional theory".
 *
 *  The program is a minimal working example and should serve as a starting point for those who aim
 *  to employ GPUs for significantly speeding up their 2D or 3D DFT calculations.
 *  For sake of readability, it is not fully optimized to obtain best performance and less memory usage as possible
 *
 *  The user can choose between single- and double-precision performance. For sufficiently low fluid packing
 *  fractions single precision should always be sufficient.
 *
 *  The equilibrium density profile is calculated via a standard Picard iteration, and convolutions are calculate via Fourier-methods (CUFFT).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <stdarg.h>
#include <limits>
#include <limits.h>
#include <iostream>
#include <cufft.h>

/// MACROS ////////////////////////////////////////////////////////////////////

#define BLOCKS  1024 //NUMBER OF BLOCKS
#define THREADS 128  //THREADS PER BLOCK

#define pi_4 (12.5663706143591729539)  /* 4*pi */
#define pi_2 (6.2831853071795864769)   /* 2*pi */

#define R 1.0 //hard-sphere radius
#define alpha (0.05) //mixing parameter for Picard-iteration
#define MAX_ITER 2500 //maximal iterations to be performed during Picard iteration
#define EPS 1.e-9 //Convergence threshold for density profile

/// FUNCTION DECLARATIONS ///////////////////////////////////////////////////////

//host functions
void freeDevice();
void define_cufftPlans();
void choose_device();
template <class rtype, class ctype> void allocateMemory();
template <class rtype, class ctype> void iterate();
template <class rtype, class ctype> void weightedDensities();
template <class rtype, class ctype> void variation();
template <class rtype> void writeData(bool fullProfile);

//device functions
template <class rtype, class ctype> __device__  void complexRealImMult(const ctype &a, ctype &c, const rtype &b, bool real);
template <class ctype> __device__  ctype operator+(const ctype &a, const ctype &b);
template <class ctype> __device__  ctype operator-(const ctype &a, const ctype &b);
template <class rtype> __global__ void initDensity(rtype *rho, int N, int M, int L);
template <class rtype> __global__ void initWeightFunctions(rtype *wF, int N, int M, int L);
template <class rtype> __global__ void mixing_step(rtype *rho, rtype* c1, rtype mu, int N, int M, int L);
template <class rtype, class ctype> __global__ void weightedDensitiesFourier(ctype *frho, rtype *fwF, ctype *fwD, int N, int M, int L);
template <class rtype, class ctype> __global__ void variationFourier(ctype *fwD, rtype *fwF, ctype *fc1, int N, int M, int L);

/// VARIABLE DECLARATIONS ///////////////////////////////////////////////////////

int N = 128; //Number of grid points in x-direction
int M = 128; //Number of grid points in y-direction
int L = 128; //Number of grid points in z-direction

double Lx; //System size in x-direction
double Ly; //System size in y-direction
double Lz; //System size in z-direction

double dx; //spatial resolution in x-direction
double dy; //spatial resolution in y-direction
double dz; //spatial resolution in z-direction

bool running = true; //flag controlling the termination of the Picard iteration
bool precision = 0; //Sets the precision by user input
bool fullProfile = 0; //flag controlling whether full 3D program should be written or not
int	iterCount = 0; //counter variable for Picard iteration
double diff = 0.;
double eta, rhob; //fluid packing fraction and density (always in double precision)

//host arrays (real)
void *rho;
void *wD;
void *c1;
void *blockSum;

//device arrays (real)
void *rho_dev;
void *rho_old_dev;
void *wD_dev;
void *fwF_dev;
void *c1_dev;
void *blockSum_dev;

//device arrays (complex)
void *frho_dev;
void *fc1_dev;
void *fgrad_rho_dev;
void *fgrad_c1_dev;
void *fwD_dev;

//FFT plans
cufftHandle plan_3d_forward;
cufftHandle plan_3d_inverse;

cufftHandle plan_3d_many_5_forward;
cufftHandle plan_3d_many_5_inverse;

cufftHandle plan_3d_many_3_forward;
cufftHandle plan_3d_many_3_inverse;


/// FUNCTION DEFINITIONS ///////////////////////////////////////////////////////

//Multiplication of complex number with purely real number or purely imaginary number
template <class rtype, class ctype> __device__  void complexRealImMult(const ctype &a, ctype &c, const rtype &b, bool real)
{
	if(real)
	{
		c.x = a.x*b;
		c.y = a.y*b;
	}
	else
	{
		c.x = -a.y*b;
		c.y = a.x*b;
	}
}

//Overload the '+' operator to define addition of complex numbers
template <class ctype> __device__  ctype operator+(const ctype &a, const ctype &b)
{
	ctype res;

	res.x = a.x + b.x;
	res.y = a.y + b.y;

	return res;
}

//Overload the '-' operator to define subtraction of complex numbers
template <class ctype> __device__  ctype operator-(const ctype &a, const ctype &b)
{
	ctype res;

	res.x = a.x - b.x;
	res.y = a.y - b.y;

	return res;
}

//very inefficient solution to correct the density at the walls ;-), but has not much impact on total computation time
template <class rtype>
__global__ void prepareDensity(rtype *rho, bool forward, int N, int M, int L, double dx, double dy, double dz)
{
	int threadIndex = threadIdx.x + blockIdx.x*blockDim.x;
	int numOfThreads = blockDim.x*gridDim.x;

	int i, j, k, ijk, dummy;
	int SIZE = N*M*L;

	int R_N_x = (int) (R/dx);
	int R_N_y = (int) (R/dy);
	int R_N_z = (int) (R/dz);

	for(ijk = threadIndex; ijk < SIZE; ijk += numOfThreads)
	{
		dummy = ijk;
		i = dummy/(M*L);
		dummy -= i*M*L;
		j = dummy/L;
		k = dummy % L;

		if(forward)
		{
			if(i == 2*R_N_x || j == 2*R_N_y || k == 2*R_N_z) rho[ijk] *= 0.5;

			if(i == N - 2*R_N_x -1 || j == M - 2*R_N_y -  1|| k == L - 2*R_N_z - 1) rho[ijk] *= 0.5;
		}
		else
		{
			if(i == 2*R_N_x || j == 2*R_N_y || k == 2*R_N_z) rho[ijk] /= 0.5;

			if(i == N - 2*R_N_x - 1 || j == M - 2*R_N_y - 1 || k == L - 2*R_N_z - 1) rho[ijk] /= 0.5;
		}
	}
}

//initialize the density within a hard box
template <class rtype>
__global__ void initDensity(rtype *rho, int N, int M, int L, double dx, double dy, double dz, double rhob)
{
	int threadIndex = threadIdx.x + blockIdx.x*blockDim.x;
	int numOfThreads = blockDim.x*gridDim.x;

	int i, j, k, ijk, dummy;

	int R_N_x = (int) (R/dx);
	int R_N_y = (int) (R/dy);
	int R_N_z = (int) (R/dz);

	for(ijk = threadIndex; ijk < N*M*L; ijk += numOfThreads)
	{
		dummy = ijk;
		i = dummy/(M*L);
		dummy -= i*M*L;
		j = dummy/L;
		k = dummy % L;

		if((i >= 2*R_N_x && i < N - 2*R_N_x) && (j >= 2*R_N_y && j < M - 2*R_N_y) && (k >= 2*R_N_z && k < L - 2*R_N_z))
			rho[ijk] = rhob;
		else
			rho[ijk] = 0.0;
	}
}


//initialize the weight functions in Fourier-space
template <class rtype>
__global__ void initWeightFunctions(rtype *wF, int N, int M, int L, double dx, double dy, double dz)
{
	int threadIndex = threadIdx.x + blockIdx.x*blockDim.x;
	int numOfThreads = gridDim.x*blockDim.x;

	int i, j, k, ijk, dummy;
	rtype kx,ky,kz, k_abs;
	kx = ky = kz = 0.;

	rtype norm = 1./(1.*N*M*L); //this is needed to have the correct normalization after calculating f(r) = FT^(-1) (FT(h(r)*FT(g(r))
	int SIZE2 = N*M*(L/2 + 1);

	for(ijk = threadIndex; ijk < SIZE2; ijk += numOfThreads)
	{
		//extract indices i,j,k from ijk = k + j*DIM(Z) + i*DIM(Y)*DIM(Z)
		dummy = ijk;
		i = dummy/(M*(L/2 + 1));
		dummy -= i*M*(L/2 + 1);
		j = dummy/(L/2 + 1);
		k = dummy % (L/2 + 1);

		if(i <= N/2) kx = pi_2*R*i/(N*dx);
		else kx = -pi_2*(N-i)*R/(N*dx);

		if(j <= M/2) ky = pi_2*R*j/(M*dy);
		else ky = -pi_2*(M-j)*R/(M*dy);

		 kz = pi_2*R*k/(L*dz);

		k_abs = sqrt(kx*kx + ky*ky + kz*kz);

		if(k_abs == 0)
		{
			wF[ijk] = norm*pi_4*R*R*R/3.;
			wF[ijk + SIZE2] = norm*pi_4*R*R;
			wF[ijk + 2*SIZE2] = 0.0;
			wF[ijk + 3*SIZE2] = 0.0;
			wF[ijk + 4*SIZE2] = 0.0;
		}
		else
		{
			wF[ijk] = norm*pi_4*(sin(k_abs*R) - k_abs*R*cos(k_abs*R))/(k_abs*k_abs*k_abs); //w3
			wF[ijk + SIZE2] = norm*pi_4*R*sin(k_abs*R)/k_abs; //w2
			wF[ijk + 2*SIZE2] = -kx*wF[ijk]; //w2vx
			wF[ijk + 3*SIZE2] = -ky*wF[ijk]; //w2vy
			wF[ijk + 4*SIZE2] = -kz*wF[ijk]; //w2vz
		}
	}
}

//calculate the convolution of density and the weight functions in Fourier-space
template <class rtype, class ctype>
__global__ void weightedDensitiesFourier(ctype *frho, rtype *fwF, ctype *fwD, int N, int M, int L)
{
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	int numOfThreads = blockDim.x*gridDim.x;

	int ijk;
	ctype frho_;
	int SIZE2 = N*M*(L/2 + 1);

	for(ijk = index; ijk < SIZE2; ijk += numOfThreads)
	{
		frho_ = frho[ijk];

		complexRealImMult<rtype, ctype>(frho_, fwD[ijk], fwF[ijk], true);
		complexRealImMult<rtype, ctype>(frho_, fwD[ijk + SIZE2], fwF[ijk + SIZE2], true);
		complexRealImMult<rtype, ctype>(frho_, fwD[ijk + 2*SIZE2], fwF[ijk + 2*SIZE2], false);
		complexRealImMult<rtype, ctype>(frho_, fwD[ijk + 3*SIZE2], fwF[ijk + 3*SIZE2], false);
		complexRealImMult<rtype, ctype>(frho_, fwD[ijk + 4*SIZE2], fwF[ijk + 4*SIZE2], false);
	}
}

//calculate the direct correlation function c1 in Fourier-space
template <class rtype, class ctype>
__global__ void variationFourier(ctype *fwD, rtype *fwF, ctype *fc1, int N, int M, int L)
{
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	int numOfThreads = blockDim.x*gridDim.x;

	ctype fc1_n3, fc1_n2, fc1_n2vx, fc1_n2vy, fc1_n2vz;

	int ijk;
	int SIZE2 = N*M*(L/2+1);
	for(ijk = index; ijk < SIZE2; ijk += numOfThreads)
	{
		complexRealImMult<rtype, ctype>(fwD[ijk], fc1_n3 , fwF[ijk], true);
		complexRealImMult<rtype, ctype>(fwD[ijk + SIZE2], fc1_n2, fwF[ijk + SIZE2], true);
		complexRealImMult<rtype, ctype>(fwD[ijk + 2*SIZE2], fc1_n2vx, fwF[ijk + 2*SIZE2], false);
		complexRealImMult<rtype, ctype>(fwD[ijk + 3*SIZE2], fc1_n2vy, fwF[ijk + 3*SIZE2], false);
		complexRealImMult<rtype, ctype>(fwD[ijk + 4*SIZE2], fc1_n2vz, fwF[ijk + 4*SIZE2], false);

		fc1[ijk] = (fc1_n3 + fc1_n2 - (fc1_n2vx + fc1_n2vy +fc1_n2vz));
	}
}



// Calculate the derivatives of the Rosenfeld functional
//Note: the derivatives are stored in the array which previously contained the
//weighted densities
template <class rtype>
__global__ void calcDerivative(rtype *wD, int N, int M, int L)
{
	int threadIndex = threadIdx.x + blockIdx.x*blockDim.x;
	int numOfThreads = gridDim.x*blockDim.x;

	rtype n3, n2, n1, n0;
	rtype n2vx, n2vy, n2vz;
	rtype n1vx, n1vy, n1vz;
	rtype n2vDotn2v;
	rtype n2vDotn1v;
	rtype tmp_n3, tmp_n3_sq, tmp_n3_inv;

	rtype dPhin0, dPhin1, dPhin2;
	rtype dPhin2vx, dPhin2vy, dPhin2vz;

	int SIZE = N*M*L;

	for(int ijk = threadIndex; ijk < SIZE; ijk += numOfThreads)
	{
		n3 = wD[ijk];
		n2 = wD[ijk + SIZE];
		n2vx = wD[ijk + 2*SIZE];
		n2vy = wD[ijk + 3*SIZE];
		n2vz = wD[ijk + 4*SIZE];

		if(n3 > 0.999) n3 = 0.999;
		tmp_n3 = 1.-n3;
		tmp_n3_inv = 1./tmp_n3;
		tmp_n3_sq = tmp_n3*tmp_n3;
		n1 = n2/(pi_4); //assumes R = 1
		n0 = n1; //assumes R = 1

		n1vx = n2vx/(pi_4); //assumes R = 1
		n1vy = n2vy/(pi_4);
		n1vz = n2vz/(pi_4);

		n2vDotn2v = n2vx*n2vx + n2vy*n2vy + n2vz*n2vz;
		n2vDotn1v = n2vx*n1vx + n2vy*n1vy + n2vz*n1vz;

		//the following could be further simplified but is kept for sake of readability
		//dPhi/dn0
		dPhin0 =  -log(1 - n3);
		//dPhi/dn1
		dPhin1 = n2/tmp_n3;
		//dPhi/dn2
		dPhin2 = n1/tmp_n3 + (n2*n2 - n2vDotn2v)/(2*pi_4*tmp_n3_sq);
		//Components of dPhi/dn2v
		dPhin2vx = -n1vx*tmp_n3_inv - (n2vx*n2)/(pi_4*tmp_n3_sq);
		dPhin2vy = -n1vy*tmp_n3_inv - (n2vy*n2)/(pi_4*tmp_n3_sq);
		dPhin2vz = -n1vz*tmp_n3_inv - (n2vz*n2)/(pi_4*tmp_n3_sq);

		//Store dPhi/dn3
		wD[ijk] = n0*tmp_n3_inv
				+ (n1*n2 - n2vDotn1v)/tmp_n3_sq
				  + (n2*n2*n2 - 3.*n2*n2vDotn2v)/(3*pi_4*(tmp_n3)*tmp_n3_sq);

		//Store dPhi/dn2 + (1/4pi)*dPhi/dn1 + (1/4pi)*dPhi/dn0
		wD[ijk + SIZE] = dPhin2 + dPhin1/(pi_4) + dPhin0/(pi_4);

		//Store components of dPhi/dn2v + (1/4pi)*dPhi/dn1v
		wD[ijk + 2*SIZE] = dPhin2vx -  n2vx*tmp_n3_inv/(pi_4);
		wD[ijk + 3*SIZE] = dPhin2vy -  n2vy*tmp_n3_inv/(pi_4);
		wD[ijk + 4*SIZE] = dPhin2vz -  n2vz*tmp_n3_inv/(pi_4);
	}
}


//calculate the weighted densities for a given density profile
template <class rtype, class ctype>
void weightedDensities()
{
	if(precision)
		cufftExecD2Z(plan_3d_forward, (double*) rho_dev, (double2*) frho_dev);
	else
		cufftExecR2C(plan_3d_forward, (float*) rho_dev, (float2*) frho_dev);

	weightedDensitiesFourier<rtype, ctype><<<BLOCKS, THREADS>>>((ctype*) frho_dev, (rtype*) fwF_dev, (ctype*) fwD_dev, N, M, L);

	if(precision)
		cufftExecZ2D(plan_3d_many_5_inverse, (double2*) fwD_dev, (double*) wD_dev);
	else
		cufftExecC2R(plan_3d_many_5_inverse, (float2*) fwD_dev, (float*) wD_dev);
}

//calculate the direct correlation function c1
template <class rtype, class ctype>
void variation()
{
	calcDerivative<<<BLOCKS, THREADS>>>((rtype*) wD_dev, N, M, L);

	if(precision)
		cufftExecD2Z(plan_3d_many_5_forward, (double*) wD_dev, (double2*) fwD_dev);
	else
		cufftExecR2C(plan_3d_many_5_forward, (float*) wD_dev, (float2*) fwD_dev);

	variationFourier<<<BLOCKS, THREADS>>>((ctype*) fwD_dev, (rtype*) fwF_dev, (ctype*) fc1_dev, N, M, L);

	if(precision)
		cufftExecZ2D(plan_3d_inverse, (double2*) fc1_dev, (double*) c1_dev);
	else
		cufftExecC2R(plan_3d_inverse, (float2*) fc1_dev, (float*) c1_dev);
}

//Mixing step of the Picard iteration for determining the equilibrium density profile
template <class rtype>
__global__ void mixing_step(rtype *rho, rtype *c1, rtype *rho_old_dev, rtype mu, int N, int M, int L, double rhob)
{
	int threadIndex = threadIdx.x + blockDim.x*blockIdx.x;
	int numOfThreads = gridDim.x*blockDim.x;

	rtype dummy = 0.0;
	rtype rho_;
	int SIZE = N*M*L;

	for(int ijk = threadIndex; ijk < SIZE; ijk += numOfThreads)
	{
		rho_ = rho[ijk];
		rho_old_dev[ijk] = rho_;
		if(rho_ > 0)
		{
			dummy = rhob*exp(-c1[ijk] + mu);
			rho[ijk] = (1. - alpha)*rho_ + alpha*dummy;
		}
	}
}

//check whether the density profiles converges or not:
//we calculate the square of the euclidian norm of the new
//density profile vs the old density profile
template <class rtype>
__global__ void checkConvergence(rtype *rho, rtype *rho_old, rtype *block_sum, int N, int M, int L)
{
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	int numOfThreads = gridDim.x*blockDim.x;

	int threadIndex = threadIdx.x;

	__shared__ rtype thread_sum[THREADS]; //use fast shared memory
	thread_sum[threadIndex] = 0.0;

	rtype tmpSum = 0.;
	rtype tmp = 0.;

	int ijk;
	int SIZE = N*M*L;
	for(ijk = index; ijk < SIZE; ijk += numOfThreads)
	{
		tmp = (rho[ijk] - rho_old[ijk]);
		tmp *= tmp;
		tmpSum += tmp;
	}
	thread_sum[threadIndex] = tmpSum; //store intermediate result in shared memory
	__syncthreads();

	int s = blockDim.x/2;
	while(s != 0)   //sum results in shared memory up; final reslt will reside in thread_sum[0]
	{
		if(threadIndex < s)
		{
			thread_sum[threadIndex] += thread_sum[threadIndex + s];
		}
		s /= 2;
		__syncthreads();
	}

	//store each sum in block_sum dependent on block id
	if(threadIndex == 0)
	{
		block_sum[blockIdx.x] = thread_sum[0];
	}
	__syncthreads();
}


//check whether the solution is converged
template <class rtype>
void converged()
{
	diff = 0.0;
	checkConvergence<rtype><<<BLOCKS, THREADS>>>((rtype*) rho_dev, (rtype*) rho_old_dev, (rtype*) blockSum_dev, N, M, L);
	cudaMemcpy((rtype*) blockSum, (rtype*) blockSum_dev, sizeof(rtype)*BLOCKS, cudaMemcpyDeviceToHost);
	for(int i = 0; i < BLOCKS; i++)
	{
		diff += ((rtype*) blockSum)[i];
	}
	diff *= dx*dy*dz;
	if(diff < EPS || iterCount > MAX_ITER || diff != diff) running = false;
}

// Percus-Yevick excess chemical potential
double muPY()
{
	double hc = eta/(1. - eta);
	return hc*(7.0 + hc*(7.5 + 3*hc)) - log(1-eta);
}

// Percus-Yevick pressure
double pPY()
{
	return rhob*(1. + eta + eta*eta)/pow(1-eta, 3);
}

//Picard iteration for determining the equilibrium density profile
template <class rtype, class ctype>
void iterate()
{
	printf("/////////////////////////////////////////////// \n");
	printf("Picard iteration ... \n");
	iterCount = 0;
	running = true;

	initDensity<rtype><<<BLOCKS, THREADS>>>((rtype*) rho_dev, N, M, L, dx, dy, dz, rhob);
	initWeightFunctions<rtype><<<BLOCKS, THREADS>>>((rtype*) fwF_dev, N, M, L, dx, dy, dz);

	double mu = muPY();
	do
	{
		prepareDensity<rtype><<<BLOCKS, THREADS>>>((rtype*) rho_dev, true, N, M, L, dx, dy, dz); //correct the density
	    weightedDensities<rtype, ctype>(); //weighted densities
		variation<rtype, ctype>(); //calcualte c(1)
		prepareDensity<rtype><<<BLOCKS, THREADS>>>((rtype*) rho_dev, false, N, M, L, dx, dy, dz); //recorrect the density
		mixing_step<rtype><<<BLOCKS, THREADS>>>((rtype*) rho_dev, (rtype*) c1_dev, (rtype*) rho_old_dev, (rtype) mu, N, M, L, rhob); //calculate the new profile
		converged<rtype>(); //check whether the solution has converged
		if(iterCount % 10 == 0) printf("# iteration: %i; diff: %1.10e \n", iterCount, diff);
		iterCount++;

	} while(running);
	printf("Converged after %i iterations! \n", iterCount);
	printf("/////////////////////////////////////////////// \n\n");
	weightedDensities<rtype, ctype>();

}

//write the data to file
template <class rtype>
void writeData(bool fullProfile)
{
	FILE* fP;
	char fln[300];

	//print density along one axis (z-Axis)
	sprintf(fln, "rho_1d_box_%1.2f_%1.2f_%1.2f_%1.3f.dat", Lx, Ly, Lz, eta);
	fP = fopen(fln, "w");
	fprintf(fP, "#Density profile in a box with dimension L^3 along the z-axis (L) where x and y are fixed at x = N/2 and y = M/2 \n");
	unsigned int i,j,k,dummy;
	int R_N_z = (int) (R/dz);
	for(unsigned int ijk = 0; ijk < N*M*L; ijk++)
	{
		dummy = ijk;
		i = dummy/(M*L);
		dummy -= i*M*L;
		j = dummy/L;
		k = dummy % L;

		if((i == N/2) && (j == M/2))
		{
			fprintf(fP,"%1.9e %1.9e %1.9e  \n", (k - L/2.)*dx, ((rtype*) rho)[ijk], ((rtype*) wD)[ijk]);
		}
		if((i == N/2-1) && (j == M/2-1) && k == 2*R_N_z)
			printf("contact: %f \n", ((rtype*) rho)[ijk]);

	}
	printf("bulk pressure: %f \n", pPY());
	fprintf(fP, "\n\n");
	fclose(fP);

	//print 2D density in a plane
	sprintf(fln, "rho_2d_box_%1.2f_%1.2f_%1.2f_%1.3f.dat", Lx, Ly, Lz, eta);
	fP = fopen(fln, "w");
	for(unsigned int ijk = 0; ijk < N*M*L; ijk++)
	{
		dummy = ijk;
		i = dummy/(M*L);
		dummy -= i*M*L;
		j = dummy/L;
		k = dummy % L;

		if((i == N/2))
		{
			fprintf(fP,"%1.9e %1.9e %1.9e  \n", (k - L/2.)*dz, (j - M/2.)*dy, ((rtype*) rho)[ijk]);
		}
	}
	fclose(fP);
	

	if(fullProfile)  //write full density as binary file (to be used e.g. with ImageJ)
	{
		sprintf(fln,"rho_3D_%i_%i_%i_%1.3f", N, M, L, eta);
		FILE *pfile = fopen(fln,"wb");
		fwrite(&((rtype*) rho)[0], sizeof(double), N*M*L, pfile);
		fclose(pfile);
	}
}

//Allocate memory on the device and host
template <class rtype, class ctype>
void allocateMemory()
{
	rho = (rtype*) malloc(sizeof(rtype)*N*M*L);
	c1 = (rtype*) malloc(sizeof(rtype)*N*M*L);
	wD = (rtype*) malloc(sizeof(rtype)*N*M*L*5);
	blockSum = (rtype*) malloc(sizeof(rtype)*BLOCKS);

	cudaMalloc((void**) &blockSum_dev, sizeof(rtype)*BLOCKS);
	cudaMalloc((void**) &rho_dev, sizeof(rtype)*N*M*L);
	cudaMalloc((void**) &rho_old_dev, sizeof(rtype)*N*M*L);
	cudaMalloc((void**) &c1_dev, sizeof(rtype)*N*M*L);
	cudaMalloc((void**) &wD_dev, sizeof(rtype)*N*M*L*5);
	cudaMalloc((void**) &fwF_dev, sizeof(rtype)*N*M*(L/2 + 1)*5);

	cudaMalloc((void**) &frho_dev, sizeof(ctype)*N*M*(L/2 + 1));
	cudaMalloc((void**) &fc1_dev, sizeof(ctype)*N*M*(L/2 + 1));
	cudaMalloc((void**) &fwD_dev, sizeof(ctype)*N*M*(L/2 + 1)*5);
}

//allocate the cufft plans depending on chosen precision
void define_cufftPlans()
{
	int n[3] = {N, M, L};

	if(precision)
	{
		cufftPlan3d(&plan_3d_forward, N, M, L, CUFFT_D2Z);
		cufftPlan3d(&plan_3d_inverse, N, M, L, CUFFT_Z2D);
		cufftPlanMany(&plan_3d_many_5_inverse, 3, n, NULL, 1, M*N*(L/2 + 1), NULL, 1, M*N*L, CUFFT_Z2D, 5);
		cufftPlanMany(&plan_3d_many_5_forward, 3, n, NULL, 1, M*N*L, NULL, 1, N*M*(L/2 + 1), CUFFT_D2Z, 5);
	}
	else
	{
		cufftPlan3d(&plan_3d_forward, N, M, L, CUFFT_R2C);
		cufftPlan3d(&plan_3d_inverse, N, M, L, CUFFT_C2R);
		cufftPlanMany(&plan_3d_many_5_inverse, 3, n, NULL, 1, M*N*(L/2 + 1), NULL, 1, M*N*L, CUFFT_C2R, 5);
		cufftPlanMany(&plan_3d_many_5_forward, 3, n, NULL, 1, M*N*L, NULL, 1, N*M*(L/2 + 1), CUFFT_R2C, 5);
	}
}

//free allocated memory
void freeDevice()
{
	cudaFree(rho_dev);
	cudaFree(wD_dev);
	cudaFree(fwF_dev);
	cudaFree(fwD_dev);
	cudaFree(frho_dev);
	cudaFree(c1_dev);
	cudaFree(fc1_dev);
	cudaFree(blockSum_dev);

	free(rho);
	free(wD);
	free(c1);
	free(blockSum);
}

void choose_device()
{
	double globalMem, maxGlobalMem, minUsedMem;
	size_t free_bytes, total_bytes;
	maxGlobalMem = minUsedMem = globalMem = 0.;
	int device = 0;

	//Choosing CUDA device based on device properties
	int numberDevices = 0;
	cudaGetDeviceCount(&numberDevices);
	if(numberDevices == 0)
	{
		printf("ERROR: NO CUDA DEVICE FOUND! \n");
		exit(1);
	}
	minUsedMem = (double) INT_MAX;
	printf("\n ///////////////////////////////////////////////  \n");
	printf("CUDA device properties \n");
	for(int i = 0; i < numberDevices; i++)
	{
		cudaSetDevice(i);
		cudaDeviceProp propDev;
		cudaGetDeviceProperties(&propDev, i);
		cudaMemGetInfo(&free_bytes, &total_bytes);
		double usedMem = ((double) total_bytes - (double) free_bytes)/1024/1024; //in megabytes
		globalMem = (double) propDev.totalGlobalMem/1024./1024.; //in megabytes

		printf("# Device: %i; Curr. used mem.: %1.2f MB; Global mem.: %1.2f MB \n", i, usedMem, globalMem);
		if(globalMem > maxGlobalMem && usedMem < minUsedMem)
		{
			maxGlobalMem = globalMem;
			minUsedMem = usedMem;
			device = i;
		}
	}
	printf("Chosen CUDA device: %i \n", device);
	printf("/////////////////////////////////////////////// \n\n");
	cudaSetDevice(device);
}


int main(int argc, char **argv)
{
	//get the input parameters
	eta = atof(argv[1]);
	Lx = atof(argv[2]);
	Ly = atof(argv[3]);
	Lz = atof(argv[4]);
	precision = atoi(argv[5]);
	fullProfile = atoi(argv[6]);

	//set the spatial resolution
	dx = Lx/N;
	dy = Ly/M;
	dz = Lz/L;

	//set the bulk density
	rhob = 3*eta/(pi_4*R*R*R);

	printf("System size: Lx = %1.3f R; Ly = %1.3f R; Lz = %1.3f R \n", Lx, Ly, Lz);
	printf("Spatial resolution: dx = %1.5f R; dy = %1.5f R; dz = %1.5f R \n", dx, dy, dz);
	printf("Number of grid points: N_x = %i; N_y = %i; N_z = %i \n", N, M, L);

	//choose CUDA device based on memory occupation and amount of global memory
	choose_device();

	define_cufftPlans();

	if(!precision)
	{
		allocateMemory<float, float2>();

		clock_t time_t = clock(),time_end;
		iterate<float, float2>();
		time_end=clock();
		printf("Done after %2.6f seconds \n",(time_end-time_t)/1000000.);

		cudaMemcpy((float*) rho, (float*) rho_dev, sizeof(float)*N*M*L, cudaMemcpyDeviceToHost);
		cudaMemcpy((float*) wD, (float*) wD_dev, sizeof(float)*N*M*L*5, cudaMemcpyDeviceToHost);

		writeData<float>(false);
	}
	else
	{
		allocateMemory<double, double2>();

		clock_t time_t = clock(),time_end;
		iterate<double, double2>();
		time_end=clock();
		printf("Done after %2.6f seconds \n",(time_end-time_t)/1000000.);

		cudaMemcpy((double*) rho, (double*) rho_dev, sizeof(double)*N*M*L, cudaMemcpyDeviceToHost);
		cudaMemcpy((double*) wD, (double*) wD_dev, sizeof(double)*N*M*L*5, cudaMemcpyDeviceToHost);

		writeData<double>(false);
	}
	cudaDeviceSynchronize();


	freeDevice();

	exit(0);
}

