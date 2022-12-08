/*
 * _MATRIXMUL_GPU_CU_
 *
 * 2022 Mert SIDE
 *
 * CS5375 Computer Systems Organization and Architecture 
 * Guest Lecture: GPU Programming
 *
 * Multiplying two matrices on the GPU
 *
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

// ------------------------------------------------------------------ GPUmatmul
__global__
void GPUmatmul(int N, double *x, double *y, double *ans)
{
	int t= (blockDim.x*blockDim.y)*threadIdx.z+(threadIdx.y*blockDim.x)+(threadIdx.x);
	int b= (gridDim.x*gridDim.y)*blockIdx.z+(blockIdx.y*gridDim.x)+(blockIdx.x);
	int T= blockDim.x*blockDim.y*blockDim.z;
	int B= gridDim.x*gridDim.y*gridDim.z;
    for (int i=b;i<N;i+=B) {
		for(int j=t;j<N;j+=T){
			for(int k=0;k<N;k++){
				ans[i*N+j]+=(x[i*N+k]*y[k*N+j]);
			}
		}
	}
}

__global__ void init(double *x, double *y, double *ans, int N) 
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
            x[i*N+j] = 5;
            y[i*N+j] = (i==j?1:0);
            ans[i*N+j] = (double)0.000000000000;
}


// ---------------------------------------------------------------------- check
bool check(int N, double *ans)
{
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) {
      if(ans[i*N+j] != 20.0) return false;
    }
  }
  return true;
}

// ----------------------------------------------------------------------- MAIN
int main(void)
{
  // size of matrix
  int N = 1<<9; // binary left-shift: 1 * 2^9 = 512
  printf("Size of matrix (N) is %d by %d.\n", N, N);
  int iter = 3;
  clock_t t;
  
  // Martices
  double *x, *y, *ans;

  // TODO: Allocate Unified Memory - accessible from both CPU and GPU
  cudaMallocManaged(&x, sizeof(double)*N*N);
  cudaMallocManaged(&y, sizeof(double)*N*N);
  cudaMallocManaged(&ans, sizeof(double)*N*N);

  // ..........................................................................
  // initialize x,y and ans arrays on the host
  /*
  for (int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) {
      x[i*N+j] = 5;
      y[i*N+j] = (i==j?1:0);
      ans[i*N+j] = (double)0.000000000000;
    }
  }
  */
    int THREADS = 8;
    int BLOCKS = N / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);
    init<<<blocks, threads>>>(N, x, y, ans);

  // ..........................................................................
  double avg=0;
  std::cout<<"Starting unoptimized GPU computation"<<std::endl;
  // Run kernel on GPU
  for(int i = 0; i <= iter; i++) {
    t = clock();
	GPUmatmul<<<dim3(16,4,4),dim3(8,8,8)>>>(N, x, y, ans);
    cudaDeviceSynchronize();
    t = clock() - t;
    if(i) avg += t; //we will ignore the first run
    // printf ("It took GPU-%d %f ms.\n",i,(((double)t)/CLOCKS_PER_SEC)*1000);
  }

  avg /= iter;
  avg /= CLOCKS_PER_SEC;
  avg *= 1000;
  printf("It took %lf ms on avg.\n", avg);
  if(check(N,ans)) std::cout<<"RUN OK."<<std::endl;
  else std::cout<<"RUN NOT OK."<<std::endl;

  // ..........................................................................
  
  // TODO: Free memory
  cudaFree(x);
  cudaFree(y);
  cudaFree(ans);

  return 0;
}
/* EOF */