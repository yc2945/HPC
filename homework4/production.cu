#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

void reduction(double* sum_ptr, const double* a, const double *b, long N){
  double sum = 0;
  #pragma omp parallel for schedule(static) reduction(+:sum)
  for (long i = 0; i < N; i++) sum += a[i] * b[i];
  *sum_ptr = sum;
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

#define BLOCK_SIZE 1024


__global__ void reduction_kernel2(double* sum, const double* a, const double *b, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx] * b[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
  }
}

void production(double* sum_ref, const double* a, const double* b, long N) {
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < N; i++) {
    double sum = 0.0; 
    for (int j = 0; j < N; j++) {
      sum += a[i * N + j] * b[j]; 
    }
    sum_ref[i] = sum; 
  }
}

__global__ void production_kernel(const double* a, const double* b, double* c, long N) {
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < N * N) {
    c[idx] = a[idx] * b[idx % N]; 
  }
}

__global__ void sum_kernel(const double* c, double* sum_ref, long N) {
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < N) {
    double sum = 0.0; 
    for (int i = 0; i < N; i++) {
      sum += c[idx * N + i]; 
    }
    sum_ref[N] = sum; 
  }
}

int main() {
  long N = (1UL<<12);
  // long N = 100; 

  double *x;
  double *y; 
  double *sum_ref; 
  cudaMallocHost((void**)&x, N * N * sizeof(double));
  cudaMallocHost((void**)&y, N * sizeof(double));
  cudaMallocHost((void**)&sum_ref, N * sizeof(double));

  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N * N; i++) x[i] = 1.0/(i+1);
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) y[i] = 1.0/(i+1); 
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) sum_ref[i] = 0.0; 

  

  double tt = omp_get_wtime();
  production(sum_ref, x, y, N);
  printf("CPU Bandwidth = %f GB/s\n", N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double *x_d, *y_d, *z_d, *sum_d; 
  cudaMalloc(&x_d, N * N * sizeof(double));
  cudaMalloc(&y_d, N * 1 * sizeof(double));
  cudaMallocHost((void**)&sum_d, N * sizeof(double));
  cudaMallocHost((void**)&z_d, N * N * sizeof(double));

  // cudaMalloc(&z_d, N * N * sizeof(double));
  // cudaMalloc(&sum_d, N * 1 * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) sum_d[i] = 0.0; 
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N * N; i++) z_d[i] = 0.0; 

  cudaMemcpyAsync(x_d, x, N * N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(y_d, y, N * 1 * sizeof(double), cudaMemcpyHostToDevice); 
  cudaDeviceSynchronize();
  tt = omp_get_wtime();

  long Nb = (N * N + BLOCK_SIZE - 1) / (BLOCK_SIZE);
  production_kernel<<<Nb,BLOCK_SIZE>>>(x_d, y_d, z_d, N);
  // Check_CUDA_Error();

  Nb = (N + BLOCK_SIZE - 1) / (BLOCK_SIZE);
  sum_kernel<<<Nb,BLOCK_SIZE>>>(z_d, sum_d, N);

  printf("GPU Bandwidth = %f GB/s\n", N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  

  double total = 0.0; 
  double error = 0.0; 
  for (long i = 0; i < N; i++) {
    // printf("index: %ld \n", i); 
    total += fabs(sum_d[i] - sum_ref[i]); 
  }

  printf("Error = %f\n", error);

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d); 
  cudaFree(sum_d); 

  cudaFreeHost(x);
  cudaFreeHost(y);
  cudaFreeHost(sum_ref);

  return 0;
}

