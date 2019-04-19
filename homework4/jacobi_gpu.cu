#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <omp.h>


using namespace std;

#define BLOCK_SIZE 64

double compute_residual(const double *F, const double *current_U, const double *A, const int N) {
    double sum = 0.0;

    for (int i = 0; i < N; i++) {
        double total = 0.0;
        for (int j = 0; j < N; j++) {
            int index = i * N + j;
            total += A[index] * current_U[j];
        }

        sum += (total - F[i]) * (total - F[i]);
    }

    return sqrt(sum);
}

__global__ void jacobi_kernel(double *A, double *previous_U, double *sum, const int i, const int N) {
    int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (idx == i) return; 
        (*sum) += A[i * N + idx] * previous_U[idx]; 
    }
}

void compute(double *F,
             double *current_U,
             double *previous_U,
             double *A,
             const int iteration_count,
             const int N,
             const double initial_residual,
             const double factor) {

    double *A_d, *previous_U_d;
    cudaMalloc(&A_d, N*N*sizeof(double));
    cudaMalloc(&previous_U_d, N*sizeof(double));

    for (int iteration = 0; iteration < iteration_count; iteration++) {
        for (int i = 0; i < N; i++) {
            double sum = 0.0;

            cudaMemcpyAsync(A_d, A, N*N*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpyAsync(previous_U_d, previous_U, N*sizeof(double), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            jacobi_kernel<<<(N+BLOCK_SIZE-1)/(BLOCK_SIZE), BLOCK_SIZE>>>(A_d, previous_U_d, &sum, i, N);

            // for (int j = 0; j < N; j++) {
            //     if (j == i) continue;
            //     sum += A[i * N + j] * previous_U[j];
            // }

            current_U[i] = (F[i] - sum) / A[i * N + i];
        }
        double residual = compute_residual(F, current_U, A, N);
    printf("Iteration: %d \n", iteration);
        printf("New residual is: %f \n", residual);
        printf("Factor: %f \n", initial_residual / residual);
        previous_U = current_U;
    }
    cudaFree(A_d);
    cudaFree(previous_U_d);
    printf("Terminating......");
}

int main() {
    int N = 1000;
    int iteration_count = 1000;
    double factor = 1000000.0;
    double h = 1.0 / (N + 1);;
    double initial_residual;
    clock_t t;

    double *current_U, *previous_U, *F, *A; 

    cudaMallocHost((void**)&current_U, N * sizeof(double));
    cudaMallocHost((void**)&previous_U, N * sizeof(double));
    cudaMallocHost((void**)&F, N * sizeof(double));
    cudaMallocHost((void**)&A, N * N * sizeof(double));

    // Initialization
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        F[i] = 1.0;
        current_U[i] = 0.0;
        previous_U[i] = 0.0;
        for (int j = 0; j < N; j++) {
            int index = i * N + j;
            A[index] = 1 / (h * h);
            if (i == j) {
                A[index] *= 2;
            } else if (i + 1 == j || j + 1 == i) {
                A[index] *= -1;
            } else {
                A[index] = 0;
            }
        }
    }

    initial_residual = compute_residual(F, current_U, A, N);
    printf("Initial residual: %f \n", initial_residual);

    t = clock();
    compute(F, current_U, previous_U, A, iteration_count, N, initial_residual, factor);
    cudaFreeHost(current_U);
    cudaFreeHost(previous_U);
    cudaFreeHost(A);
    cudaFreeHost(F);
    t = clock() - t;
    double time_taken = ((double) t) / CLOCKS_PER_SEC; // in seconds

    printf("Total time in seconds: %f \n", time_taken);
    printf("N: %d \n", N); 
    printf("iteration count: %d \n", iteration_count); 
    return 0;
}
