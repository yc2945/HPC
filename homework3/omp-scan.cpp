#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {

  int nthreads; 
  int chunk_size; 
  #pragma omp parallel 
  {
    nthreads = omp_get_num_threads();
    double size = (double) n / (double) nthreads;
    double rounded = std::ceil(size);
    chunk_size = (int) rounded;

    #pragma omp for
    for (int i = 0; i < nthreads; i++) {
      int tid = omp_get_thread_num();

      if (tid == 0) {
        nthreads = omp_get_num_threads();
        printf("Number of threads = %d\n", nthreads);
      }
      printf("Thread %d is starting...\n",tid);

      int start = tid * chunk_size; 
      int end; 
      if (tid == nthreads - 1) {
        end = n; 
      } else {
        end = start + chunk_size; 
      }

      for (int j = start; j < end; j++) {
        if (j == 0) {
          prefix_sum[j] = 0; 
        } else {
          prefix_sum[j] = A[j-1]; 
          if (j != start) {
            prefix_sum[j] += prefix_sum[j-1]; 
          }
        }
      }
    }
  }

  for (int i = chunk_size; i < n; i++) {
    int chunk_id = i / chunk_size; 
    prefix_sum[i] += prefix_sum[chunk_id * chunk_size - 1]; 
  }

}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
