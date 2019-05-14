// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>

void init_array(int *vec, int N, int rank) {
  srand(rank);
  for (int i = 0; i < N; i++) {
    vec[i] = (int)rand();
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, node_count;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &node_count);

  // Number of random numbers per processor (this should be increased
  // for actual tests or could be passed in through the command line
  int N = 100;

  // Read input
  sscanf(argv[1], "%ld", &N);

  int lN = N / node_count;
  if ((N % node_count != 0) && rank == 0) {
    printf("N: %ld, local N: %ld\n", N, lN);
    printf("Exiting. N must be a multiple of node_count\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  int splitter_count = (node_count - 1) * node_count;

  int *vector = (int *)calloc(sizeof(int *), lN);
  int *splitter_vec = (int *)calloc(sizeof(int *), node_count - 1);
  int *global_splitter_vec = (int *)calloc(sizeof(int *), splitter_count);

  init_array(vector, lN, rank); 
  double tt = MPI_Wtime();

  // select local splitters
  for (int i = 0; i < node_count - 1; i++) {
    splitter_vec[i] = vector[i];
  }

  // sort random numbers
  std::sort(vector, vector + lN); 

  //Gather all splitter_vec to rank == 0
  MPI_Gather(splitter_vec, node_count - 1, MPI_INT, global_splitter_vec, node_count - 1, MPI_INT, 0, MPI_COMM_WORLD);

  // select global splitter
  if (rank == 0) { 
    std::sort(global_splitter_vec, global_splitter_vec + splitter_count);
    int gap = (splitter_count - 1) / (node_count - 1);
    for (int i = 1; i <= node_count - 1; i++) {
      splitter_vec[i - 1] = vector[i * gap];
    }
    std::sort(splitter_vec, splitter_vec + node_count - 1);
  }

  // Broadcast splitters
  MPI_Bcast(splitter_vec, node_count - 1, MPI_INT, 0, MPI_COMM_WORLD);

  int *send_dis = (int *)calloc(sizeof(int *), node_count);
  int *bcounts = (int *)calloc(sizeof(int *), node_count);
  int *count_vec = (int *)calloc(sizeof(int *), node_count);
  int *recv_dis = (int *)calloc(sizeof(int *), node_count);
  int *buckets = (int *)calloc(sizeof(int *), 2 * lN);

  int temp = 0;
  send_dis[0] = 0;

  for (int i = 0; i < node_count - 1; i++) { 
    send_dis[i + 1] = std::lower_bound(vector, vector + lN, splitter_vec[i]) - vector;
    bcounts[i] = send_dis[i + 1] - temp;
    temp = send_dis[i + 1];
  }
  bcounts[node_count - 1] = lN - temp;

  MPI_Alltoall(bcounts, 1, MPI_INT, count_vec, 1, MPI_INT, MPI_COMM_WORLD);
  for (int i = 1; i < node_count; i++) {
    recv_dis[i] = count_vec[i - 1] + recv_dis[i - 1]; 
  }
  MPI_Alltoallv(vector, bcounts, send_dis, MPI_INT, buckets, count_vec, recv_dis, MPI_INT, MPI_COMM_WORLD);

  int total = 0;
  for (int i = 0; i < node_count; i++) { 
    total += count_vec[i];
  }

  std::sort(buckets, buckets + total); 

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (rank == 0) {
    printf("Time elapsed is %f seconds. \n", elapsed);
  }

  // Write to files
  FILE *fd = NULL;
  char filename[256];
  snprintf(filename, 256, "output%02d.txt", rank);
  fd = fopen(filename, "w+");
  if (NULL == fd) {
    printf("Error opening file \n");
    return 1;
  }
  for (int i = 0; i < 2 * lN; i++) { 
    if (buckets[i] < buckets[i - 1]) { 
      break;
    }
    fprintf(fd, "  %d\n", buckets[i]);
  }
  fclose(fd);

  free(vector);
  free(splitter_vec);
  free(global_splitter_vec);
  free(send_dis);
  free(bcounts);
  free(count_vec);
  free(recv_dis);
  free(buckets);

  MPI_Finalize();
  return 0;
}
