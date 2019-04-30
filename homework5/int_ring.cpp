#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <iostream>

double time_latency(long Nrepeat, long Nsize, MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  int processCount;
  MPI_Comm_size(MPI_COMM_WORLD, &processCount);

  MPI_Barrier(comm);
  double tt = MPI_Wtime();

  int msgIn; 
  int msgOut = rank; 

  for (long repeat = 0; repeat < Nrepeat; repeat++) {
  	int sender = (rank == 0) ? (processCount - 1) : (rank - 1); 
  	int receiver = (rank == (processCount - 1)) ? 0 : (rank + 1); 
	MPI_Status status;

  	if (!(repeat == 0 && rank == 0)) {
  		int tag = (rank == 0) ? (repeat - 1) : repeat; 
  		MPI_Recv(&msgIn, Nsize, MPI_INT, sender, tag, comm, &status);
  		msgOut = msgIn + rank; 
  	}

  	if (!(repeat == (Nrepeat - 1) && (rank == processCount - 1))) {
  		MPI_Send(&msgOut, Nsize, MPI_INT, receiver, repeat, comm);
  	}
  }

  tt = MPI_Wtime() - tt;

  MPI_Barrier(comm);

  if (rank == processCount - 1) {
  	printf("actual message count: %d \n", msgIn); 
  }
  return tt;
}

double time_bandwidth(long Nrepeat, long Nsize, MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  int processCount;
  MPI_Comm_size(MPI_COMM_WORLD, &processCount);

  // int* msg = (int*) malloc(sizeof(int) * Nsize);
  int msg[sizeof(int) * Nsize]; 
  for (long i = 0; i < Nsize; i++) {
  	msg[i] = 0; 
  }

  // printf("finish inintialization, rank %d \n", rank); 

  MPI_Barrier(comm);
  double tt = MPI_Wtime();

  for (long repeat = 0; repeat < Nrepeat; repeat++) {
  	int sender = (rank == 0) ? (processCount - 1) : (rank - 1); 
  	int receiver = (rank == (processCount - 1)) ? 0 : (rank + 1); 
	MPI_Status status;

  	if (!(repeat == 0 && rank == 0)) {
  		MPI_Recv(&msg, Nsize, MPI_INT, sender, 0, comm, &status);
  	}

  	if (!(repeat == (Nrepeat - 1) && (rank == processCount - 1))) {
  		MPI_Send(&msg, Nsize, MPI_INT, receiver, 0, comm);
  	}
  }

  tt = MPI_Wtime() - tt;

  // free(msg);
  return tt;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  int processCount;
  MPI_Comm_size(MPI_COMM_WORLD, &processCount);

  if (processCount < 2) {
  	printf("Need at lease two nodes for communication \n");
    abort();
  }

  long Nrepeat = 1000;
  double tt = time_latency(Nrepeat, 1, comm);

  int expectedCount = 0; 
  for (int i = 0; i < processCount; i++) {
  	expectedCount += i; 
  }
  expectedCount *= Nrepeat; 
  expectedCount -= (processCount - 1); 


  if (!rank) {
  	printf("Total nodes: %d \n", processCount); 
  	printf("Expeced count: %d \n", expectedCount); 
  	printf("Latency: %e ms\n", tt/Nrepeat * 1000 * processCount);
  }

  Nrepeat = 10000;
  long Nsize = 2e6 / sizeof(int);
  tt = time_bandwidth(Nrepeat, Nsize, comm);
  
  if (!rank) {
  	printf("Array size: %d \n", (Nsize * sizeof(int))); 
  	printf("Bandwidth: %e GB/s\n", (Nsize * Nrepeat * sizeof(int) * processCount)/tt/1e9);
  }

  MPI_Finalize();
}

