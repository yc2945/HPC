/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has N unknowns, each processor works with its
 * part, which has lN = N/p unknowns.
 * Author: Georg Stadler
 */
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *lu, int lN, double invhsq){
  int i;
  double tmp, gres = 0.0, lres = 0.0;

  for (i = 1; i <= lN; i++){
    tmp = ((2.0*lu[i] - lu[i-1] - lu[i+1]) * invhsq - 1);
    lres += tmp * tmp;
  }
  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}

void updateU(int maxIterations, int localN, double h, double *localU, double *localUnew, 
          double *leftSendBuf, double *rightSendBuf, double *upSendBuf, double *downSendBuf, 
          double *leftRecvBuf, double *rightRecvBuf, double *upRecvBuf, double *downRecvBuf) {
  int mpirank; 
  int nodeCount; 

  MPI_Status status1, status2, status3, status4;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &nodeCount);

  int gridDimention = sqrt(localN) + 2; // Number of item per row/col in the grid
  int nodeDimention = sqrt(nodeCount); // Number of node per row/col

  int nodeRow = mpirank / nodeDimention; // Row of node in all nodes
  int nodeCol = mpirank % nodeDimention; // Column of node in all nodes

  bool leftBoundary = false; 
  bool rightBoundary = false; 
  bool upBoundary = false; 
  bool downBoundary = false; 
  if (nodeRow == 0) upBoundary = true; 
  if (nodeRow == (nodeDimention - 1)) downBoundary = true; 
  if (nodeCol == 0) leftBoundary = true; 
  if (nodeCol == (nodeDimention - 1)) rightBoundary = true; 

  printf("Rank %d finish running.\n", mpirank);
  printf("Grid dimention %d \n", gridDimention); 

  for (int i = 0; i < gridDimention; i++) {
    printf("%d: %d \n", i, localU[i]); 
  }

  for (int i = 1; i < gridDimention - 1; i++) {
    for (int j = 1; j < gridDimention - 1; j++) {
      int index = i * gridDimention + j; 
      if (i == 1 && nodeRow == 0) localUnew[index] = 0; 
      else if (i == gridDimention - 2 && nodeRow == (nodeDimention - 1)) localUnew[index] = 0;
      else if (j == 1 && nodeCol == 0) localUnew[index] = 0; 
      else if (j == gridDimention - 2 && nodeCol == (nodeDimention - 1)) localUnew[index] = 0;
      else localUnew[index] =  0.25 
        * (h * h + localU[index-1] + localU[index+1] + localU[index-gridDimention] + localU[index+gridDimention]);   
    }
  }

  printf("Rank %d finish main updates.\n", mpirank);



  return; 

  // left tag 1, right tag 2, up tag 3, down tag 4. 
  // if (!leftBoundary) {
  //   for (int i = 0; i < gridDimention; i++) leftSendBuf[i] = localUnew[i * gridDimention + 1];  
  //   MPI_Send(&leftSendBuf, gridDimention, MPI_DOUBLE, mpirank-1, 1, MPI_COMM_WORLD);
  // }
  // printf("Rank %d finish left updates.\n", mpirank);

  // if (!rightBoundary) {
  //   for (int i = 0; i < gridDimention; i++) rightSendBuf[i] = localUnew[i * gridDimention + (gridDimention-2)];  
  //   MPI_Send(&rightSendBuf, gridDimention, MPI_DOUBLE, mpirank+1, 2, MPI_COMM_WORLD);
  // }
  // printf("Rank %d finish right updates.\n", mpirank);

  // if (!upBoundary) {
  //   for (int i = 0; i < gridDimention; i++) upSendBuf[i] = localUnew[gridDimention + i];  
  //   MPI_Send(&upSendBuf, gridDimention, MPI_DOUBLE, mpirank-nodeDimention, 3, MPI_COMM_WORLD);
    
  // }
  // printf("Rank %d finish up updates.\n", mpirank);

  // if (!downBoundary) {
  //   for (int i = 0; i < gridDimention; i++) upSendBuf[i] = localUnew[gridDimention * (gridDimention-2) + i];  
  //   MPI_Send(&downSendBuf, gridDimention, MPI_DOUBLE, mpirank+nodeDimention, 4, MPI_COMM_WORLD);
  // }
  // printf("Rank %d finish down updates.\n", mpirank);



  // if (!leftBoundary) {
  //   MPI_Recv(&leftRecvBuf, gridDimention, MPI_DOUBLE, mpirank-1, 2, MPI_COMM_WORLD, &status1);
  //   // for (int i = 0; i < gridDimention; i++) {
  //   //   printf("Rank %d receiving left with index %d .\n", mpirank, i);
  //   //   localUnew[i * gridDimention + 0] = leftRecvBuf[i]; 
  //   // }
  // }
  // printf("Rank %d finish left rec.\n", mpirank);

  // if (!rightBoundary) {
  //   MPI_Recv(&rightRecvBuf, gridDimention, MPI_DOUBLE, mpirank+1, 1, MPI_COMM_WORLD, &status2);
  //   // for (int i = 0; i < gridDimention; i++) localUnew[i * gridDimention + (gridDimention-1)] = rightRecvBuf[i]; 
  // }
  // printf("Rank %d finish right rec.\n", mpirank);

  // if (!upBoundary) {
  //   MPI_Recv(&upRecvBuf, gridDimention, MPI_DOUBLE, mpirank-nodeDimention, 4, MPI_COMM_WORLD, &status3);
  //   // for (int i = 0; i < gridDimention; i++) localUnew[i] = upRecvBuf[i]; 
  // }
  // printf("Rank %d finish up rec.\n", mpirank);

  // if (!downBoundary) {
  //   MPI_Recv(&downRecvBuf, gridDimention, MPI_DOUBLE, mpirank+nodeDimention, 3, MPI_COMM_WORLD, &status4);
  //   // for (int i = 0; i < gridDimention; i++) localUnew[gridDimention * (gridDimention-1) + i] = upRecvBuf[i]; 
  // }
  // printf("Rank %d finish down rec.\n", mpirank);

  // printf("Rank %d ready to exit.\n", mpirank);
  double *localUtemp = localU; 
  localU = localUnew; 
  localUnew = localUtemp; 
  printf("Rank %d existed.\n", mpirank);
}

void computeAllU(int max_iters, int localN, double h) {

}


int main(int argc, char * argv[]){
  int mpirank, i, p, N, lN, iter, max_iters;
  // MPI_Status status, status1;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

  sscanf(argv[1], "%d", &N);
  sscanf(argv[2], "%d", &max_iters);

  // N = 16; 
  // max_iters = 1; 

  /* compute number of unknowns handled by each process */
  lN = N / p;
  if ((N % p != 0) && mpirank == 0 ) {
    printf("N: %d, local N: %d\n", N, lN);
    printf("Exiting. N must be a multiple of p\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  // int totalGridDimention = sqrt(N) + 2; 
  int gridDimention = sqrt(lN) + 2; 

  /* Allocation of vectors, including left/upper and right/lower ghost points */
  double * lu    = (double *) calloc(sizeof(double), gridDimention * gridDimention);
  double * lunew = (double *) calloc(sizeof(double), gridDimention * gridDimention);
  double * lutemp;

  for (int i = 0; i < gridDimention * gridDimention; i++) {
    lu[i] = i; 
    lunew[i] = i; 
  }

  double *leftSendBuf = (double *) calloc(sizeof(double), gridDimention);  
  double *rightSendBuf = (double *) calloc(sizeof(double), gridDimention);  
  double *upSendBuf = (double *) calloc(sizeof(double), gridDimention);  
  double *downSendBuf = (double *) calloc(sizeof(double), gridDimention);  

  double *leftRecvBuf = (double *) calloc(sizeof(double), gridDimention);  
  double *rightRecvBuf = (double *) calloc(sizeof(double), gridDimention);  
  double *upRecvBuf = (double *) calloc(sizeof(double), gridDimention);  
  double *downRecvBuf = (double *) calloc(sizeof(double), gridDimention);  

  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-5;

  // /* initial residual */
  // gres0 = compute_residual(lu, lN, invhsq);
  // gres = gres0;

  printf("Rank %d/%d finish init.\n", mpirank, p);

  for (int iter = 0; iter < max_iters; i++) {
    if (mpirank == 0) {
      printf("Running iteration %d/%d.\n", iter, max_iters);
    }
    updateU(max_iters, lN, h, lu, lunew,
          leftSendBuf, rightSendBuf, upSendBuf, downSendBuf, 
          leftRecvBuf, rightRecvBuf, upRecvBuf, downRecvBuf); 
    if (mpirank == 0) {
      printf("Finish iteration %d/%d.\n", iter, max_iters);
    }
  }

  printf("To call barrier \n"); 

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;

  printf("barrier called \n"); 

  /* Clean up */
  free(lu);
  free(lunew);
  free(leftSendBuf);
  free(rightSendBuf);
  free(upSendBuf);
  free(downSendBuf);
  free(leftRecvBuf);
  free(rightRecvBuf);
  free(upRecvBuf);
  free(downRecvBuf);

  if (0 == mpirank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();


  return 0;
}
