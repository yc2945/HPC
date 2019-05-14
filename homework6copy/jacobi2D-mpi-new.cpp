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
double compute_residual(double *lu, int gridDimention, double invhsq){
  double tmp, gres = 0.0, lres = 0.0;
  for (long i = 0; i < gridDimention * gridDimention; i++){
      bool bounndary = false;
      if(i <= gridDimention) bounndary = true; 
      if(i % gridDimention == 0) bounndary = true; 
      if(i % gridDimention == (gridDimention-1)) bounndary = true; 
      if(i >= gridDimention * gridDimention - gridDimention) bounndary = true; 
      
      if(!bounndary){
        tmp = ((4.0*lu[i] - lu[i-1] - lu[i+1] - lu[i + gridDimention] - lu[i - gridDimention]) * invhsq - 1);
        lres += tmp * tmp;
      }
  }
  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}

void computeAllU(int max_iters, int localN, double h) {
  double invhsq = 1.0 / (h * h); 
  double gres; 

  MPI_Status status1, status2, status3, status4;
  int mpirank, nodeCount; 
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

  /* Allocation of vectors, including left/upper and right/lower ghost points */
  double * localU    = (double *) calloc(sizeof(double), gridDimention * gridDimention);
  double * localUnew = (double *) calloc(sizeof(double), gridDimention * gridDimention);
  double * lutemp;

  double leftSendBuf[gridDimention]; 
  double rightSendBuf[gridDimention]; 
  double leftRecvBuf[gridDimention]; 
  double rightRecvBuf[gridDimention]; 

  double upSendBuf[gridDimention]; 
  double downSendBuf[gridDimention]; 
  double upRecvBuf[gridDimention]; 
  double downRecvBuf[gridDimention]; 

  for (long i = 0; i < gridDimention * gridDimention; i++){
      bool boundary = false;
      if((i < gridDimention) && (mpirank < nodeDimention)) {boundary = true;}
      if((i % gridDimention == 0) && (mpirank%nodeDimention == 0)) {boundary = true;}
      if((i % gridDimention == (gridDimention-1)) && ((mpirank+1)%(nodeDimention) == 0)) {boundary = true;}
      if((i >= gridDimention*gridDimention - gridDimention) && (mpirank >= nodeDimention*nodeDimention - nodeDimention)) {boundary = true;}
      
      if (!boundary) {
        localU[i] = 1;
      } else{
        localU[i] = 0;
      }
  }

  for (int iter = 0; iter < max_iters; iter++) {
      for (int i = 0; i < gridDimention; i++) {
        for (int j = 0; j < gridDimention; j++) {
          int index = i * gridDimention + j; 

          if (i == 0) localUnew[index] = 0; 
          else if (i == gridDimention - 1) localUnew[index] = 0;
          else if (j == 0) localUnew[index] = 0; 
          else if (j == gridDimention - 1) localUnew[index] = 0;
          else localUnew[index] =  0.25 
            * (h * h + localU[index-1] + localU[index+1] + localU[index-gridDimention] + localU[index+gridDimention]);   

        }
      }

      // left tag 1, right tag 2, up tag 3, down tag 4. 
      if (!leftBoundary) {
        for (int i = 0; i < gridDimention; i++) leftSendBuf[i] = localUnew[i * gridDimention + 1];  
        MPI_Send(leftSendBuf, gridDimention, MPI_DOUBLE, mpirank-1, 1, MPI_COMM_WORLD);
      }
      if (!rightBoundary) {
        for (int i = 0; i < gridDimention; i++) rightSendBuf[i] = localUnew[i * gridDimention + (gridDimention-2)];  
        MPI_Send(rightSendBuf, gridDimention, MPI_DOUBLE, mpirank+1, 2, MPI_COMM_WORLD);
      }
      if (!upBoundary) {
        for (int i = 0; i < gridDimention; i++) upSendBuf[i] = localUnew[gridDimention + i];  
        MPI_Send(&upSendBuf, gridDimention, MPI_DOUBLE, mpirank-nodeDimention, 3, MPI_COMM_WORLD);
        
      }
      if (!downBoundary) {
        for (int i = 0; i < gridDimention; i++) upSendBuf[i] = localUnew[gridDimention * (gridDimention-2) + i];  
        MPI_Send(&downSendBuf, gridDimention, MPI_DOUBLE, mpirank+nodeDimention, 4, MPI_COMM_WORLD);
      }


      if (!leftBoundary) {
        MPI_Recv(leftRecvBuf, gridDimention, MPI_DOUBLE, mpirank-1, 2, MPI_COMM_WORLD, &status1);
        for (int i = 0; i < gridDimention; i++) localUnew[i * gridDimention + 0] = leftRecvBuf[i]; 
      }
      if (!rightBoundary) {
        MPI_Recv(rightRecvBuf, gridDimention, MPI_DOUBLE, mpirank+1, 1, MPI_COMM_WORLD, &status2);
        for (int i = 0; i < gridDimention; i++) localUnew[i * gridDimention + (gridDimention-1)] = rightRecvBuf[i]; 
      }
      if (!upBoundary) {
        MPI_Recv(&upRecvBuf, gridDimention, MPI_DOUBLE, mpirank-nodeDimention, 4, MPI_COMM_WORLD, &status3);
        for (int i = 0; i < gridDimention; i++) localUnew[i] = upRecvBuf[i]; 
      }
      if (!downBoundary) {
        MPI_Recv(&downRecvBuf, gridDimention, MPI_DOUBLE, mpirank+nodeDimention, 3, MPI_COMM_WORLD, &status4);
        for (int i = 0; i < gridDimention; i++) localUnew[gridDimention * (gridDimention-1) + i] = upRecvBuf[i]; 
      }

    // if (iter % 100 == 0) {
    //   gres = compute_residual(localU, gridDimention, invhsq);
    //   if (mpirank == 0) {
    //     printf("Iter %d: Residual: %g\n", iter, gres);
    //   }
    // } 

    double *localUtemp = localU; 
    localU = localUnew; 
    localUnew = localUtemp; 
  }

  /* Clean up */
  free(localU);
  free(localUnew);
}


int main(int argc, char * argv[]){
  int mpirank, p, N, lN, iter, max_iters;

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

  N = N * N; 

  /* compute number of unknowns handled by each process */
  lN = N / p;
  if ((N % p != 0) && mpirank == 0 ) {
    printf("N: %d, local N: %d\n", N, lN);
    printf("Exiting. N must be a multiple of p\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }

  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-5;

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  // /* initial residual */
  // gres0 = compute_residual(lu, lN, invhsq);
  // gres = gres0;


  computeAllU(max_iters, lN, h); 

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;

  if (0 == mpirank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();


  return 0;
}
