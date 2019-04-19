#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int N = 100;
int iteration_count = 20000;
double factor = 1000000.0;

double *current_U;
double *previous_U;
double *F;

double **A;
double h;
double initial_residual;

double compute_residual() {
    double sum = 0.0;

    for (int i = 0; i < N; i++) {
        double total = 0.0;
        for (int j = 0; j < N; j++) {
            total += A[i][j] * current_U[j];
        }

        sum += (total - F[i]) * (total - F[i]);
    }

    return sqrt(sum);
}

void compute_single_Jacobi(int i) {
    double sum = 0.0;

    for (int j = 0; j < N; j++) {
        if (j == i) continue; 
        sum += A[i][j] * previous_U[j];
    }

    current_U[i] = (F[i] - sum) / A[i][i];
}

void compute_single_GS(int i) {
    double sum = 0.0;

    for (int j = 0; j < N;j++) {
        if (j < i) {
            sum += A[i][j] * current_U[j];
        } else if (j > i) {
            sum += A[i][j] * previous_U[j];
        }
    }

    current_U[i] = (F[i] - sum) / A[i][i];
}

void compute() {
    for (int iteration = 0; iteration < iteration_count; iteration++) {
        for (int i = 0; i < N; i++) {
            compute_single_Jacobi(i);
            // compute_single_GS(i); 
        }
        double residual = compute_residual();
        printf("Iteration: %d \n", iteration); 
        printf("New residual is: %f \n", residual);
        printf("Factor: %f \n", initial_residual / residual); 
        if (residual < initial_residual / factor) {
            printf("Termination program with iteration: %s \n", iteration);
            return;
        } else {
            previous_U = current_U;
        }
    }
    printf("Terminating......");
}

void initialize() {
    // Initialize F
    F = (double *) malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        F[i] = 1.0;
    }

    // Initialize U
    current_U = (double *) malloc(N * sizeof(double));
    previous_U = (double *) malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        current_U[i] = 0.0;
        previous_U[i] = 0.0;
    }

    // Initialize A
    A = (double **) malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        A[i] = (double *) malloc(N * sizeof(double));
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = 1 / (h * h);
            if (i == j) {
                A[i][j] *= 2;
            } else if (i + 1 == j || j + 1 == i) {
                A[i][j] *= -1;
            } else {
                A[i][j] = 0;
            }
        }
    }
}

int main() {
    h = 1.0 / (N + 1);
    clock_t t;

    initialize();
    initial_residual = compute_residual();
    printf("Initial residual: %f \n", initial_residual);

    t = clock();
    compute();
    free(current_U); 
    free(previous_U); 
    free(A); 
    free(F); 
    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds


    printf("Total time in seconds: %f \n", time_taken);
    return 0;
}