#include <iostream>
#include <vector>
#include <math.h>
#include <ctime>
#include <stdio.h>
#include "utils.h"
#ifdef _OPENMP
# include <omp.h>
#endif

using namespace std;

double h;
int N = 100;

double computeDistance(std::vector<std::vector<double>> &matrixU,
                       std::vector<std::vector<double>> &matrixA,
                       std::vector<std::vector<double>> &matrixF) {
    double distance = 0.0;
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            double sum = 0.0;
            for (int k = 1; k <= N; k++) {
                sum += matrixA[i][k] * matrixU[k][j];
            }
            distance += (sum - matrixF[i][j]) * (sum - matrixF[i][j]);
        }
    }
    return sqrt(distance);
}

void computeGS(std::vector<std::vector<double>> &matrixU,
                   std::vector<std::vector<double>> &matrixF,
                   std::vector<std::vector<double>> &matrixA,
                   int N,
                   int iterationCount,
                   double h) {

    for (int ite = 0; ite < iterationCount; ite++) {
        std::vector<std::vector<double>> nextU(N + 2, std::vector<double>(N + 2, 0.0));
        for (int i = 1; i <= N; i++) {
            int j;

            #pragma opm parallel for
            for (j = i % 2 == 0 ? 2 : 1; j <= N; j += 2) {
                nextU[i][j] = 0.25 * (h * h * matrixF[i][j]
                                       + matrixU[i - 1][j] + matrixU[i][j - 1] + matrixU[i + 1][j] + matrixU[i][j + 1]);
            }

            #pragma omp parallel for
            for (j = i % 2 == 0 ? 1 : 2; j <= N; j += 2) {
                nextU[i][j] = 0.25 * (h * h * matrixF[i][j]
                                       + nextU[i - 1][j] + nextU[i][j - 1] + nextU[i + 1][j] + nextU[i][j + 1]);
            }
        }
        matrixU = nextU;

        if (ite % 1000 == 0) {
            double residual = computeDistance(matrixU, matrixA, matrixF);
            cout << "iteration: " << ite << " " << "residual: " << residual << endl;
        }
    }
}

int main() {
    cout << "Enter N: " << endl;
    cin >> N;

    h = 1.0 / (N + 1);
    std::vector<std::vector<double>> matrixU(N + 2, std::vector<double>(N + 2, 0.0));
    std::vector<std::vector<double>> matrixA(N + 2, std::vector<double>(N + 2, 0.0));
    std::vector<std::vector<double>> matrixF(N + 2, std::vector<double>(N + 2, 1.0));

    for (int i = 0; i < N + 2; i++) {
        for (int j = 0; j < N + 2; j++) {
            if (i == j) {
                matrixA[i][j] = 2.0;
            } else if (i == j - 1 || j == i - 1) {
                matrixA[i][j] = -1.0;
            }
            matrixA[i][j] *= 1 / (h * h);
        }
    }

    Timer t;
    t.tic();
    computeGS(matrixU, matrixF, matrixA, N, 5000, h);
    double time = t.toc();
#ifdef _OPENMP
    int nthreads = omp_get_num_threads();
        cout << "Total threads: " << nthreads << endl;
#endif
    cout << "N: " << N << endl;
    cout << "Duration: " << time << endl;

    return 0;
}
