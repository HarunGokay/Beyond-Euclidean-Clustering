#include <iostream>
#include "utils.h"
#include <vector>
#include <limits>
#include <queue>
#include <algorithm>
#include <execution>
#include <ctime>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Spectra/DavidsonSymEigsSolver.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/DenseSymMatProd.h>

int kthSmallestIndex(const std::vector<double>& v, int k);

std::tuple<Eigen::MatrixXd, Eigen::VectorXd> EigenDecompSpectra(const Eigen::SparseMatrix<double>& L, int m);

Eigen::SparseMatrix<double> getGraphLaplacianHeatKernelSparse(
    const std::vector<std::vector<std::pair<double, int>>> &KNNalignedDistanceList, //aka: adjacency list
    const std::vector<std::vector<int>> &knnList,
    const std::vector<std::vector<double>> &data,
    const std::vector<double> &sigmas);

Eigen::SparseMatrix<double> getNormalizedLaplacianSparse(
    const Eigen::SparseMatrix<double>& L,
    const Eigen::VectorXd& degree);

    
Eigen::VectorXd getCoeff(const Eigen::VectorXd &eigenvalues, double t);

Eigen::MatrixXd scaleEigenvectors(const Eigen::MatrixXd &eigenvectors, const Eigen::VectorXd& coeff);

double diffusion_distance(int i, int j, const Eigen::MatrixXd& Psi);