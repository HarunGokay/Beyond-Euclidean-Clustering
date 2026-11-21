#include <iostream>
#include "utils.h"
#include <vector>
#include <limits>
#include <queue>
#include <algorithm>
#include <execution>
#include <ctime>
#include <cmath>
#include "heatkernel.h"

using namespace Spectra;

/*
int kthSmallestIndex(const std::vector<std::pair<double, int>> &v, int k)
{
    // Print input vector
    std::cout << "Input vector:\n";
    for (const auto &elem : v)
    {
        std::cout << "(" << elem.first << ", " << elem.second << ") ";
    }
    std::cout << std::endl;

    // Initialize index vector
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // Print initial indices
    std::cout << "Initial indices:\n";
    for (int i = 0; i < idx.size(); ++i)
    {
        std::cout << idx[i] << " ";
    }
    std::cout << std::endl;

    // Perform nth_element
    std::cout << "Performing nth_element to find the " << k << "-th smallest element..." << std::endl;
    std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                     [&](int a, int b)
                     {
                         std::cout << "Comparing v[" << a << "] = (" << v[a].first << ", " << v[a].second << ") ";
                         std::cout << "and v[" << b << "] = (" << v[b].first << ", " << v[b].second << ")\n";
                         return v[a].first < v[b].first;
                     });

    // Print the resulting order of indices
    std::cout << "Indices after nth_element:\n";
    for (int i = 0; i < idx.size(); ++i)
    {
        std::cout << idx[i] << " ";
    }
    std::cout << std::endl;

    // Return the k-th index
    std::cout << "Returning index: " << idx[k] << " with value: " << v[idx[k]].first << std::endl;

    return idx[k];
}
*/

std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
EigenDecompSpectra(const Eigen::SparseMatrix<double> &L, int m)
{
    // Operator: matrix-vector product
    SparseSymMatProd<double> op(L);

    // arnoldi size
    int ncv = std::min(2 * m, (int)L.rows());

    // Solver: smallest m eigenvalues
    SymEigsSolver<SparseSymMatProd<double>> eigs(op, m, ncv); // search space. 3 is return

    eigs.init();
    int nconv = eigs.compute(SortRule::SmallestAlge);

    if (eigs.info() != CompInfo::Successful)
    {
        throw std::runtime_error("Eigen decomposition failed in Spectra");
    }

    Eigen::VectorXd vals = eigs.eigenvalues().real();
    Eigen::MatrixXd vecs = eigs.eigenvectors().real(); // made it real

    return {vecs, vals};
}

/*

✔️ When you won’t see the constant eigenvector (e.g., Laplacian matrices)

If your matrix has a known null eigenvector (e.g., the graph Laplacian always has the constant vector 1 as its 0-eigenvector), you will only get that eigenvector if:

you ask Spectra to compute the smallest eigenvalues, and

the null eigenvalue is among those requested.

If you ask for:

Spectra::SymEigsSolver<...> eigs(op, k, m);
eigs.init();
eigs.compute(Spectra::LARGEST_MAGN);


then:

the zero eigenvalue will not be returned,

the constant eigenvector will not be returned,

but nothing has been “deleted” — it just isn’t among the requested eigenpairs.

*/

/*
std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
EigenDecompSpectra(const Eigen::SparseMatrix<double> &L, int m)
{
    // Print matrix size
    std::cout << "Matrix size: " << L.rows() << "x" << L.cols() << std::endl;

    // Operator: matrix-vector product
    SparseSymMatProd<double> op(L);

    // Arnoldi size
    int ncv = std::min(2 * m, (int)L.rows());
    std::cout << "Arnoldi size (ncv): " << ncv << std::endl;

    // Solver: smallest m eigenvalues
    SymEigsSolver<SparseSymMatProd<double>> eigs(op, m, ncv); // search space

    std::cout << "Initializing eigen solver..." << std::endl;
    eigs.init();

    // Compute eigenvalues
    std::cout << "Computing smallest " << m << " eigenvalues..." << std::endl;
    int nconv = eigs.compute(SortRule::SmallestAlge);  // 3 is return

    // Debug: Number of converged eigenvalues
    std::cout << "Number of converged eigenvalues: " << nconv << std::endl;

    // Check for success
    if (eigs.info() != CompInfo::Successful)
    {
        std::cerr << "Eigen decomposition failed" << std::endl;
        throw std::runtime_error("Eigen decomposition failed in Spectra");
    }

    // Debug: Eigenvalues and eigenvectors
    Eigen::VectorXd vals = eigs.eigenvalues();
    Eigen::MatrixXd vecs = eigs.eigenvectors();

    std::cout << "Eigenvalues: \n" << vals.transpose() << std::endl;  // Print eigenvalues
    std::cout << "Eigenvectors (columns): \n" << vecs << std::endl; // Print eigenvectors

    return {vecs, vals};
}
*/
Eigen::SparseMatrix<double> getGraphLaplacianHeatKernelSparse(
    const std::vector<std::vector<std::pair<double, int>>> &KNNalignedDistanceList, // aka: adjacency list
    const std::vector<std::vector<int>> &knnList,
    const std::vector<std::vector<double>> &data,
    const std::vector<double> &sigma)
{
    int N = data.size();

    // Dii = Σj Aij if adjacency list is binary, the degree too

    std::vector<Eigen::Triplet<double>> Wtriplets;
    Wtriplets.reserve(N * 10 * 2); // KNN symmetric

    std::vector<double> degree(N, 0.0);

    // Build sparse W and degree
    for (int i = 0; i < N; i++)
    {
        for (int t = 0; t < knnList[i].size(); t++)
        {
            int j = knnList[i][t];

            double dist = euclideanDist(data[i], data[j]);
            double w = exp(-(dist * dist) / (sigma[i] * sigma[j]));

            Wtriplets.emplace_back(i, j, w);
            Wtriplets.emplace_back(j, i, w);

            degree[i] += w;
            degree[j] += w;
        }
    }

    // Build sparse W
    Eigen::SparseMatrix<double> W(N, N);
    W.setFromTriplets(Wtriplets.begin(), Wtriplets.end());

    // Build sparse D
    std::vector<Eigen::Triplet<double>> Dtriplets;
    Dtriplets.reserve(N);
    for (int i = 0; i < N; i++)
        Dtriplets.emplace_back(i, i, degree[i]);

    Eigen::SparseMatrix<double> D(N, N);
    D.setFromTriplets(Dtriplets.begin(), Dtriplets.end());

    auto result = getNormalizedLaplacianSparse(D - W, Eigen::Map<Eigen::VectorXd>(degree.data(), degree.size()));

    // L = D - W
    return result; // diagonal = vector
}

/*
Eigen::SparseMatrix<double> getGraphLaplacianHeatKernelSparse(
    const std::vector<std::vector<std::pair<double, int>>> &KNNalignedDistanceList, // aka: adjacency list
    const std::vector<std::vector<int>> &knnList,
    const std::vector<std::vector<double>> &data)
{
    int N = data.size();
    std::cout << "Number of data points (N): " << N << std::endl;

    // Dii = Σj Aij if adjacency list is binary, the degree too
    std::vector<Eigen::Triplet<double>> Wtriplets;
    Wtriplets.reserve(N * 10 * 2); // KNN symmetric

    std::vector<double> degree(N, 0.0);

    std::vector<double> sigma(N);

    // Compute sigmas directed
    std::cout << "Computing sigmas for each point..." << std::endl;
    for (int i = 0; i < N; i++)
    {
        int idx = kthSmallestIndex(KNNalignedDistanceList[i], 5); // breaks if k > size of distances
        sigma[i] = std::max(KNNalignedDistanceList[i][idx].first, 1e-8); // not smaller than 1e-8
        std::cout << "Sigma[" << i << "] = " << sigma[i] << " (for point " << i << ")" << std::endl;
    }


    std::vector<double> sigma(N);
    // Compute sigmas directed
    std::cout << "Computing sigmas for each point..." << std::endl;
    for (int i = 0; i < N; i++)
    {
        sigma[i] = std::max(meanPAIR(KNNalignedDistanceList[i]), 1e-8); // not smaller than 1e-8
        std::cout << "Sigma[" << i << "] = " << sigma[i] << " (for point " << i << ")" << std::endl;
    }



    // Build sparse W and degree
    std::cout << "Building the sparse weight matrix W and degree vector..." << std::endl;
    for (int i = 0; i < N; i++)
    {
        for (int t = 0; t < knnList[i].size(); t++)
        {
            int j = knnList[i][t];

            // Euclidean distance between points i and j
            double dist = euclideanDist(data[i], data[j]);
            double w = exp(-(dist * dist) / (sigma[i] * sigma[j])); // Heat kernel

            std::cout << "Distance between " << i << " and " << j << ": " << dist << ", weight: " << w << std::endl;

            Wtriplets.emplace_back(i, j, w);
            Wtriplets.emplace_back(j, i, w);

            degree[i] += w;
            degree[j] += w;
        }
    }

    // Build sparse W matrix
    std::cout << "Building sparse matrix W..." << std::endl;
    Eigen::SparseMatrix<double> W(N, N);
    W.setFromTriplets(Wtriplets.begin(), Wtriplets.end());

    // Print sparse matrix W summary
    std::cout << "Sparse matrix W built with " << W.nonZeros() << " non-zero elements." << std::endl;

    // Build sparse D matrix (degree matrix)
    std::cout << "Building sparse degree matrix D..." << std::endl;
    std::vector<Eigen::Triplet<double>> Dtriplets;
    Dtriplets.reserve(N);
    for (int i = 0; i < N; i++)
    {
        Dtriplets.emplace_back(i, i, degree[i]);
    }

    Eigen::SparseMatrix<double> D(N, N);
    D.setFromTriplets(Dtriplets.begin(), Dtriplets.end());

    // Print sparse matrix D summary
    std::cout << "Sparse matrix D built with " << D.nonZeros() << " non-zero elements." << std::endl;

    // Compute the normalized Laplacian
    std::cout << "Computing the normalized Laplacian..." << std::endl;
    auto result = getNormalizedLaplacianSparse(D - W, Eigen::Map<Eigen::VectorXd>(degree.data(), degree.size()));

    std::cout << "Normalized Laplacian computed successfully." << std::endl;

    // Return the result (Laplacian matrix)
    return result;
}
*/

Eigen::SparseMatrix<double> getNormalizedLaplacianSparse(
    const Eigen::SparseMatrix<double> &L,
    const Eigen::VectorXd &degreeinput)
{
    Eigen::VectorXd degree = degreeinput;
    int N = L.rows();

    Eigen::VectorXd inv_sqrt = degree.array().inverse().sqrt();

    std::cout << "Checking degrees\n";
    for (int i = 0; i < N; i++)
    {
        if (degree[i] <= 1e-12 || !std::isfinite(degree[i]))
            std::cout << "Node " << i << " has bad degree: " << degree[i] << std::endl;

        if (degree[i] < 1e-12)
        {
            degree[i] = 1e-12;
        }
    }

    Eigen::SparseMatrix<double> Dinv(N, N);
    std::vector<Eigen::Triplet<double>> T;
    T.reserve(N);
    for (int i = 0; i < N; i++)
        T.emplace_back(i, i, inv_sqrt[i]);
    Dinv.setFromTriplets(T.begin(), T.end());

    return Dinv * L * Dinv;
}

/*
Eigen::SparseMatrix<double> getNormalizedLaplacianSparse(
    const Eigen::SparseMatrix<double> &L,
    const Eigen::VectorXd &degreeinput)
{
    Eigen::VectorXd degree = degreeinput;
    int N = L.rows();

    // Compute the inverse square root of degrees
    Eigen::VectorXd inv_sqrt = degree.array().inverse().sqrt();

    // Debug: Check degrees and detect invalid ones
    std::cout << "Checking degrees...\n";
    for (int i = 0; i < N; i++)
    {
        if (degree[i] <= 1e-12 || !std::isfinite(degree[i]))
        {
            std::cout << "Node " << i << " has bad degree: " << degree[i] << std::endl;
        }

        if (degree[i] < 1e-12)
        {
            degree[i] = 1e-12; // Prevent division by zero
            std::cout << "Node " << i << " degree set to 1e-12 to avoid division by zero.\n";
        }
    }

    // Debug: Show degree values after correction
    std::cout << "Degrees after correction:\n";
    for (int i = 0; i < N; i++)
    {
        std::cout << "Degree[" << i << "] = " << degree[i] << std::endl;
    }

    // Create sparse matrix Dinv (diagonal inverse square root of degree matrix)
    std::cout << "Building sparse Dinv matrix...\n";
    Eigen::SparseMatrix<double> Dinv(N, N);
    std::vector<Eigen::Triplet<double>> T;
    T.reserve(N);
    for (int i = 0; i < N; i++)
    {
        T.emplace_back(i, i, inv_sqrt[i]);
    }
    Dinv.setFromTriplets(T.begin(), T.end());

    // Debug: Print the first few elements of Dinv
    std::cout << "Dinv matrix (first few values):\n";
    for (int i = 0; i < std::min(10, N); i++) // Print first 10 diagonal values
    {
        std::cout << "Dinv[" << i << ", " << i << "] = " << Dinv.coeff(i, i) << std::endl;
    }

    // Calculate the normalized Laplacian L = Dinv * L * Dinv
    std::cout << "Computing normalized Laplacian...\n";
    Eigen::SparseMatrix<double> L_normalized = Dinv * L * Dinv;

    // Debug: Print the number of non-zero elements in the result
    std::cout << "Normalized Laplacian matrix has " << L_normalized.nonZeros() << " non-zero elements.\n";

    return L_normalized;
}
*/

Eigen::VectorXd getCoeff(const Eigen::VectorXd &eigenvalues_, double t)
{

    if (eigenvalues_.size() == 0)
        throw std::invalid_argument("[getCoeff] eigenvalues empty!");

    Eigen::VectorXd coeff(eigenvalues_.size());

    for (size_t i = 0; i < eigenvalues_.size(); i++)
    {
        double ck = exp(-eigenvalues_[i] * t);
        coeff[i] = ck;
    }

    return coeff;
}

Eigen::MatrixXd scaleEigenvectors(const Eigen::MatrixXd &eigenvectors, const Eigen::VectorXd &coeff)
{

    if (eigenvectors.rows() == 0 || eigenvectors.cols() == 0)
       throw std::invalid_argument("[scaleEigenvectors] eigenvectors empty!");
    if (coeff.size() == 0)
       throw std::invalid_argument("[scaleEigenvectors] coeff empty!");

    // remember: each column is eigenvector m

    Eigen::MatrixXd Psi = eigenvectors * coeff.asDiagonal();

    return Psi; 
}

double diffusion_distance(int i, int j, const Eigen::MatrixXd& Psi) {
    double sum = 0;
    for (int k = 0; k < Psi.cols(); k++) {
        double diff = Psi(i,k) - Psi(j,k)       ;
        sum += diff * diff;
    }
    return std::sqrt(sum);
}