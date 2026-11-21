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
#include <random>
#include <numeric>
#include <stdexcept>
#include <string>

std::vector<std::vector<double>> matrixToVector2D(const Eigen::MatrixXd &mat)
{
    std::vector<std::vector<double>> result(mat.rows(), std::vector<double>(mat.cols()));

    for (int i = 0; i < mat.rows(); ++i)
    {
        for (int j = 0; j < mat.cols(); ++j)
        {
            result[i][j] = mat(i, j);
        }
    }

    return result;
}

double euclideanDist(const std::vector<double> &a, const std::vector<double> &b)
{

    // takes in vector
    if (a.size() != b.size())
    {
        std::cerr << "[euclideanDist] Dimension mismatch: a.size() = " << a.size()
                  << ", b.size() = " << b.size() << std::endl;
        exit(1); // Or th3row exception
    }

    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }

    // sums up the squared differences of each single dimension
    // measures how far theyre apart in high dimensional space
    return sqrt(sum);
}

std::vector<int> Medoids(const std::vector<std::vector<double>> &data,
                         const std::vector<std::vector<int>> &clusterIndices,
                         const Eigen::MatrixXd &Psi,
                         bool geodesic)
{
    if (data.size() != Psi.rows())
    {
        throw std::invalid_argument("[Medoid] ClusterIndices and Geodesics don't align!");
    }

    std::cout << "Data size: " << data.size() << ", Geodesics size: " << Psi.rows() << std::endl;

    std::vector<int> medoids;

    for (int i = 0; i < clusterIndices.size(); i++)
    { // loop over all clusters

        if (clusterIndices[i].empty())
        {
            throw std::invalid_argument("[Medoid] Cluster " + std::to_string(i) + " is empty!");
        }

        std::cout << "[Medoid] Processing cluster " << i << " with " << clusterIndices[i].size() << " points." << std::endl;

        std::vector<std::pair<double, int>> distsums;

        for (int j = 0; j < clusterIndices[i].size(); j++)
        { // loop over the point ids in a cluster [i]
            double distsum = 0.0;
            int seed = clusterIndices[i][j];

            // Check if seed is a valid index for data and geodesics
            if (seed >= data.size())
            {
                throw std::out_of_range("[Medoid] Invalid index in clusterIndices for data: " + std::to_string(seed));
            }
            if (seed >= Psi.rows())
            {
                throw std::out_of_range("[Medoid] Invalid index in clusterIndices for geodesics: " + std::to_string(seed));
            }

            for (const auto &pid : clusterIndices[i])
            {
                if (pid >= data.size())
                {
                    throw std::out_of_range("[Medoid] Invalid index in clusterIndices for data: " + std::to_string(pid));
                }
                if (pid >= Psi.rows())
                {
                    throw std::out_of_range("[Medoid] Invalid index in clusterIndices for geodesics: " + std::to_string(pid));
                }

                if (seed != pid)
                {
                    if (geodesic)
                    {
                        double geodesicDist = diffusion_distance(seed, pid, Psi);
                        distsum += geodesicDist;
                    }
                    else
                    {
                        double eucDist = euclideanDist(data[seed], data[pid]);
                        distsum += eucDist;
                    }
                }
            }

            distsums.emplace_back(distsum, j); // oh wait the seed can be out of bounds, so i need to push back j. thats the location in the cluster
        }

        // Debug: Print the size of distsums and minval_index
        std::cout << "[Medoid] distsums size: " << distsums.size() << std::endl;

        // Ensure argmin is within bounds

        // int minval_index = argmin(distsums); !!
        int minval_index = kthSmallestIndexPairs(distsums, 0); // 0 is smallest

        // Debug: Check the value of minval_index before accessing distsums
        std::cout << "[Medoid] argmin index: " << minval_index << ", distsums size: " << distsums.size() << std::endl;

        if (minval_index >= distsums.size())
        {
            throw std::out_of_range("[Medoid] argmin index out of bounds: " + std::to_string(minval_index));
        }

        int local_index = distsums[minval_index].second; // .second is the index in the cluster
        int medoidID = clusterIndices[i][local_index];   // NOW this is a valid global index

        // Debug: Print the value of medoidID and the size of clusterIndices[i]
        std::cout << "[Medoid] medoidID: " << medoidID << ", clusterIndices[" << i << "] size: " << clusterIndices[i].size() << std::endl;

        // This is the line where the crash happens:
        medoids.push_back(medoidID); // something is going wrong here
    }

    return medoids;
    // Mentor: You confused the cluster-local index with the global data index!! thats what i thought actually
    // Mentor: You were feeding global indices into a place that expects cluster-local indices. Classic medoid-implementation faceplant.
}

std::vector<std::vector<std::pair<double, int>>>
alignedDistanceList(
    const std::vector<std::vector<double>> &distanceMatrix,
    const std::vector<std::vector<int>> &knnList)
{
    std::cout << "knnlist size: " << knnList.size() << std::endl;
    if (distanceMatrix.size() != knnList.size())
        throw std::invalid_argument("distanceMatrix and knnList size mismatch.");

    int n = distanceMatrix.size();
    std::vector<std::vector<std::pair<double, int>>> list;
    list.reserve(n);

    for (int i = 0; i < n; i++)
    {
        std::vector<std::pair<double, int>> alignedRow;
        alignedRow.reserve(knnList[i].size());

        for (int j = 0; j < knnList[i].size(); j++)
        {
            int idx = knnList[i][j];
            if (idx < 0 || idx >= n)
                throw std::out_of_range("Invalid neighbor index.");

            double dist = distanceMatrix[i][idx];
            alignedRow.emplace_back(dist, idx);
        }

        list.push_back(std::move(alignedRow));
    }

    return list;
}

std::vector<std::vector<double>> distanceMatrix(const std::vector<std::vector<double>> &data)
{
    std::cout << "data size: " << data.size() << std::endl;
    // std::cout << "Entered: [distanceMatrix]\n";
    //  why does this need to be euclidian, because theyre local. and thats fine. we need local data to connect together to get global distances (geodesic)
    std::vector<std::vector<double>> matrix;
    matrix.reserve(data.size());

    for (const auto &vec : data)
    {
        std::vector<double> localbuffer;
        for (const auto &othervec : data)
        {
            double dist = euclideanDist(vec, othervec);
            localbuffer.push_back(dist);
        }
        matrix.push_back(localbuffer);
    }
    // std::cout << "Exited: [distanceMatrix]\n";
    return matrix;
}

std::vector<std::vector<Edge>> adjacencyList(const std::vector<std::vector<std::pair<double, int>>> &alignedDistList)
{

    std::cout << "alignedDistList size: " << alignedDistList.size() << std::endl;

    int N = alignedDistList.size();
    std::vector<std::vector<Edge>> graph(N);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < alignedDistList[i].size(); j++)
        {

            double w = alignedDistList[i][j].first;
            int idx = alignedDistList[i][j].second;

            if (idx < 0 || idx >= N)
            {
                throw std::runtime_error("Invalid neighbor index.");
            }

            graph[i].push_back({idx, w});
        }
    }

    return graph;
}


std::vector<std::vector<int>> connectedComponents(const std::vector<std::vector<Edge>> &adj)
{
    int N = adj.size();
    std::vector<int> visited(N, 0);
    std::vector<std::vector<int>> comps;

    std::vector<int> stack;
    stack.reserve(N);

    for (int start = 0; start < N; start++)
    {
        if (visited[start])
            continue;

        // New component
        comps.emplace_back();
        auto &comp = comps.back();

        // DFS
        stack.push_back(start);
        visited[start] = 1;

        while (!stack.empty())
        {
            int u = stack.back();
            stack.pop_back();
            comp.push_back(u);

            for (auto &e : adj[u])
            {
                int v = e.to;
                if (!visited[v])
                {
                    visited[v] = 1;
                    stack.push_back(v);
                }
            }
        }
    }

    return comps;
}

BridgeEdge findClosestBridge( // returns an object
    const std::vector<int> &A,
    const std::vector<int> &B,
    const std::vector<std::vector<double>> &distanceMatrix)
{
    BridgeEdge best{-1, -1, std::numeric_limits<double>::infinity()};

    for (int i : A)
    {
        for (int j : B)
        {
            double d = distanceMatrix[i][j];
            if (d < best.dist)
            {
                best = {i, j, d};
            }
        }
    }

    return best;
}

void connectComponents(
    std::vector<std::vector<Edge>> &adj,
    const std::vector<std::vector<double>> &distanceMatrix)
{
    auto comps = connectedComponents(adj);
    int C = comps.size();

    if (C <= 1)
        return; // already fully connected

    std::cout << "Before connecting, number of components: " << comps.size() << std::endl;

    // Connect components linearly in order of closest possible edges
    for (int i = 0; i < C - 1; i++)
    {
        BridgeEdge b = findClosestBridge(comps[i], comps[i + 1], distanceMatrix);

        if (b.u == -1)
            throw std::runtime_error("Failed to find bridging edge? Something is wrong.");

        // Add bidirectional edge
        adj[b.u].push_back({b.v, b.dist});
        adj[b.v].push_back({b.u, b.dist});
    }

    auto comps_after = connectedComponents(adj);
    if (comps_after.size() > 1)
    {
        std::cerr << "Warning: Graph still has disconnected components!" << std::endl;
    }
    else
    {
        std::cout << "Graph is fully connected!" << std::endl;
    }
}

int kthSmallestIndex(const std::vector<int> &v, int k)
{
    if (k < 0 || k >= static_cast<int>(v.size()))
        throw std::out_of_range("k out of range");

    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                     [&](int a, int b)
                     { return v[a] < v[b]; });

    // but if.. ah wait. kth smallest != index. the indexes only represent locations, and location (NODE ID) has nothing to do if its "smaller".
    // i just have to switch to, .first, right

    return idx[k];
}

int kthSmallestIndexDouble(const std::vector<double> &v, int k)
{
    if (k < 0 || k >= static_cast<int>(v.size()))
        throw std::out_of_range("k out of range");

    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                     [&](int a, int b)
                     { return v[a] < v[b]; });

    // but if.. ah wait. kth smallest != index. the indexes only represent locations, and location (NODE ID) has nothing to do if its "smaller".
    // i just have to switch to, .first, right

    return idx[k];
}

int kthSmallestIndexPairs(const std::vector<std::pair<double, int>> &v, int k)
{
    if (k < 0 || k >= static_cast<int>(v.size()))
        throw std::out_of_range("k out of range");

    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::nth_element(
        idx.begin(),
        idx.begin() + k,
        idx.end(),
        [&](int a, int b)
        {
            const double va = v[a].first;
            const double vb = v[b].first;

            // Handle NaNs: push them to the end
            if (std::isnan(va))
                std::cout << "NAN detected [kthSmallestIndexPairs]";
            return false;
            if (std::isnan(vb))
                std::cout << "NAN detected [kthSmallestIndexPairs]";
            return true;

            // Compare by .first, stable tie-break by index
            return va < vb || (va == vb && a < b);
        });

    return idx[k]; // index of the k-th smallest by .first
}

double avgClusterSize(const std::vector<std::vector<std::vector<double>>> &clusters)
{
    double sum = 0;
    for (auto cluster : clusters)
    {
        sum = sum + cluster.size();
    }
    return sum / clusters.size();
}

double distanceBlend(int iter, bool trigger, int startIter, int rampLength)
{

    if (!trigger)
        return 0.0; // fully Euclidean
    if (iter >= startIter + rampLength)
        return 1.0;                               // fully Mahalanobis after past this amount of iterations
    return double(iter - startIter) / rampLength; // returns the actual blend value
}

double adaptiveAlpha(int clusterSize)
{

    int minSize = 10;
    int maxSize = 500;
    double alphaMin = 0.05;
    double alphaMax = 0.3;
    // Clamp cluster size
    clusterSize = std::max(minSize, std::min(clusterSize, maxSize));
    // Linear mapping clusterSize -> alpha in [alphaMax, alphaMin]
    double alpha = alphaMax - (clusterSize - minSize) * (alphaMax - alphaMin) / (maxSize - minSize);
    return alpha;
}

bool allClustersHaveEnoughPoints(
    const std::vector<std::vector<std::vector<double>>> &clusters,
    size_t featureDim,
    size_t safetyFactor)
{
    if (clusters.empty())
        return false;

    for (const auto &cluster : clusters)
    {
        if (cluster.size() < safetyFactor * featureDim)
        {
            return false;
        }
    }
    return true;
}

Eigen::VectorXd toEigenVector(const std::vector<double> &vec)
{
    Eigen::VectorXd eigenVec(vec.size());
    for (size_t i = 0; i < vec.size(); ++i)
    {
        eigenVec(i) = vec[i];
    }
    return eigenVec;
}

bool choleskyDecompLLT(const Eigen::MatrixXd &cov)
{

    if (cov.rows() != cov.cols() || cov.rows() == 0)
        return false;

    Eigen::MatrixXd sym = 0.5 * (cov + cov.transpose());
    Eigen::LLT<Eigen::MatrixXd> llt(sym);
    return llt.info() == Eigen::Success; // better than numerical issue, checks more
}

Eigen::MatrixXd getInvCov(const Eigen::MatrixXd &data)
{

    Eigen::MatrixXd cov = data;

    cov = 0.5 * (cov + cov.transpose()); // exacty symmetric

    // Cholesky decomposition
    Eigen::LLT<Eigen::MatrixXd> llt(cov);
    Eigen::MatrixXd cov_inv;

    // Check if the matrix is positive definite
    if (llt.info() == Eigen::NumericalIssue)
    {
        std::cerr << "[getInvCov] Matrix is not positive definite! Using Pseudo-Inverse!" << std::endl;
        cov += Eigen::MatrixXd::Identity(cov.rows(), cov.cols()) * 1e-3;
        cov_inv = cov.inverse(); // slower but safer fallback
    }
    else
    {
        // Compute the inverse using the Cholesky factor
        cov_inv = llt.solve(Eigen::MatrixXd::Identity(cov.rows(), cov.cols()));
    }

    return cov_inv;
}

std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> CovarianceMatrixGPT(
    const std::vector<std::vector<std::vector<double>>> &clusters,
    const std::vector<std::vector<double>> &centroids,
    const std::vector<std::vector<double>> &data)
{
    int d = centroids[0].size();
    std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> results;

    // Compute global covariance once for shrinkage
    Eigen::MatrixXd global_cov = computeRobustCovarianceMAD(data); // user-defined
    double global_trace = global_cov.trace();

    for (size_t z = 0; z < clusters.size(); ++z)
    {
        int n = static_cast<int>(clusters[z].size());

        // Small clusters -> fallback
        if (n <= 1)
        {
            results.push_back(Eigen::MatrixXd::Identity(d, d) * 1e-4);
            continue;
        }

        // Compute centered data matrix for cluster (d x n)
        Eigen::MatrixXd centered(d, n);
        for (int i = 0; i < n; ++i)
        {
            centered.col(i) = Eigen::Map<const Eigen::VectorXd>(clusters[z][i].data(), d) - Eigen::Map<const Eigen::VectorXd>(centroids[z].data(), d);
        }

        // Unbiased covariance
        Eigen::MatrixXd cov = (centered * centered.transpose()) / (n - 1);

        // Regularization: small diagonal scaled by average variance
        double diag_eps = std::max(1e-6, 1e-4 * cov.trace() / d); // so its not too weak
        cov += Eigen::MatrixXd::Identity(d, d) * diag_eps;

        // Adaptive shrinkage toward global covariance
        double lambda = 1e-4; //+ 1e-2 * (1.0 / std::sqrt(n)); // smaller clusters get more shrinkage
        cov = (1.0 - lambda) * cov + lambda * global_cov;

        // Normalize average variance ~1 for numerical stability
        // cov *= d / cov.trace(); might kill variance

        // Eigenvalue check for numerical issues
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(cov);
        Eigen::VectorXd evals = es.eigenvalues();
        double min_eval = evals.minCoeff();

        double eps_threshold = 1e-8;                                // lower threshhold
        double adjust_eps = std::max(1e-6, 1e-4 * cov.trace() / d); // adaptive guard

        if (min_eval <= eps_threshold)
        {
            cov += Eigen::MatrixXd::Identity(d, d) * adjust_eps;
        }

        results.push_back(cov);
    }

    return results;
}

double mahalanobisDistanceSingle(const Eigen::VectorXd &point, const Eigen::VectorXd &centroid, const Eigen::MatrixXd &covariance)
{

    // Eigen::Matrix<double, 14, 14> cov_reg = covariance;
    // cov_reg += 1e-3 * Eigen::Matrix<double, 14, 14>::Identity();   //pretty sure we dont need this since we already did shrinking in the covariancematrix function

    // Defensive copy to avoid modifying the original matrix
    Eigen::MatrixXd cov = covariance;

    cov = 0.5 * (cov + cov.transpose()); // exacty symmetric

    // Cholesky decomposition
    Eigen::LLT<Eigen::MatrixXd> llt(covariance);
    Eigen::MatrixXd cov_inv;

    // Check if the matrix is positive definite
    if (llt.info() == Eigen::NumericalIssue)
    {
        std::cerr << "[MdistSingle]Matrix is not positive definite! Using Pseudo-Inverse!" << std::endl;
        cov += Eigen::MatrixXd::Identity(cov.rows(), cov.cols()) * 1e-3;
        cov_inv = cov.inverse(); // slower but safer fallback
    }
    else
    {
        // Compute the inverse using the Cholesky factor
        cov_inv = llt.solve(Eigen::MatrixXd::Identity(covariance.rows(), covariance.cols())); // redeclaration fucked me up
    }

    // Calculate the difference between the point and the mean
    Eigen::VectorXd diff = point - centroid;

    if (diff.size() != cov_inv.rows())
    {
        std::cerr << "[MdistSingle] Dimension mismatch in Mahalanobis: diff=" << diff.size()
                  << " cov=" << cov_inv.rows() << std::endl;
    }

    // Compute Mahalanobis distance
    double distance = std::sqrt(diff.transpose() * cov_inv * diff); // someone said to do this, idk why lol

    if (distance < 0)
        distance = 0.0;

    return distance;
}

std::vector<double> MahalanobisDistMulti(
    const std::vector<double> &point,
    const std::vector<std::vector<double>> &meanvecs,
    const std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> &covmatrices)
{

    // Convert input point into Eigen
    Eigen::VectorXd x = toEigenVector(point);

    std::vector<double> distances;
    distances.reserve(meanvecs.size());

    for (size_t c = 0; c < meanvecs.size(); ++c)
    {
        // Convert mean

        Eigen::VectorXd mu = toEigenVector(meanvecs[c]);

        // Regularize covariance to avoid singularities
        Eigen::MatrixXd cov_reg = covmatrices[c];

        // cov_reg += 1e-3 * Eigen::Matrix<double, 14, 14>::Identity();
        cov_reg = 0.5 * (cov_reg + cov_reg.transpose());

        // Inverse covariance
        Eigen::MatrixXd cov_inv;
        // Cholesky decomposition
        Eigen::LLT<Eigen::MatrixXd> llt(cov_reg);

        // Check if the matrix is positive definite
        if (llt.info() == Eigen::NumericalIssue)
        {
            std::cerr << "[MdistMulti] Matrix is not positive definite!" << std::endl;
        }
        else
        {
            // Compute the inverse using the Cholesky factor
            Eigen::MatrixXd cov_inv = llt.solve(Eigen::MatrixXd::Identity(cov_reg.rows(), cov_reg.cols()));
        }

        // Compute diff
        Eigen::VectorXd diff = x - mu;

        // Mahalanobis distance
        double dist_sq = (diff.transpose() * cov_inv * diff)(0, 0); // That ensures im not silently storing a matrix object with garbage or misaligned memory.
        distances.push_back(std::sqrt(dist_sq));
    }

    return distances;
}

double cosineSimilarity(const Eigen::VectorXd &A,
                        const Eigen::VectorXd &B)
{ // takes 2 vectors

    float epsilon = 1e-10f;

    // i think a is comparing it a point thats not there?

    double dot_product = A.dot(B); // A · B  //does A . B
    double norm_A = A.norm();      // ||A||
    double norm_B = B.norm();      // ||B||  //length vector from origin. "length" of a vector. // why is B 0

    if (norm_A < epsilon || norm_B < epsilon)
    {
        // if shits too small
        return 0.0;
    }

    // dividing by the norm allows you to focus on the direction of the vectors (the angle between them), rather than their magnitudes.

    return dot_product / (norm_A * norm_B);
}

double mahalanobisDistanceSingleFast(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const Eigen::MatrixXd &invCov)
{

    Eigen::VectorXd diff = x - y;
    double distsquared = std::sqrt((diff.transpose() * invCov * diff).value());
    double dist = std::sqrt(std::max(distsquared, 0.0));
    if (dist < 0)
        dist = 0;

    if (diff.size() != invCov.rows())
    {
        std::cerr << "[MdistSingleFast] Dimension mismatch: diff=" << diff.size()
                  << " cov=" << invCov.rows() << std::endl;
    }

    return dist;
}

Eigen::MatrixXd computeRobustCovarianceMAD(const std::vector<std::vector<double>> &data)
{
    size_t n = data.size();
    if (n < 2)
    {
        throw std::invalid_argument("[ComputeRCovMAD] At least two data points are required to compute covariance.");
    }

    size_t d = data[0].size();
    for (const auto &x : data)
    {
        if (x.size() != d)
        {
            throw std::invalid_argument("[ComputeRCovMAD] Inconsistent row size detected.");
        }
    }

    // Step 1: Compute the mean of each dimension
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(d);
    for (const auto &x : data)
    {
        for (size_t j = 0; j < d; ++j)
        {
            mean(j) += x[j];
        }
    }
    mean /= n;

    // Step 2: Compute the MAD for each dimension
    Eigen::MatrixXd madMatrix = Eigen::MatrixXd::Zero(d, d);

    // For each dimension, compute the MAD
    for (size_t j = 0; j < d; ++j)
    {
        std::vector<double> columnValues;
        for (const auto &x : data)
        {
            columnValues.push_back(x[j]);
        }

        double mad = computeMAD(columnValues);

        // The "robust covariance" is simply MAD applied to each dimension for now
        madMatrix(j, j) = mad * mad; // Using MAD^2 as a measure of spread for variance
    }

    return madMatrix;
}

double computeMAD(const std::vector<double> &values)
{
    std::vector<double> absDeviations(values.size());
    double median = 0;

    // Compute the median of the values
    std::vector<double> sortedValues = values;
    std::sort(sortedValues.begin(), sortedValues.end());

    size_t n = sortedValues.size();
    if (n % 2 == 0)
    {
        median = (sortedValues[n / 2 - 1] + sortedValues[n / 2]) / 2.0;
    }
    else
    {
        median = sortedValues[n / 2];
    }

    // Calculate the absolute deviations from the median
    for (size_t i = 0; i < values.size(); ++i)
    {
        absDeviations[i] = std::abs(values[i] - median);
    }

    // Compute the median of the absolute deviations
    std::sort(absDeviations.begin(), absDeviations.end());
    double mad = 0;
    if (n % 2 == 0)
    {
        mad = (absDeviations[n / 2 - 1] + absDeviations[n / 2]) / 2.0;
    }
    else
    {
        mad = absDeviations[n / 2];
    }

    return mad;
}

