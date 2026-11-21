#pragma once 
#include <Eigen\Dense>
#include <vector>

struct Edge {
    int to;
    double weight;
};

double euclideanDist(const std::vector<double>& a, const std::vector<double>& b);
std::vector<std::vector<double>> matrixToVector2D(const Eigen::MatrixXd& mat);

std::vector<std::vector<double>> distanceMatrix(const std::vector<std::vector<double>> &data);
std::vector<std::vector<std::pair<double, int>>> alignedDistanceList(const std::vector<std::vector<double>> &distanceMatrix, const std::vector<std::vector<int>> &knnmatrix);
std::vector<std::vector<Edge>> adjacencyList(const std::vector<std::vector<std::pair<double, int>>> &list);

int argmin(const std::vector<std::pair<double, int>>& vec);
std::vector<int> Medoids(const std::vector<std::vector<double>> &data, const std::vector<std::vector<int>> &clusterIndices, const Eigen::MatrixXd &Psi, bool geodesic);

struct BridgeEdge {
    int u, v;
    double dist;
};
std::vector<std::vector<int>> connectedComponents(const std::vector<std::vector<Edge>>& adj);
BridgeEdge findClosestBridge( //returns an object
        const std::vector<int>& A,
        const std::vector<int>& B,
        const std::vector<std::vector<double>>& distanceMatrix);
void connectComponents(
        std::vector<std::vector<Edge>>& adj,
        const std::vector<std::vector<double>>& distanceMatrix);

int kthSmallestIndex(const std::vector<int> &v, int k);
int kthSmallestIndexPairs(const std::vector<std::pair<double, int>>& v, int k);

// kmeansM stuff
Eigen::VectorXd toEigenVector(const std::vector<double>& vec);
Eigen::MatrixXd getInvCov(const Eigen::MatrixXd &covariance);
bool choleskyDecompLLT(const Eigen::MatrixXd &cov);
std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> CovarianceMatrixGPT(
    const std::vector<std::vector<std::vector<double>>>& clusters,
    const std::vector<std::vector<double>>& centroids,
    const std::vector<std::vector<double>>& data);
Eigen::MatrixXd computeRobustCovarianceMAD(const std::vector<std::vector<double>>& data);
double computeMAD(const std::vector<double>& values);
double mahalanobisDistanceSingle(const Eigen::VectorXd& point, const Eigen::VectorXd& centroid, const Eigen::MatrixXd& covariance);
double mahalanobisDistanceSingleFast(const Eigen::VectorXd& point, const Eigen::VectorXd& centroid, const Eigen::MatrixXd& covariance);
bool allClustersHaveEnoughPoints(
    const std::vector<std::vector<std::vector<double>>>& clusters,
    size_t featureDim,
    size_t safetyFactor
);
double adaptiveAlpha(int clusterSize);
double distanceBlend(int iter, bool trigger, int startIter, int rampLength);
double avgClusterSize(const std::vector<std::vector<std::vector<double>>>& clusters);
