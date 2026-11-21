#include <iostream>
#include <vector>
#include <string>
#include "utils.h"
#include "kmeans.h"
#include "hdbscan.h"
#include "heatkernel.h"
#include "preservetopo.h"
#include "datagen.h"

// usage: diffusion_clustering [dataset] [method] [k] [eps] [minpts] [t]
//
// dataset: clusters | torus | swissroll | mobius
// method:  kmeansdiff | dbscandiff | kmeansM | all
//
// eps and t are tuned per dataset (see the table below), anything you pass overrides them.
// kmeansM is kmeans with a mahalanobis distance and runs on the raw data.

struct Params
{
    int k;
    double eps;
    int minpts;
    double t;
};

void printClusterSizes(const std::vector<std::vector<int>> &clusterIndices)
{
    for (int c = 0; c < clusterIndices.size(); c++)
        std::cout << "cluster " << c << ": " << clusterIndices[c].size() << " points" << std::endl;
}

Eigen::MatrixXd getPsi(const std::vector<std::vector<double>> &data, double t)
{
    // graph
    auto distMatrix = distanceMatrix(data);
    auto [KNNList, SIGMAS] = symmetric_knn(data, 15);
    auto AdistList = alignedDistanceList(distMatrix, KNNList);
    auto AdjListGraph = adjacencyList(AdistList);
    connectComponents(AdjListGraph, distMatrix);

    // spectral decomp
    auto matrixL = getGraphLaplacianHeatKernelSparse(AdistList, KNNList, data, SIGMAS);

    int m_aronoldi = 100; // how many eigenvectors to keep
    auto [eigenvectors, eigenvalues] = EigenDecompSpectra(matrixL, m_aronoldi);

    int m = eigenvalues.size();
    int n = eigenvectors.rows();

    // find index of smallest absolute eigenvalue
    int minIndex;
    eigenvalues.cwiseAbs().minCoeff(&minIndex); // minIndex is where near-zero lives

    // Build trimmed evals and evecs excluding minIndex
    Eigen::VectorXd evals_trimmed(m - 1);
    Eigen::MatrixXd evecs_trimmed(n, m - 1);

    int outcol = 0;
    for (int col = 0; col < m; ++col)
    {
        if (col == minIndex)
            continue;
        evals_trimmed[outcol] = eigenvalues[col];
        evecs_trimmed.col(outcol) = eigenvectors.col(col);
        ++outcol;
    }

    Eigen::VectorXd coeffs = getCoeff(evals_trimmed, t);
    return scaleEigenvectors(evecs_trimmed, coeffs);
}

int main(int argc, char **argv)
{
    std::string dataset = argc > 1 ? argv[1] : "clusters";
    std::string method = argc > 2 ? argv[2] : "all";

    std::vector<std::vector<double>> data;
    Params p;

    if (dataset == "torus")
    {
        data = generateTorus(1000, 3.0, 1.0, 0.05);
        p = {1, 0.1, 10, 5.0};
    }
    else if (dataset == "swissroll")
    {
        data = generateSwissRoll(1000, 0.1);
        p = {1, 0.1, 10, 5.0};
    }
    else if (dataset == "mobius")
    {
        generateInvertedMobiusStrip(data, 1000, 3);
        p = {1, 0.1, 10, 5.0};
    }
    else
    {
        data = generateClusters(1000, 5, 15, 0.7);
        p = {5, 0.01, 10, 20.0};
    }

    if (argc > 3) p.k = std::stoi(argv[3]);
    if (argc > 4) p.eps = std::stod(argv[4]);
    if (argc > 5) p.minpts = std::stoi(argv[5]);
    if (argc > 6) p.t = std::stod(argv[6]);

    bool all = method == "all";

    if (method == "kmeansdiff" || method == "dbscandiff" || all)
    {
        Eigen::MatrixXd Psi = getPsi(data, p.t);

        if (method == "kmeansdiff" || all)
        {
            // diffusion kmeans, plain kmeans on the rows of Psi
            auto [clusterskm, centroidskm, silhouette] = kmeans(matrixToVector2D(Psi), p.k, 200);
            std::cout << "diffusion kmeans silhouette: " << silhouette << std::endl;
            for (int c = 0; c < clusterskm.size(); c++)
                std::cout << "cluster " << c << ": " << clusterskm[c].size() << " points" << std::endl;
        }

        if (method == "dbscandiff" || all)
        {
            // diffusion dbscan
            HDBSCAN dbscan(p.eps, p.minpts);
            auto [coreIndices, binaryOverview] = dbscan.getCorePoints(data, Psi, true);
            auto [clustersdbscan, clusterIndices] = dbscan.getClusters(data, coreIndices, binaryOverview, Psi, true);
            double dbscan_score = dbscan.dbcvScore(data, clusterIndices, Psi, true);

            int clustered = 0;
            for (const auto &c : clusterIndices)
                clustered += c.size();

            std::cout << "diffusion dbscan clusters: " << clusterIndices.size() << std::endl;
            printClusterSizes(clusterIndices);
            std::cout << "diffusion dbscan noise points: " << data.size() - clustered << std::endl;
            std::cout << "diffusion dbscan dbcv: " << dbscan_score << std::endl;
        }
    }

    if (method == "kmeansM" || all)
    {
        auto [clustersM, centroidsM, silhouetteM] = kmeansM(data, p.k, 200);
        std::cout << "kmeansM (mahalanobis) silhouette: " << silhouetteM << std::endl;
        for (int c = 0; c < clustersM.size(); c++)
            std::cout << "cluster " << c << ": " << clustersM[c].size() << " points" << std::endl;
    }

    return 0;
}
