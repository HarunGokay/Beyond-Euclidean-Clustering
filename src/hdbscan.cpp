#include <iostream>
#include "utils.h"
#include <vector>
#include <algorithm>
#include <execution>
#include <unordered_set>
#include "hdbscan.h"
#include "heatkernel.h"

double HDBSCAN::getRadius()
{
    return radius;
}

int HDBSCAN::getMin_points()
{
    return min_points;
}

std::tuple<std::vector<int>, std::vector<bool>> HDBSCAN::getCorePoints(
     const std::vector<std::vector<double>> &data,
     const Eigen::MatrixXd &Psi, 
     bool geodesic)
{
    //std::cout << "Entered: [getCorePoints]\n";
    int N = data.size();
    std::vector<bool> binaryOverview(N, false);
    std::vector<int> coreIndices;

    double radius = getRadius();
    int minPts = getMin_points();

    for (int i = 0; i < N; i++)
    {
        int count = 0;
        for (int j = 0; j < N; j++)
        {
            if (geodesic)
            {
                if (diffusion_distance(i, j, Psi) <= radius)
                {
                    count++;
                }
            }
            else
            {
                if (euclideanDist(data[i], data[j]) <= radius)
                {
                    count++;
                }
            }
        }

        if (count >= minPts)
        {
            binaryOverview[i] = true; // core
            coreIndices.push_back(i);
        }
    }
    //std::cout << "Exited: [getCorePoints]\n";
    return {coreIndices, binaryOverview};
}

std::tuple<std::vector<HDBSCAN::ClusterVectors>, std::vector<std::vector<int>>>
HDBSCAN::getClusters(
    const std::vector<std::vector<double>> &data,
    const std::vector<int> &coreIndices,
    const std::vector<bool> &binaryOverview,
    const Eigen::MatrixXd &Psi, 
    bool geodesic)
{
    //std::cout << "Entered: [getClusters]\n";
    double radius = getRadius();
    int N = data.size();

    std::vector<HDBSCAN::Clusters> Clusters;

    // cluster labels for all points
    // -1 = unassigned, -2 = noise
    std::vector<int> labels(N, -1);

    int C = coreIndices.size();
    std::vector<bool> coreVisited(C, false);

    // returns the neighbours of each point
    auto regionQuery = [&](int pid)
    {
        std::vector<int> neighbors;
        for (int i = 0; i < N; i++)
        {
            if (geodesic)
            {
                if (diffusion_distance(pid, i, Psi) <= radius)
                {
                    neighbors.push_back(i);
                }
            }
            else
            {
                if (euclideanDist(data[pid], data[i]) <= radius)
                {
                    neighbors.push_back(i);
                }
            }
        }
        return neighbors;
    };

    int clusterID = 0;

    for (int ci = 0; ci < C; ci++)
    {
        if (coreVisited[ci])
            continue;

        // start cluster
        HDBSCAN::Clusters cluster;
        std::vector<int> queue;
        queue.push_back(ci);
        coreVisited[ci] = true;

        // BFS over core points
        for (int q = 0; q < queue.size(); q++)
        {
            int coreIdx = coreIndices[queue[q]];
            labels[coreIdx] = clusterID;
            cluster.points.push_back(coreIdx); //push a core in, data[coreIdx] was original vector

            auto neighbors = regionQuery(coreIdx);
            for (int n : neighbors)
            {
                if (binaryOverview[n])
                {
                    auto it = std::find(coreIndices.begin(), coreIndices.end(), n);
                   // if (it == coreIndices.end())
                      //      continue; // 'n' is not a core, skip (cant find) //Problem part: what if n is NOT in coreIndices? std::find returns coreIndices.end() → then if (!coreVisited[nCoreIdx])   // OUT OF RANGE (coreVisited only has size C)
                    int nCoreIdx = std::distance(coreIndices.begin(), it);

                    if (!coreVisited[nCoreIdx])
                    {
                        coreVisited[nCoreIdx] = true;
                        queue.push_back(nCoreIdx);
                    }
                }
            }
        }

        // Attach BORDER points
        for (int i = 0; i < N; i++) //is on order, we just iterate over each vector
        {
            if (binaryOverview[i])
                continue;
            if (labels[i] != -1)
                continue;

            for (const auto &pid : cluster.points) //j isnt tho, j loops over random vectors that are NOT in order
            {
                if (geodesic)
                {
                    if (diffusion_distance(i, pid, Psi) <= radius)
                    {
                        labels[i] = clusterID;
                        cluster.points.push_back(i);
                        break;
                    }
                }
                else
                {
                    if (euclideanDist(data[pid], data[i]) <= radius)
                    {
                        labels[i] = clusterID;
                        cluster.points.push_back(i); //i is the oringal index in the data
                        break;
                    }
                }
            }
        }

        Clusters.push_back(cluster);
        clusterID++;
    }

    // Mark remaining points as noise
    for (int i = 0; i < N; i++)
    {
        if (labels[i] == -1)
            labels[i] = -2;
    }

    // ---- HERE: Build index clusters ----
    std::vector<std::vector<int>> clusterIndices(clusterID); //clusterID = num clusters too

    for (int i = 0; i < N; i++)
    {
        if (labels[i] >= 0) // ignore noise -2
            clusterIndices[labels[i]].push_back(i); //if its a good point push back its index, labels is also cluster labels
    }

    std::vector<HDBSCAN::ClusterVectors> clustervecs(clusterIndices.size()); //IT NEEDS TO HAVE A SIZE
    for(size_t i = 0; i < clusterIndices.size(); i++){ //for each cluster
        for(const auto &pid : clusterIndices[i]){ //go over all id's. TECHNICALLY this is pointless since i already have Clusters, but its easier, and i dont want to mod other shit to comply with objects
            clustervecs[i].points.push_back(data[pid]); //in each object (cluster), get the vectors
        }
    }

    //cluster.points.push_back(data[idx]); if i truly need actual vectors
    //std::cout << "Exited: [getCorePoints]\n";
    return {clustervecs, clusterIndices}; //the code does not include noise points in Clusters.
    //clusters is a object based vecor, each object is a vector. and clusterindices is 2d vector based
}

double HDBSCAN::dbcvScore(
    const std::vector<std::vector<double>>& data,
    const std::vector<std::vector<int>>& clusters,
    const Eigen::MatrixXd &Psi,
    bool geodesic)
{
    std::cout << "Entered: [dbcvScore]\n";
    int minPts = getMin_points();
    int K = clusters.size();

    std::vector<double> compactness(K, 0.0);
    std::vector<double> separation(K, std::numeric_limits<double>::max());

    // ---- Compute compactness ----
    for (int cid = 0; cid < K; cid++) {
        const auto& C = clusters[cid];
        int M = C.size();
        if (M <= 1) {
            std::cout << "Skipping cluster " << cid << " because it has 1 or fewer points.\n";
            continue;
        }

        std::cout << "Computing compactness for cluster " << cid << " with " << M << " points.\n";
        double total = 0.0;

        for (int ii = 0; ii < M; ii++) {
            int i = C[ii];
            std::vector<double> dists;
            dists.reserve(M - 1);

            for (int jj = 0; jj < M; jj++) {
                if (ii == jj) continue;
                int j = C[jj];

                if (geodesic) {
                    // Check bounds for geodesics access
                    if (i < 0 || i >= Psi.rows() || j < 0 || j >= Psi.rows()) {
                        throw std::out_of_range("Psi matrix access out of bounds.");
                    }
                    dists.push_back(diffusion_distance(i, j, Psi));
                } else {
                    // Check bounds for data access
                    if (i >= data.size() || j >= data.size()) {
                        throw std::out_of_range("Data matrix access out of bounds.");
                    }
                    dists.push_back(euclideanDist(data[i], data[j]));
                }
            }

            std::sort(dists.begin(), dists.end());
            int k = std::min((int)dists.size(), minPts);

            double sum = 0.0;
            for (int t = 0; t < k; t++) {
                sum += dists[t];
            }
            total += sum / k;
        }

        compactness[cid] = total / M;
        std::cout << "Compactness for cluster " << cid << ": " << compactness[cid] << "\n";
    }

    // ---- Compute separation ----
    for (int cid = 0; cid < K; cid++) {
        const auto& C1 = clusters[cid];
        if (C1.size() <= 1) continue;

        for (int cid2 = 0; cid2 < K; cid2++) {
            if (cid == cid2) continue;

            const auto& C2 = clusters[cid2];
            if (C2.size() <= 1) continue;

            for (int i : C1) {
                for (int j : C2) {
                    double d = 0.0;
                    if (geodesic) {
                        if (i < 0 || i >= Psi.rows() || j < 0 || j >= Psi.rows()) {
                            throw std::out_of_range("Psi matrix access out of bounds.");
                        }
                        d = diffusion_distance(i, j, Psi);
                    } else {
                        if (i >= data.size() || j >= data.size()) {
                            throw std::out_of_range("Data matrix access out of bounds.");
                        }
                        d = euclideanDist(data[i], data[j]);
                    }

                    if (d < separation[cid]) {
                        separation[cid] = d;
                    }
                }
            }
        }
    }

    // ---- Compute normalized cluster DBCV validity ----
    double totalScore = 0.0;
    int totalPts = 0;

    for (int cid = 0; cid < K; cid++) {
        int size = clusters[cid].size();
        if (size <= 1) continue;

        double comp = compactness[cid];
        double sep = separation[cid];

        // Avoid divide by zero
        if (sep == 0 && comp == 0) {
            std::cout << "Skipping cluster " << cid << " because both compactness and separation are 0.\n";
            continue;
        }

        double score = (sep - comp) / std::max(sep, comp);
        totalScore += score * size;
        totalPts += size;

        std::cout << "Cluster " << cid << " score: " << score << ", compactness: " << comp << ", separation: " << sep << "\n";
    }

    if (totalPts == 0) {
        std::cout << "No valid points found in clusters.\n";
        return -1.0;
    }

    std::cout << "Exited: [dbcvScore]\n";
    return totalScore / totalPts;
}