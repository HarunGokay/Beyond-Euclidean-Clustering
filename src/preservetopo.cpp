#include <iostream>
#include "utils.h"
#include <vector>
#include "preservetopo.h"
#include <algorithm>
#include <execution>
#include <unordered_set>

std::tuple<std::vector<std::vector<int>>, std::vector<double>> directedknnMatrix(std::vector<std::vector<double>> data, int num_neighbours)
{ // knn cares about order
    // std::cout << "Entered: [knn]\n";

    std::vector<std::vector<int>> knn_result;

    for (size_t idx = 0; idx < data.size(); idx++)
    {

        if (data[idx].empty())
        {
            throw std::invalid_argument("[KNN] Empty vector!");
        }

        std::vector<int> neighboursofvec;
        neighboursofvec.reserve(num_neighbours); // was resize first
        std::vector<std::pair<double, int>> distance_buffer;

        for (size_t i = 0; i < data.size(); i++)
        {
            if (!data[i].empty())
            {
                if (i != idx)
                {
                    double dist = euclideanDist(data[idx], data[i]); // local, so euclidean is fine. geodesic when global
                    distance_buffer.emplace_back(dist, i);
                }
            }
            else
            {
                throw std::invalid_argument("[KNN] Empty vector!");
                continue;
            }
        }
        std::sort(distance_buffer.begin(), distance_buffer.end(),
                  [](const auto &a, const auto &b)
                  { return a.first < b.first; }); // we want the distance sorted, not the indexes. we need those
        //(1.1, 2), (22.4, 0), (2.5, 1) for example

        for (size_t i = 0; i < num_neighbours && i < distance_buffer.size(); i++)
        {
            neighboursofvec.push_back(distance_buffer[i].second); // get the actual index
        }

        knn_result.push_back(std::move(neighboursofvec));
    }

    std::vector<double> sigmas;

    int node = 0;
    for(const auto &vec : knn_result){
        int kth = std::min(10, (int)vec.size() - 1); //10, could be median or 5 -20
        int node_kth_nn = vec[kth]; //vec is already sorted by distance, so this is the kth nearest neighbour
        double weight = std::max(euclideanDist(data[node], data[node_kth_nn]), 1e-8);
        sigmas.push_back(weight);
        node++;
    }
    node = 0;

    return {knn_result, sigmas};
}

std::tuple<std::vector<std::vector<int>>, std::vector<double>> symmetric_knn(std::vector<std::vector<double>> data, int k)
{
    // First: compute the directed KNN exactly as before.
    auto [directed, sigmas] = directedknnMatrix(data, k); //directed, unnsymetric

    // Second: build symmetric adjacency via union.
    size_t n = data.size();
    std::vector<std::vector<int>> symmetric(n);

    // Use sets to avoid duplicates. You're not in performance hell yet.
    std::vector<std::unordered_set<int>> buffer(n);

    for (size_t i = 0; i < n; i++)
    {
        // All directed neighbors of i
        for (int nb : directed[i])
        {
            buffer[i].insert(nb);
            buffer[nb].insert(i); // symmetry
        }
    }

    // Convert sets → vectors
    for (size_t i = 0; i < n; i++)
    {
        symmetric[i].reserve(buffer[i].size());
        for (int nb : buffer[i])
            symmetric[i].push_back(nb);
    }

    return {symmetric, sigmas};
}

double computeEps(const std::vector<std::vector<double>> &data)
{
    // std::cout << "Entered: [computeEps]\n";
    int N = data.size();
    std::vector<double> pointMeans(N, 0.0);

    for (int i = 0; i < N; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < N; j++)
        {
            if (i == j)
                continue;
            sum += euclideanDist(data[i], data[j]);
        }
        pointMeans[i] = sum / (N - 1);
    }

    double totalMean = 0.0;
    for (double d : pointMeans)
        totalMean += d;
    // std::cout << "Exited: [computeEps]\n";
    return totalMean / N;
}
