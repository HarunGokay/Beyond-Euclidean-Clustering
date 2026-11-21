#pragma once
#include <vector>

// gaussian blobs, 5 dims, centers on a circle
std::vector<std::vector<double>> generateClusters(
    int num_points, int num_clusters,
    double cluster_spacing, double noise_std);

// 3d torus, R = distance from center to the tube, r = tube radius
std::vector<std::vector<double>> generateTorus(
    int num_points, double R, double r, double noise_std);

// 3d swiss roll
std::vector<std::vector<double>> generateSwissRoll(
    int num_points, double noise_std);

// inverted mobius strip in 3d, extra dims (numDims > 3) are uniform noise
void generateInvertedMobiusStrip(
    std::vector<std::vector<double>>& data,
    int numRows,
    int numDims);
