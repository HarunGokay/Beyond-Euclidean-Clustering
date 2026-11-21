#pragma once
#include <iostream>
#include "utils.h"
#include <vector>


//radius = distance one vector can reach to see other points. we always use this
//min_points = min points within distance (if, dist < radius) 1/4 points, if 4/4 = core point

// we assign corepoitns to the same cluster with the same radius at this point every single point in the first cluster is a core point


class HDBSCAN{

    public:

    HDBSCAN(double radius_input, int num_points_input)
        : radius(radius_input), min_points(num_points_input){} //set private members

    struct Clusters{
        std::vector<int> points; //Cluster cluster1 etc, or std::vector<Cluster> clusters
    };

    struct ClusterVectors{
        std::vector<std::vector<double>> points;
    };

    std::tuple<std::vector<int>, std::vector<bool>> getCorePoints(const std::vector<std::vector<double>> &data, const Eigen::MatrixXd &Psi, bool); //dist and is it core?
    std::tuple<std::vector<HDBSCAN::ClusterVectors>, std::vector<std::vector<int>>> getClusters(const std::vector<std::vector<double>>& data, const std::vector<int>& coreIndices, const std::vector<bool>& binaryOverview, const Eigen::MatrixXd &Psi,  bool);
    double dbcvScore(const std::vector<std::vector<double>>& data, const std::vector<std::vector<int>> &clusteridxs, const Eigen::MatrixXd &Psi,  bool);
    double getRadius();
    int getMin_points();


    private:

    double radius;
    int min_points;  

};

