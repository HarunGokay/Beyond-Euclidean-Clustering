#pragma once
#include <iostream>
#include "utils.h"
#include <vector>

std::tuple<std::vector<std::vector<int>>, std::vector<double>> directedknnMatrix(std::vector<std::vector<double>> data, int num_neighbours);
std::tuple<std::vector<std::vector<int>>, std::vector<double>> symmetric_knn(std::vector<std::vector<double>> data, int k);
double computeEps(const std::vector<std::vector<double>>& data);
