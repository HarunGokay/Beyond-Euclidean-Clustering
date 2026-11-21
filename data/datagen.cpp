#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include <ctime>
#include <cstdlib>
#include "datagen.h"

std::vector<std::vector<double>> generateClusters(
    int num_points, int num_clusters,
    double cluster_spacing, double noise_std)
{
    if (num_points <= 0)
        throw std::invalid_argument("num_points must be positive");
    if (num_clusters <= 0)
        throw std::invalid_argument("num_clusters must be positive");
    if (num_clusters > num_points)
        throw std::invalid_argument("num_clusters cannot exceed num_points");

    std::vector<std::vector<double>> data;
    data.reserve(num_points);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, noise_std);

    // --- Generate cluster centers arranged roughly in a circle ---
    std::vector<std::pair<double, double>> centers2D;
    centers2D.reserve(num_clusters);

    const double radius = cluster_spacing * std::max(1.0, std::sqrt(num_clusters) / 2.0);
    const double angle_step = 2.0 * M_PI / num_clusters;

    for (int c = 0; c < num_clusters; ++c) {
        double angle = c * angle_step;
        double x = radius * std::cos(angle);
        double y = radius * std::sin(angle);
        centers2D.emplace_back(x, y);
    }

    // --- Random cluster assignment for variety ---
    std::uniform_int_distribution<> cluster_dist(0, num_clusters - 1);

    for (int i = 0; i < num_points; ++i) {
        int cluster_idx = cluster_dist(gen);
        const auto& center = centers2D[cluster_idx];

        std::vector<double> point(5);
        // Strong separation in first two dimensions
        point[0] = center.first + noise(gen);
        point[1] = center.second + noise(gen);
        // Remaining three: small random variation around 0
        for (int d = 2; d < 5; ++d)
            point[d] = noise(gen);

        data.push_back(std::move(point));
    }

    return data;
}


std::vector<std::vector<double>> generateTorus(
    int num_points, double R, double r, double noise_std)
{
    if (num_points <= 0)
        throw std::invalid_argument("num_points must be positive");

    std::vector<std::vector<double>> data;
    data.reserve(num_points);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> angle(0.0, 2.0 * M_PI);
    std::normal_distribution<> noise(0.0, noise_std);

    for (int i = 0; i < num_points; ++i)
    {
        double u = angle(gen); // around the big circle
        double v = angle(gen); // around the tube

        std::vector<double> point(3);
        point[0] = (R + r * std::cos(v)) * std::cos(u) + noise(gen);
        point[1] = (R + r * std::cos(v)) * std::sin(u) + noise(gen);
        point[2] = r * std::sin(v) + noise(gen);

        data.push_back(std::move(point));
    }

    return data;
}

std::vector<std::vector<double>> generateSwissRoll(
    int num_points, double noise_std)
{
    if (num_points <= 0)
        throw std::invalid_argument("num_points must be positive");

    std::vector<std::vector<double>> data;
    data.reserve(num_points);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> unit(0.0, 1.0);
    std::uniform_real_distribution<> height(0.0, 20.0);
    std::normal_distribution<> noise(0.0, noise_std);

    for (int i = 0; i < num_points; ++i)
    {
        double t = 1.5 * M_PI * (1.0 + 2.0 * unit(gen)); // 1.5pi to 4.5pi

        std::vector<double> point(3);
        point[0] = t * std::cos(t) + noise(gen);
        point[1] = height(gen) + noise(gen);
        point[2] = t * std::sin(t) + noise(gen);

        data.push_back(std::move(point));
    }

    return data;
}

void generateInvertedMobiusStrip(
    std::vector<std::vector<double>> &data,
    int numRows,
    int numDims)
{

    double width = 0.4;
    double R = 1.0;
    double tiltAmplitude = 0.35; // controls how much the strip tilts in/out of plane
    double zWaveAmp = 0.15;      // gentle centerline Z drifting

    if (numRows < 1)
        return;

    data.assign(numRows, std::vector<double>(numDims, 0.0));
    std::srand((unsigned)std::time(nullptr));

    const double twistSign = -1.0; // inverted twist

    for (int i = 0; i < numRows; ++i)
    {

        double u = 2.0 * M_PI * (double)i / (double)numRows;

        // width coordinate
        double v = width * (((std::rand() % 10001) / 10000.0) - 0.5);

        // centerline dir vectors
        double rx = std::cos(u), ry = std::sin(u), rz = 0.0;
        double tx = -std::sin(u), ty = std::cos(u), tz = 0.0;

        // Mobius twist rotation
        double halfTheta = twistSign * 0.5 * u;
        double c = std::cos(halfTheta);
        double s = std::sin(halfTheta);

        // XY width direction before tilt
        double wx = c * rx + s * tx;
        double wy = c * ry + s * ty;

        // vertical component added for beauty and anti-pancaking
        double wz = tiltAmplitude * std::sin(3.0 * u);

        // normalize full width direction
        double mag = std::sqrt(wx * wx + wy * wy + wz * wz);
        wx /= mag;
        wy /= mag;
        wz /= mag;

        // center circle base position
        double cx = R * rx;
        double cy = R * ry;
        double cz = zWaveAmp * std::sin(2.0 * u); // smoother vertical sweep

        // final point
        double x = cx + v * wx;
        double y = cy + v * wy;
        double z = cz + v * wz;

        data[i][0] = x;
        data[i][1] = y;
        data[i][2] = z;

        // high-dimensional noise
        for (int j = 3; j < numDims; ++j)
            data[i][j] = ((std::rand() % 10001) / 10000.0) - 0.5;
    }
}
