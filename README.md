# modified clustering

this is a project i did a while ago when learning c++ and math, i was learning about ml methods and spectral theory, was curious on how i could combine them (improve)

The idea is to run clustering algorithms (k-means and DBSCAN) not directly in raw Euclidean input space, but in an intrinsic spectral coordinate space induced by the heat kernel of a graph Laplacian (a diffusion map embedding). Additionally, a covariance-adaptive Mahalanobis k-means implementation (`kmeansM`) is included for comparison.

---

## Mathematical Background: Spectral Diffusion & Laplacians

### 1. Graph Construction and Local Bandwidths
Given an input dataset $X = \{x_1, x_2, \dots, x_N\} \subset \mathbb{R}^d$, we first build a symmetric $k$-nearest-neighbor ($k\text{NN}$) graph to capture local neighborhood geometry.

For each point $x_i$, a local bandwidth $\sigma_i$ is determined from the distance to its $k$-th neighbor. Edge weights are computed via a localized Gaussian heat kernel:

$$W_{ij} = \exp\left(-\frac{\|x_i - x_j\|_2^2}{\sigma_i \sigma_j}\right) \quad \text{for } (i, j) \in E$$

with $W_{ii} = 0$.

### 2. Normalized Graph Laplacian
Let $D \in \mathbb{R}^{N \times N}$ denote the diagonal degree matrix with entries:

$$D_{ii} = \sum_{j=1}^N W_{ij}$$

The unnormalized graph Laplacian is $L = D - W$. To ensure scale-invariance and numerical stability, we compute the normalized symmetric Laplacian:

$$L_{\text{sym}} = D^{-1/2} (D - W) D^{-1/2} = I - D^{-1/2} W D^{-1/2}$$

$L_{\text{sym}}$ is symmetric and positive semi-definite with real eigenvalues:

$$0 = \lambda_0 \le \lambda_1 \le \lambda_2 \le \dots \le \lambda_{N-1}$$

The trivial zero eigenvalue $\lambda_0 = 0$ corresponds to the stationary harmonic eigenvector $v_0 = D^{1/2} \mathbf{1}$. Non-zero eigenvalues $\lambda_1, \lambda_2, \dots$ characterize the fundamental vibration modes and geometry of the graph.

### 3. Diffusion Coordinates ($\Psi$) and Diffusion Distance
Continuous diffusion on the manifold over a time parameter $t > 0$ is governed by the heat semigroup operator:

$$H_t = \exp(-t L_{\text{sym}})$$

Higher frequencies (large $\lambda_k$) decay exponentially at rate $e^{-\lambda_k t}$, filtering out localized high-frequency noise and leaving the predominant topological structure.

Selecting the first $m$ non-trivial eigenvectors $v_1, v_2, \dots, v_m \in \mathbb{R}^N$ with corresponding eigenvalues $\lambda_1 \le \lambda_2 \le \dots \le \lambda_m$, we define the diffusion coordinate matrix $\Psi \in \mathbb{R}^{N \times m}$:

$$\Psi = \begin{bmatrix} v_1 & v_2 & \dots & v_m \end{bmatrix} \mathrm{diag}\left(e^{-\lambda_1 t}, e^{-\lambda_2 t}, \dots, e^{-\lambda_m t}\right)$$

Each data point $x_i$ is mapped to a vector row $\Psi(x_i) \in \mathbb{R}^m$:

$$\Psi(x_i) = \left[ e^{-\lambda_1 t} v_1(i), \; e^{-\lambda_2 t} v_2(i), \; \dots, \; e^{-\lambda_m t} v_m(i) \right]$$

The diffusion distance $D_t(x_i, x_j)$ between any two points is the Euclidean distance in this diffusion space:

$$D_t(x_i, x_j) = \|\Psi(x_i) - \Psi(x_j)\|_2 = \sqrt{\sum_{k=1}^m e^{-2 \lambda_k t} \left(v_k(i) - v_k(j)\right)^2}$$

This metric reflects the probability of traveling between $x_i$ and $x_j$ via all possible paths of length proportional to $t$. Points that reside on the same manifold or dense cluster exhibit small diffusion distances even if their straight-line Euclidean distance is large.

---

## Methods

- **`kmeansdiff`**: Runs k-means (with k-means++ seeding) on the row vectors of $\Psi \in \mathbb{R}^{N \times m}$. Cluster separation is evaluated using the standard Silhouette coefficient.
- **`dbscandiff`**: Density-based clustering (DBSCAN) using the intrinsic diffusion metric $D_t(x_i, x_j)$. Evaluated using Density-Based Clustering Validation (DBCV).
- **`kmeansM`**: Adaptive Mahalanobis k-means on the raw data space. Utilizes cluster-wise covariance shrinkage, LLT Cholesky checks, and dynamic distance blending during iterations.
- **`all`**: Runs all methods sequentially on the selected dataset.

---

## Datasets (`data/datagen`)

Synthetic generator functions are provided in `data/datagen.h` and `data/datagen.cpp`:

1. **`clusters`**: 5 distinct Gaussian blobs in $\mathbb{R}^5$ arranged along a circle.
2. **`torus`**: 3D torus with major radius $R$ and minor tube radius $r$ plus Gaussian noise.
3. **`swissroll`**: Classic 3D Swiss Roll spiral manifold.
4. **`mobius`**: 3D Inverted Möbius Strip with non-orientable single-sided topology.

### Parameter Tuning Note
Continuous single manifolds (Torus, Swiss Roll, Möbius Strip) are topologically cohesive. With appropriate parameters (e.g. $t = 5.0, \varepsilon = 0.1, \text{minPts} = 10$), DBSCAN correctly recognizes them as a single continuous cluster ($1$ cluster, $0$ noise, $\text{DBCV} = 1.0$). If smaller $\varepsilon$ values are selected, DBSCAN will partition the manifold into local patches or identify sparse regions as noise. For discrete multi-cluster data (`clusters`), settings of $t = 20.0, \varepsilon = 0.01, \text{minPts} = 10$ isolate all 5 clusters with DBCV $> 0.92$.

---

## Build

Requires a C++17 compliant compiler and CMake $\ge 3.14$. Eigen3 and Spectra are linked automatically (or fetched via CMake if not pre-installed).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

---

## Usage

```bash
diffusion_clustering [dataset] [method] [k] [eps] [minpts] [t]
```

### Options:
- **`dataset`**: `clusters` (default), `torus`, `swissroll`, `mobius`
- **`method`**: `all` (default), `kmeansdiff`, `dbscandiff`, `kmeansM`
- **`k`**: Number of clusters for k-means (default: dataset-specific, 5 for clusters, 1 for manifolds)
- **`eps`**: DBSCAN neighborhood radius $\varepsilon$ in diffusion space
- **`minpts`**: DBSCAN minimum points threshold
- **`t`**: Diffusion time parameter $t$

### Examples:

```bash
# Run all methods on Gaussian clusters
./diffusion_clustering clusters all

# Run diffusion DBSCAN on the 3D Torus
./diffusion_clustering torus dbscandiff 1 0.1 10 5.0

# Run diffusion k-means on the Swiss Roll with k=4
./diffusion_clustering swissroll kmeansdiff 4 0.1 10 5.0

# Run Mahalanobis k-means on the Inverted Möbius Strip
./diffusion_clustering mobius kmeansM 4
```

---

## File Structure

| File | Description |
|---|---|
| `src/heatkernel.cpp / .h` | Sparse graph Laplacian, Spectra eigensolver, scale coefficients, diffusion distance |
| `src/hdbscan.cpp / .h` | DBSCAN density clustering engine with DBCV validation score |
| `src/kmeans.cpp / .h` | Standard k-means with k-means++, Silhouette score, and adaptive Mahalanobis `kmeansM` |
| `src/preservetopo.cpp / .h` | Symmetric $k\text{NN}$ graph generation, local $\sigma$ bandwidth calculation |
| `src/utils.cpp / .h` | Distance matrices, adjacency graphs, connected component bridging, medoids |
| `data/datagen.cpp / .h` | Data generators: Gaussian clusters, 3D Torus, Swiss Roll, Inverted Möbius Strip |
| `src/main.cpp` | Command-line driver and benchmark pipeline |

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
