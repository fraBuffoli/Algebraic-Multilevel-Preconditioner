# Algebraic Two-Level Schwarz Preconditioner

This C++ project implements an algebraic overlapping two-level Schwarz preconditioner based on the paper by Al Daas et al. (2023): "Efficient Algebraic Two-Level Schwarz Preconditioner for Sparse Matrices". The solver is designed to handle large-scale, sparse linear systems of equations Ax = b from the MatrixMarket Collection. The primary goal is to overcome the scalability limits of classical one-level domain decomposition methods by constructing a fully algebraic Spectral Coarse Space. This second level is built efficiently via a local lumping technique in the subdomain overlap, requiring only O(n_i) operations per subdomain.

## Class Structure and Architecture

The project features a modular, object-oriented design that decouples sparse data management, algebraic topology, local spectral solvers, and the parallel iterative engine.

### Data Management and IO
* sparse_matrix.hpp: A lightweight wrapper around Eigen::SparseMatrix<double, Eigen::RowMajor>. It provides an optimized interface for accessing sparse rows and coefficients in Compressed Sparse Row (CSR) format.
* matrix_market_io.hpp: Handles parsing and loading of .mtx input files, supporting both symmetric and non-symmetric matrix storage patterns.

### Domain Decomposition and Topology
* graph_partitioner.hpp: Interfaces with the METIS library. It builds the undirected adjacency graph G(A + A^T) and partitions the n global nodes into N initial, non-overlapping, and balanced subdomains.
* subdomain_topology.hpp: Analyzes the original matrix sparsity pattern via G(A) to extend the initial non-overlapping partitions by a distance of one. It strictly isolates the index sets for each subdomain: interior (Omega_Ii), boundary/overlap (Omega_Gamma_i), and external complement (Omega_ci).
* restriction_operator.hpp: Manages the index mapping between the global system and local subdomains. It implements the apply (restriction) and applyTranspose (prolongation/injection) operations using raw index vectors instead of instantiating large, zero-padded matrices.

### Partition of Unity and Algebraic Splitting
* partition_of_unity.hpp: Computes and stores the local diagonal partition of unity matrix (D_i). It scales overlapping contributions to ensure they sum up to the identity matrix globally.
* local_block_splitting.hpp: Extracts the local subdomain matrix A_ii and implements the lumping algorithm (Definition 3.1). It sums the absolute values of the coefficients connecting the boundary to the external complement (A_Gamma_ci) and subtracts them from the main diagonal of A_ii to assemble the modified matrix tilde{A}_ii.

### Second Level (Spectral Coarse Space)
* local_eigensolver.hpp: Solves the local generalized eigenvalue problem (Equations 3.1 and 3.2) concurrently for each subdomain while handling potential singular null spaces. It filters the spectrum, keeping only the "critical" eigenvectors whose eigenvalues satisfy |lambda| >= 1/tau, where tau is a user-defined threshold. It outputs the local basis matrix Z_i.
* coarse_space.hpp: Gathers and orchestrates the local Z_i bases to assemble the global projection operator R_0^H. It constructs and factorizes (via a direct sparse or dense solver) the small global coarse matrix A_00 = R_0 A R_0^H.

### Iterative Solvers and Parallelism
* one_level_preconditioner.hpp: Implements the classical restricted additive Schwarz (RAS) or additive Schwarz (ASM) one-level preconditioner (Restriction -> Local LU Solve -> Scaling -> Prolongation).
* two_level_preconditioner.hpp: Combines the first-level subdomain corrections with the second-level coarse grid correction. It supports both additive combinations and the more robust deflated combination (M_deflated^{-1}).
* krylov_gmres.hpp: Implements the Generalized Minimal Residual (GMRES) iterative acceleration method with restart capabilities. It drives the global residual minimization and calls the two-level preconditioner at every iteration.
* mpi_context.hpp: Manages the parallel MPI environment, maps hardware processes to mathematical subdomains, and coordinates boundary data exchanges (halo exchange).

### Diagnostics and Profiling
* timers.hpp: High-resolution timing utilities utilizing std::chrono to profile and separate execution costs into Setup Time (partitioning, lumping, local eigensolves, coarse grid assembly) and Solve Time (Krylov iterations).

## Development Roadmap

To systematically verify the algebraic correctness of the implementation, development is divided into three incremental phases:

1. Sequential Phase (1-Level): Implement basic data structures, Matrix Market IO, METIS integration, and achieve GMRES convergence using the one-level preconditioner simulated via sequential loops on a single core.
2. Spectral Phase (2-Level): Introduce local block splitting and eigensolver classes. Algebraically verify that the addition of the coarse space drastically dampens the global GMRES iteration count compared to Phase 1.
3. Parallel Phase (MPI and Cluster): Wrap the modules within the MPI context to distribute memory structures across physical processors. Finalize the codebase for scaling analysis on the Galileo100 cluster (CINECA).