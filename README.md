# Algebraic Two-Level Schwarz Preconditioner

This C++ project implements an algebraic overlapping two-level Schwarz preconditioner based on the paper by Al Daas et al. (2023): "Efficient Algebraic Two-Level Schwarz Preconditioner for Sparse Matrices". The solver is designed to handle large-scale, sparse linear systems of equations Ax = b from the MatrixMarket Collection. The primary goal is to overcome the scalability limits of classical one-level domain decomposition methods by constructing a fully algebraic Spectral Coarse Space. This second level is built efficiently via a local lumping technique in the subdomain overlap, requiring only O(n_i) operations per subdomain.

NOTE on implementation: Eigen does not provide a complex generalized eigensolver. In the complex case, the inversion of matrix Atilda_ii can be difficult since it might be non singular. Thus, it is difficult and computationally inefficient to reduce the problem to a classic eigenvalue problem. The choice is to stick with the Eigen library and only work with real numbers. This is done since the application of this preconditioner to a practical case proposed is CFD, in which only real matrices are involved.

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

## Disclamer: use of AI
This project was developed as part of an academic course. AI tools were used exclusively to support implementation strategies and code debugging.Minor exceptions include specific utility functions, such as the timer, which were generated using AI to maximize efficiency and ease of use.

## Prerequisites and Execution Guide

### Local Linux Environment Setup

To compile and run this project on a local Linux machine (or Windows Subsystem for Linux - WSL), you need a working C++17 compiler and the METIS library installed at the system level.

1. Update your package manager and install the standard build utilities:
   ```bash
   sudo apt update
   sudo apt install build-essential g++ make
   ```

2. Install the official METIS development library:
   ```bash
   sudo apt install libmetis-dev
   ```

3. Ensure your `external/eigen3` folder contains the Eigen library header files before compiling.

4. Compile the project using the provided Makefile:
   ```bash
   make clean
   make
   ```

5. Run the sequential test executable:
   ```bash
   ./schwarz_solver
   ```

### Running on CINECA Galileo100 (G100)

When moving to the CINECA Galileo100 cluster, you do not need to install libraries manually. The cluster provides pre-compiled environments managed through the environment modules system.

1. Connect to the cluster via SSH and navigate to your project directory.

2. Load the required compiler and METIS modules into your current session:
   ```bash
   module load gnu
   module load metis
   ```

3. Compile the code directly on the login node using the same universal Makefile:
   ```bash
   make clean
   make
   ```

4. Since computing nodes on G100 are completely isolated from the internet for security reasons, ensure all your target `.mtx` matrix files are already placed inside your local `matrices/` resources directory before submitting a job.

5. To execute the solver in parallel across multiple cores, you must submit a batch script to the Slurm scheduler. Create a job script (e.g., `job_submit.sh`) with the following structure:

   ```bash
   #!/bin/bash
   #SBATCH --job-name=schwarz_solver
   #SBATCH --output=schwarz_%j.out
   #SBATCH --error=schwarz_%j.err
   #SBATCH --partition=g100_usr_prod
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=4
   #SBATCH --time=00:10:00
   #SBATCH --account=<YOUR_ACCOUNT_CODE>

   # Load the same modules used during compilation
   module load gnu
   module load metis
   module load openmpi

   # Execute the parallel solver using mpirun
   mpirun ./schwarz_solver
   ```

6. Submit your job to the Slurm queue:
   ```bash
   sbatch job_submit.sh
   ```
