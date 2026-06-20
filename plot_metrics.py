import matplotlib.pyplot as plt

# Dataset declaration
cores = [1, 2, 4, 8]
ideal_speedup = [1, 2, 4, 8]

omp_speedup = [1.0, 1.88, 3.40, 4.90]
mpi_speedup = [1.0, 1.78, 3.10, 4.20]

omp_efficiency = [100.0, 94.0, 85.0, 61.3]
mpi_efficiency = [100.0, 89.0, 77.5, 52.5]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Speedup vs Core Allocations
ax1.plot(cores, ideal_speedup, 'k--', label='Ideal Linear Speedup')
ax1.plot(cores, omp_speedup, 'o-', color='teal', linewidth=2, label='OpenMP Speedup')
ax1.plot(cores, mpi_speedup, 's-', color='crimson', linewidth=2, label='MPI Speedup')
ax1.set_xlabel('Number of Processors / Threads (P)')
ax1.set_ylabel('Speedup (S)')
ax1.set_title('Speedup Analysis vs Scaling')
ax1.grid(True, linestyle=':')
ax1.legend()

# Plot 2: Efficiency vs Core Allocations
ax2.plot(cores, omp_efficiency, 'o-', color='teal', linewidth=2, label='OpenMP Efficiency')
ax2.plot(cores, mpi_efficiency, 's-', color='crimson', linewidth=2, label='MPI Efficiency')
ax2.set_xlabel('Number of Processors / Threads (P)')
ax2.set_ylabel('Efficiency (%)')
ax2.set_title('Processor Efficiency Analysis vs Scaling')
ax2.grid(True, linestyle=':')
ax2.legend()

plt.tight_layout()
plt.savefig('parallel_performance_metrics.png', dpi=300)
print("[System] Metrics plots saved successfully as 'parallel_performance_metrics.png'.")