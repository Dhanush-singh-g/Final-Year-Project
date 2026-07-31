import compiler_gym

# Create the environment
env = compiler_gym.make("llvm-v0")

# List available benchmarks
benchmarks = list(env.datasets.benchmarks())

print(f"Total Benchmarks: {len(benchmarks)}\n")

print("First 10 Benchmarks:")
for benchmark in benchmarks[:10]:
    print(benchmark.uri)

env.close()
