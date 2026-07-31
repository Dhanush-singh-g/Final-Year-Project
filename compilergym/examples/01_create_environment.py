import compiler_gym

# Create the LLVM environment
env = compiler_gym.make("llvm-v0")

print("Environment created successfully!")
print(env)

# Always close the environment
env.close()
