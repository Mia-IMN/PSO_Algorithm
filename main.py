import numpy as np
import matplotlib.pyplot as plt


# Define the Rastrigin function
def rastrigin(r):
    r = np.array(r) # This ensures this function works with either python lists or numpy arrays
    return 10 * len(r) + np.sum(r ** 2 - 10 * np.cos(2 * np.pi * r))

# Define the PSO algorithm
def pso_optimized(cost_func, dim=2, num_particles=30, max_iter=100, w=0.5, c1=1, c2=2):
    # Initialize particle positions and velocities
    particles = np.random.uniform(-5.12, 5.12, (num_particles, dim))
    velocities = np.zeros_like(particles)

    # Initialize personal and global bests
    personal_best_positions = particles.copy()
    personal_best_scores = np.apply_along_axis(cost_func, 1, particles)
    global_best_position = particles[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)

    # Main PSO loop
    for _ in range(max_iter):
        # Update velocities
        r1, r2 = np.random.rand(2, num_particles, dim)
        velocities = (
                w * velocities +
                c1 * r1 * (personal_best_positions - particles) +
                c2 * r2 * (global_best_position - particles)
        )

        # Update positions
        particles += velocities

        # Evaluate fitness and update personal and global bests
        fitness = np.apply_along_axis(cost_func, 1, particles)
        better_fitness_mask = fitness < personal_best_scores
        personal_best_positions[better_fitness_mask] = particles[better_fitness_mask]
        personal_best_scores[better_fitness_mask] = fitness[better_fitness_mask]

        if np.min(fitness) < global_best_score:
            global_best_position = particles[np.argmin(fitness)]
            global_best_score = np.min(fitness)

    return global_best_position, global_best_score


# Run the PSO algorithm on the Rastrigin function
dim = 2
solution, fitness = pso_optimized(rastrigin, dim=dim)

# Print the results
print('Optimal solution:', solution)
print('Fitness at solution:', fitness)

# Visualize the Rastrigin function and the solution
x = np.linspace(-5.12, 5.12, 100)
y = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(x, y)
Z = 10 * 2 + (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax.scatter(solution[0], solution[1], fitness, color='red', s=100)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Fitness')
plt.show()
