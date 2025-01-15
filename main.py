import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the Rastrigin function
def rastrigin(r):
    r = np.array(r)  # Ensure compatibility with lists or arrays
    return 10 * len(r) + np.sum(r ** 2 - 10 * np.cos(2 * np.pi * r))

# Define the PSO algorithm
def pso(cost_func, dim=2, num_particles=30, max_iter=100, w=0.5, c1=1, c2=2):
    # Initialize particle positions and velocities
    particles = np.random.uniform(-5.12, 5.12, (num_particles, dim))
    velocities = np.zeros_like(particles)

    # Initialize personal and global bests
    personal_best_positions = particles.copy()
    personal_best_scores = np.apply_along_axis(cost_func, 1, particles)
    global_best_position = particles[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)

    # List to store positions at each iteration
    particle_positions = []

    # Main PSO loop
    for _ in range(max_iter):
        # Record particle positions for this iteration
        particle_positions.append(particles.copy())

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

    return global_best_position, global_best_score, particle_positions

# Run the PSO algorithm
dim = 2
solution, fitness, positions = pso(rastrigin, dim=dim, max_iter=50)

# Prepare positions for animation
particle_positions = [
    [pos[:, 0].tolist(), pos[:, 1].tolist()] for pos in positions
]

# Visualize the Rastrigin function
x = np.linspace(-5.12, 5.12, 100)
y = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(x, y)
Z = 10 * 2 + (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y))

# 3D Visualization of the PSO result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax.scatter(solution[0], solution[1], fitness, color='red', s=100)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Fitness')
plt.title("Particle Swarm Optimization (PSO) 3D Visualization")
plt.show(block=False)

# Set up the 2D animation plot
figure, ax1 = plt.subplots()
ax1.set_xlim(-5.12, 5.12)
ax1.set_ylim(-5.12, 5.12)

# Initialize data
ln, = plt.plot([], [], 'ro', lw=2)

# Animation initialization
def init():
    ln.set_data([], [])
    return ln,

# Animation update
def update(frame):
    ln.set_data(particle_positions[frame][0], particle_positions[frame][1])
    return ln,

# Animate
ani = FuncAnimation(
    figure, update, frames=len(particle_positions), init_func=init, interval=500, blit=True
)
plt.grid()
plt.title("Particle Swarm Optimization (PSO) Animated Visualization")
plt.show()
