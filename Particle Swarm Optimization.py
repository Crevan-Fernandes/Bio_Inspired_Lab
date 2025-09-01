import random

# Objective function: De Jong's Sphere function
def fitness(position):
    x, y = position
    return x**2 + y**2

# PSO parameters
num_particles = 10
max_iter = 10
w = 0.5          # inertia weight
c1 = 1.5         # cognitive coefficient
c2 = 1.5         # social coefficient

# Initialize particles (positions and velocities) in 2D space
particles = [[random.uniform(-10, 10), random.uniform(-10, 10)] for _ in range(num_particles)]
velocities = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(num_particles)]

# Personal best positions and scores
pbest_positions = [p[:] for p in particles]
pbest_scores = [fitness(p) for p in particles]

# Global best position and score
gbest_index = pbest_scores.index(min(pbest_scores))
gbest_position = pbest_positions[gbest_index][:]
gbest_score = pbest_scores[gbest_index]

print(f"Initial global best position: {gbest_position}, fitness: {gbest_score:.6f}\n")

# PSO main loop
for iteration in range(1, max_iter + 1):
    for i in range(num_particles):
        r1, r2 = random.random(), random.random()

        # Update velocity
        velocities[i][0] = (w * velocities[i][0]
                            + c1 * r1 * (pbest_positions[i][0] - particles[i][0])
                            + c2 * r2 * (gbest_position[0] - particles[i][0]))

        velocities[i][1] = (w * velocities[i][1]
                            + c1 * r1 * (pbest_positions[i][1] - particles[i][1])
                            + c2 * r2 * (gbest_position[1] - particles[i][1]))

        # Update position
        particles[i][0] += velocities[i][0]
        particles[i][1] += velocities[i][1]

        # Calculate fitness
        score = fitness(particles[i])

        # Update personal best
        if score < pbest_scores[i]:
            pbest_scores[i] = score
            pbest_positions[i] = particles[i][:]

            # Update global best
            if score < gbest_score:
                gbest_score = score
                gbest_position = particles[i][:]

    # Print iteration summary
    print(f"Iteration {iteration}: Global Best Position = {gbest_position}, Fitness = {gbest_score:.6f}")

print("\nOptimization finished.")
print(f"Best position found: {gbest_position}")
print(f"Best fitness value: {gbest_score:.6f}")
