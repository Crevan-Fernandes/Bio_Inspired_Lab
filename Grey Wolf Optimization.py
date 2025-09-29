import random
import numpy as np

# ---------- Problem Definition ----------
jobs = [4, 2, 7, 3, 5]       # processing times of jobs
m = 2                        # number of machines
pop_size = 6                 # number of wolves
max_iter = 20                # maximum iterations
dim = len(jobs)              # dimension = number of jobs


# ---------- Fitness Function ----------
def evaluate(schedule):
    """
    Given a job sequence, assign jobs greedily to machines
    and return makespan (maximum completion time).
    """
    machine_load = [0] * m
    for job in schedule:
        idx = min(range(m), key=lambda k: machine_load[k])
        machine_load[idx] += jobs[job]
    return max(machine_load)


# ---------- Helpers ----------
def decode(priority_vector):
    """Convert a continuous priority vector to a job permutation."""
    return list(np.argsort(priority_vector))


def random_wolf():
    """Create a random priority vector."""
    return [random.random() for _ in range(dim)]


# ---------- Initialize Wolves ----------
wolves = [random_wolf() for _ in range(pop_size)]
fitness = [evaluate(decode(w)) for w in wolves]

# identify Alpha, Beta, Delta
def sort_wolves():
    idx = np.argsort(fitness)
    return idx[0], idx[1], idx[2]

alpha_idx, beta_idx, delta_idx = sort_wolves()


# ---------- Main GWO Loop ----------
for iteration in range(1, max_iter + 1):
    a = 2 - iteration * (2 / max_iter)  # control parameter

    for i in range(pop_size):
        X = wolves[i][:]
        new_pos = []
        for d in range(dim):
            # Grey Wolf update for each dimension
            A1, C1 = 2 * a * random.random() - a, 2 * random.random()
            A2, C2 = 2 * a * random.random() - a, 2 * random.random()
            A3, C3 = 2 * a * random.random() - a, 2 * random.random()

            alpha = wolves[alpha_idx][d]
            beta  = wolves[beta_idx][d]
            delta = wolves[delta_idx][d]

            D_alpha = abs(C1 * alpha - X[d])
            D_beta  = abs(C2 * beta  - X[d])
            D_delta = abs(C3 * delta - X[d])

            X1 = alpha - A1 * D_alpha
            X2 = beta  - A2 * D_beta
            X3 = delta - A3 * D_delta

            new_pos.append((X1 + X2 + X3) / 3)

        wolves[i] = new_pos
        fitness[i] = evaluate(decode(new_pos))

    # Update Alpha, Beta, Delta
    alpha_idx, beta_idx, delta_idx = sort_wolves()

    print(f"Iter {iteration:02d} | "
          f"Best Makespan: {fitness[alpha_idx]:2d} | "
          f"Best Schedule: {decode(wolves[alpha_idx])}")

# ---------- Results ----------
best_schedule = decode(wolves[alpha_idx])
best_makespan = fitness[alpha_idx]
print("\n=== Final Result ===")
print("Best Job Order:", best_schedule)
print("Minimum Makespan:", best_makespan)
