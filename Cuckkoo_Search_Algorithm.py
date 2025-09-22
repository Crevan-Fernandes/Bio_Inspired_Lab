import numpy as np
import random

# Sigmoid activation function and its derivative for neural network
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define a simple feedforward neural network
def neural_network(weights, inputs):
    input_layer = inputs
    hidden_layer = sigmoid(np.dot(input_layer, weights['W1']) + weights['b1'])
    output_layer = sigmoid(np.dot(hidden_layer, weights['W2']) + weights['b2'])
    return output_layer, hidden_layer

# Fitness function (Mean Squared Error)
def fitness_function(weights, inputs, outputs, target):
    predictions, _ = neural_network(weights, inputs)
    error = np.mean((predictions - target) ** 2)
    return error

# Initialize nests (cuckoos) with random weights
def initialize_nests(population_size, input_size, hidden_size, output_size):
    nests = []
    for _ in range(population_size):
        nest = {
            'W1': np.random.randn(input_size, hidden_size),
            'b1': np.random.randn(hidden_size),
            'W2': np.random.randn(hidden_size, output_size),
            'b2': np.random.randn(output_size)
        }
        nests.append(nest)
    return nests

# Lévy flight for generating new solutions
def levy_flight(current_nest, levy_step_size):
    new_nest = {
        'W1': current_nest['W1'] + levy_step_size * np.random.randn(*current_nest['W1'].shape),
        'b1': current_nest['b1'] + levy_step_size * np.random.randn(*current_nest['b1'].shape),
        'W2': current_nest['W2'] + levy_step_size * np.random.randn(*current_nest['W2'].shape),
        'b2': current_nest['b2'] + levy_step_size * np.random.randn(*current_nest['b2'].shape)
    }
    return new_nest


# Replace a fraction of nests with random solutions
def replace_nests_with_random_discovery(nests, discovery_rate, input_size, hidden_size, output_size):
    num_replace = int(discovery_rate * len(nests))
    for i in range(num_replace):
        nests[i] = {
            'W1': np.random.randn(input_size, hidden_size),
            'b1': np.random.randn(hidden_size),
            'W2': np.random.randn(hidden_size, output_size),
            'b2': np.random.randn(output_size)
        }
    return nests

# Cuckoo Search main function
def cuckoo_search(inputs, target, population_size, max_iterations, discovery_rate, levy_step_size):
    input_size = inputs.shape[1]  # Number of input features
    hidden_size = 5  # Hidden layer size (can be adjusted)
    output_size = target.shape[1]  # Number of output neurons (1 for regression or number of classes for classification)
    
    # Initialize nests (cuckoos)
    nests = initialize_nests(population_size, input_size, hidden_size, output_size)
    
    # Track the best nest
    best_nest = nests[0]
    best_fitness = fitness_function(best_nest, inputs, target, target)
    
    # Main loop of Cuckoo Search
    for iteration in range(max_iterations):
        for i in range(population_size):
            # Evaluate fitness of the current nest (cuckoo)
            fitness = fitness_function(nests[i], inputs, target, target)
            
            # If the fitness is better, update the best solution
            if fitness < best_fitness:
                best_nest = nests[i]
                best_fitness = fitness
            
            # Generate new candidate solution via Lévy flight
            new_nest = levy_flight(nests[i], levy_step_size)
            
            # Evaluate fitness of the new solution
            new_fitness = fitness_function(new_nest, inputs, target, target)
            
            # If the new solution is better, replace the old one
            if new_fitness < fitness:
                nests[i] = new_nest
        
        # Replace some nests with random solutions (discovery rate)
        nests = replace_nests_with_random_discovery(nests, discovery_rate, input_size, hidden_size, output_size)
        
        # Print the progress
        print(f"Iteration {iteration + 1}/{max_iterations}, Best Fitness: {best_fitness}")

    return best_nest

# Main function to run the neural network training with Cuckoo Search
if __name__ == "__main__":
    # User-defined parameters
    population_size = int(input("Enter population size: "))
    max_iterations = int(input("Enter max iterations: "))
    discovery_rate = float(input("Enter discovery rate (between 0 and 1): "))
    levy_step_size = float(input("Enter Levy step size: "))
    
    # Example data (XOR problem, you can replace with your data)
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    target = np.array([[0], [1], [1], [0]])  # XOR outputs

    # Train the neural network using Cuckoo Search
    best_weights = cuckoo_search(inputs, target, population_size, max_iterations, discovery_rate, levy_step_size)

    # Final trained model's weights
    print("\nFinal Trained Weights:")
    print("W1:", best_weights['W1'])
    print("b1:", best_weights['b1'])
    print("W2:", best_weights['W2'])
    print("b2:", best_weights['b2'])

    # Optionally test the model
    predictions, _ = neural_network(best_weights, inputs)
    print("\nPredictions on the training set:")
    print(predictions)
