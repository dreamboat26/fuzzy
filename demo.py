import numpy as np

# Step 1: Define Nodes, Edges, and Structures
class Node:
    def __init__(self, name, value):
        self.name = name
        self.value = value

class Edge:
    def __init__(self, source, target, weight=1.0):
        self.source = source
        self.target = target
        self.weight = weight

class Structure:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
    
    def adjacency_matrix(self):
        """Creates an adjacency matrix for the structure."""
        size = len(self.nodes)
        matrix = np.zeros((size, size))
        for edge in self.edges:
            source_idx = self.nodes.index(edge.source)
            target_idx = self.nodes.index(edge.target)
            matrix[source_idx, target_idx] = edge.weight
        return matrix

# Step 2: Define a revised Lagrangean-based optimization with gradient correction
def lagrangian_optimization(input_vector, target_vector, adjacency_matrix, iterations=100, lr=0.001):
    """
    A simple optimization function that aims to minimize the Lagrangean loss
    based on pattern recognition and structural constraints.
    """
    # Initialize the weights randomly
    weights = np.random.rand(len(input_vector))
    loss_history = []

    for i in range(iterations):
        # Forward pass: calculate the predicted output
        output = np.dot(weights, input_vector)
        
        # Compute the loss: mean squared error + regularization term
        mse_loss = np.mean((output - target_vector) ** 2)
        reg_loss = np.sum(adjacency_matrix @ weights)
        lagrangian_loss = mse_loss + 0.1 * reg_loss
        loss_history.append(lagrangian_loss)
        
        # Print the loss every 10 iterations
        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}, Loss: {lagrangian_loss:.4f}")
        
        # Gradient descent step with corrected gradient calculation
        gradient_mse = 2 * (output - target_vector) * input_vector
        gradient_reg = 0.1 * (adjacency_matrix @ weights)
        gradient = gradient_mse + gradient_reg
        weights -= lr * gradient

    return weights, loss_history

# Step 3: Create a Sample Structure and Perform Optimization with loss tracking
def demo_optimization():
    # Define sample nodes and edges
    nodes = [Node("A", 0.5), Node("B", 1.0), Node("C", 1.5)]
    edges = [Edge(nodes[0], nodes[1], 0.8), Edge(nodes[1], nodes[2], 1.2)]
    
    # Create a structure and generate the adjacency matrix
    structure = Structure(nodes, edges)
    adjacency_matrix = structure.adjacency_matrix()
    
    # Define input and target data
    input_vector = np.array([0.5, 1.0, 1.5])
    target_vector = np.array([0.8, 1.2, 1.6])
    
    # Perform optimization
    optimized_weights, loss_history = lagrangian_optimization(input_vector, target_vector, adjacency_matrix)
    
    print("\nOptimized Weights:", optimized_weights)
    return loss_history

# Run the demo and get the loss history
loss_history = demo_optimization()

# Visualize the loss history
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Lagrangean Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Lagrangean Loss Over Iterations')
plt.legend()
plt.grid(True)
plt.show()
