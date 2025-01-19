# Geometry Awareness Function for Adaptive Data Representation

This project implements a **Geometry Awareness Function (GAF)**, a dynamic mechanism for adapting data representations based on the underlying geometric properties. The model supports multiple geometric spaces, including **Euclidean**, **Hyperbolic**, **Spherical**, and **Fractal** geometries, and intelligently selects and embeds data in the most appropriate geometry.

## Key Features

- **Multi-Geometry Embedding**: Supports embedding in different geometric spaces, including Euclidean, Hyperbolic, Spherical, and Fractal spaces.
- **Geometry Detection**: A neural network dynamically detects the underlying geometry of the input data and applies appropriate embeddings.
- **Geometry-Aware Representation**: The model learns to represent data in the most suitable geometric space, enhancing adaptability and flexibility for various tasks.
- **Recursive Fractal Embedding**: Mimics fractal-like embeddings using recursive transformations.

## Model Components

1. **HyperbolicEmbedder**: Projects input data into a hyperbolic space using the Poincaré ball model.
2. **SphericalEmbedder**: Projects data onto the surface of a unit sphere.
3. **FractalEmbedder**: Uses recursive transformations to embed data in a fractal-like space.
4. **GeometryAwarenessFunction**: A neural network that classifies the geometry of the input data and performs adaptive embedding using the most suitable geometry.

## How It Works

- The **GeometryAwarenessFunction** first classifies the geometry of the input data using a classifier.
- Based on the classified geometry, it selects the appropriate embedding method (Euclidean, Hyperbolic, Spherical, or Fractal).
- The model outputs an adaptive data representation suited for the selected geometry.

## Model Usage

1. **Input**: A tensor of input data, where each sample is a feature vector.
2. **Output**: The model outputs the geometry-specific embedding of the input data, as well as the probabilities of each geometry type (Euclidean, Hyperbolic, Spherical, Fractal).

## Test Function

The `test_geometry_awareness` function demonstrates the working of the model by:
- Generating random input data.
- Passing the data through the **GeometryAwarenessFunction**.
- Outputting the resulting embedded features and geometry probabilities.

## Example Output

The example output will display:
- Input shape: The dimensions of the input data.
- Embedded features shape: The shape of the output after applying geometry-aware embedding.
- Geometry probabilities: The predicted probabilities for each geometry type (averaged over the batch).
