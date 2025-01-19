# Hyperbolic Chess Trainer 

Auto trains LLM models to play chess by having them simulate the game board in hyperbolic space
## Overview

**Hyperbolic Chess Trainer** is an experimental AI-driven chess playing and training tool. It utilizes hyperbolic geometry-inspired embeddings and neural networks to evaluate chessboard states, make intelligent moves, and store past games for improving decision-making. The project is released under the MIT license, promoting open collaboration and usage.

## Features

- **Hyperbolic Embeddings**: Encodes chessboard states using hyperbolic transformations for efficient and insightful move evaluations.
- **Memory Module**: A custom memory system retrieves historically relevant board states based on hyperbolic similarity metrics.
- **Language Model Integration**: Leverages a transformer-based language model to generate moves and interpret game contexts.
- **Chess Gameplay**: Plays full chess games with up to 100 moves, adhering to legal game rules and maintaining a rich memory of past moves.
- **Customizability**: Adjustable memory size, embedding dimensions, and model checkpoints to suit different computational needs.

## Project Structure

- **`app.py`**: Entry point for running the chess agent.
- **`HyperbolicChessAgent`**: Core class managing chess gameplay and AI decision-making.
- **`HyperbolicChessMemory`**: Memory module for storing and retrieving hyperbolic embeddings of past board states.
- **`HyperbolicChessEmbedding`**: Neural network for generating hyperbolic embeddings from board states.

## License

Licensed under the [MIT License](LICENSE).
