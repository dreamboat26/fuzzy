# GroverGPT
GroverGPT is a cutting-edge AI model that combines the strengths of Grover and GPT architectures to generate highly accurate and coherent text for various applications such as natural language understanding, content creation, and more.

## Features

- **State-of-the-Art Performance**: Leverages the latest advancements in language modeling.
- **Customizable**: Fine-tune the model for specific tasks.
- **Scalable**: Efficient implementation for both small-scale and large-scale projects.
- **Open Source**: Fully accessible for research and development.

## Abstract (from paper directly)

Quantum computing is an exciting non-Von Neumann paradigm, offering provable speedups over classical computing for specific problems. However, the practical limits of classical simulatability for quantum circuits remain unclear, especially with current noisy quantum devices. In this work, we explore the potential of leveraging Large Language Models (LLMs) to simulate the output of a quantum Turing machine using Grover’s quantum circuits, known to provide quadratic speedups over classical counterparts.

To this end, we developed GroverGPT, a specialized model based on LLaMA’s 8-billion-parameter architecture, trained on over 15 trillion tokens. Unlike brute-force state-vector simulations, which demand substantial computational resources, GroverGPT employs pattern recognition to approximate quantum search algorithms without explicitly representing quantum states. Analyzing 97K quantum search instances, GroverGPT consistently outperformed OpenAI’s GPT-4o (45% accuracy), achieving nearly 100% accuracy on 6- and 10-qubit datasets when trained on 4-qubit or larger datasets. It also demonstrated strong generalization, surpassing 95% accuracy for systems with over 20 qubits when trained on 3- to 6-qubit data. Analysis indicates GroverGPT captures quantum features of Grover’s search rather than classical patterns, supported by novel prompting strategies to enhance performance. Although accuracy declines with increasing system size, these findings offer insights into the practical boundaries of classical simulatability. This work suggests task-specific LLMs can surpass general-purpose models like GPT-4o in quantum algorithm learning and serve as powerful tools for advancing quantum research.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

Clone this repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/dreamboat26/fuzzy.git
cd GroverGPT

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

GroverGPT was inspired by:

- [Grover](https://arxiv.org/abs/1905.12616)
- [GPT](https://openai.com/research/gpt)
