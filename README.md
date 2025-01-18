# PEFT Spatial Reasoning Model

This work demonstrates the application of **Parameter Efficient Fine-Tuning (PEFT)** using **LoRA (Low-Rank Adaptation)** to fine-tune a transformer model for **spatial reasoning tasks**.

## Project Overview

The goal of this project is to fine-tune a pretrained transformer model (`SmolLM2-1.7B`) on a set of spatial reasoning prompts. These prompts involve determining the positioning of elements inside certain boundaries. PEFT with LoRA is used to adapt the model efficiently without the need to fine-tune the entire model.

## Key Features

- **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning of large models by freezing the original model weights and updating only low-rank adapters.
- **Spatial Reasoning Tasks**: The model is trained on a set of prompts such as determining whether certain elements are "inside" or "outside" a boundary.
- **Custom Reward Function**: The reward function is based on the similarity between generated and correct responses, incorporating structural similarity and penalizing repetition.

## Conclusion

This repository provides an example of how to fine-tune large language models efficiently using PEFT techniques like LoRA. It showcases the model's ability to handle spatial reasoning tasks with minimal computational overhead.
