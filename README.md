# What Can Transformers Learn In-Context? A Case Study of Simple Function Classes
## Overview
### Context: What is In-Context Learning?
In-context learning is the paradigm that large language transformer models can use to learn a wide variety of tasks: given a prompt containing examples from a task (input-output pairs) and a new query input, the language model can generate the corresponding output.
>prompt: maison → house, chat → cat, chien → 

>completion: dog

The goal of this paper is to investigate in-context learning by training large language models on relatively simple function classes and analyzing the results.

### Problem Statement
Can in-context learning enable large language models to generalize on simple function classes?

What can we change about a model to improve its ability to in-context learn?

### Approach
This study begins by training a large language transformer model from scratch to in-context learn the class of linear functions. It is trained on input vectors and weight vectors sampled from an isotropic Gaussian distribtion in 20 dimensions. The transformer then approximates an input function and evaluated by its loss (MSE) compared to that of the optimal least squares estimator.

Similar methods are used to gauge the model's ability to generalize to out-of-distribution prompts and more complex function classes, while comparing loss to the state-of-the-art estimator for each class (see results).

To discover what aspects of a model matter for in-context learning, this study varies problem dimension and capacity, curriculum training, and training set size and investigates how this affects loss. 

## Model Overview and Architecture
### Informal Specifications
This study uses a decoder-only transformer in the GPT-2 family with 12 layers, 8 attention heads, and a 256-dimensional embedding space (22.4M parameters). This architecture takes as input a sequence of vectors in its embedding space and predicts the next vector in the sequence within the same space. This is fairly standard for text generation models, but the inputs and outputs are restricted.

### Formal Specifications
The model architecture does not differ from the GPT-2, but it is trained from scratch on specific inputs/outputs. 
Changes from the decoder-only architecture from the Formal Algorithms for Transformers paper are highlighted in `blocks`



## Results
### Ability to In-Context Learn Functions

### What Does This Mean?

## Critical Analysis

## Discussion Questions

## Resource Links
- [Research GitHub Repo](https://github.com/dtsip/in-context-learning)
- More on In-Context Learning: [An explanation of in-context learning as implicit bayesian inference](https://openreview.net/pdf?id=RdJVFCHjUMI)
- More on Transformers' Ability to Approximate Functions: [Transformers can do bayesian inference](https://openreview.net/pdf?id=KSugKcbNf9)
- How transformers learn to learn: [Meta-learning in neural networks: A survey](https://www.computer.org/csdl/journal/tp/2022/09/09428530/1twaJR3AcJW)
- Other Ways to Discover Algorithms for Different Learning Problems: [Data-driven algorithm design](https://arxiv.org/pdf/2011.07177.pdf)
