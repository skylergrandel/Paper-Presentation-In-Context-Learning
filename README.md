# What Can Transformers Learn In-Context? A Case Study of Simple Function Classes
[Paper by Garg, Tsipras, Liang, and Valiant](https://arxiv.org/pdf/2208.01066.pdf); Presented by Skyler Grandel
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
This study begins by training a large language transformer model from scratch to in-context learn the class of linear functions. It is trained on input vectors and weight vectors sampled from an isotropic Gaussian distribtion in 20 dimensions. The transformer then approximates an input function and is evaluated by its loss (MSE) compared to that of the optimal least squares estimator.

Similar methods are used to gauge the model's ability to generalize to out-of-distribution prompts and more complex function classes, while comparing loss to the state-of-the-art estimator for each class (see results).

To discover what aspects of a model matter for in-context learning, this study varies problem dimension and capacity, curriculum training, and training set size and investigates how this affects loss. 

## Model Overview and Architecture
### Informal Specifications
This study uses a decoder-only transformer in the GPT-2 family with 12 layers, 8 attention heads, and a 256-dimensional embedding space (22.4M parameters). This architecture takes as input a sequence of vectors in its embedding space and predicts the next vector in the sequence within the same space. This is fairly standard for GPT-2, but the inputs and outputs are restricted. This prediction model was trained from scratch on prompts of the form (x<sub>1</sub>,f(x<sub>1</sub>), . . . , x<sub>k+1</sub>, f(x<sub>k+1</sub>)). Therefore, inputs and outputs of the transformer are in the form of its latent embedding space as usual, but the learned linear transformations from input to embedding and embedding to output are quite different because they are trained on sequences of vectors. 

In general, nothing really changes from the standard GPT-2 decoder algorithm, though the authors do ignore predictions made about the input prompts (x<sub>1</sub>, x<sub>2</sub>, ...) and only focus on predictions for output prompts (f(x<sub>1</sub>), f(x<sub>2</sub>), ...).


### Formal Specifications
While the algorithm for the model changes very little, the algorithms for the training changes a fair amount because it is specific to the prompt format.
Changes from the traditional decoder-only training schema from the Formal Algorithms for Transformers paper are highlighted in **bold**

Input: **{𝒙<sub>𝑛</sub>}<sup>𝑁<sub>data</sub></sup> <sub>𝑛=1</sub>, a dataset of sequences of vectors of the form (x<sub>1</sub>, f(x<sub>1</sub>), . . . , x<sub>k+1</sub>, f(x<sub>k+1</sub>)) in the given dimensionality.**

Input: 𝜽, initial decoder-only transformer parameters.

Output: 𝜽ˆ, the trained parameters.

Hyperparameters: 𝑁<sub>epochs</sub> ∈ ℕ, 𝜂 ∈ (0, ∞)

1 for 𝑖 = 1, 2, . . . , 𝑁<sub>epochs</sub> do

2   for 𝑛 = 1, 2, . . ., 𝑁<sub>data</sub> do

3     **_l_ ← length(𝒙<sub>𝑛</sub>)/2, the number of input prompts in 𝒙<sub>𝑛</sub>**

4 		𝑷(𝜽) ← DTransformer(𝒙<sub>𝑛</sub> | 𝜽)

5 		**loss(𝜽) = 1/_l_ \* _&Sigma;_<sup>*l*</sup><sub>t=1</sub> (𝒙<sub>𝑛</sub>[2t+1] - 𝑷(𝜽)[𝒙<sub>𝑛</sub>[2t+1]])<sup>2</sup>, the MSE between the actual and predicted output prompts**

6 		𝜽 ← 𝜽 − 𝜂 · ∇loss(𝜽)

7 	end

8 end

9 return 𝜽ˆ = 𝜽

## Results
### Ability to In-Context Learn Functions
Evaluating the trained Transformer on in-context learning linear functions
![image](https://user-images.githubusercontent.com/54510074/195228100-ac2b4f8e-e83e-420f-9edd-7ecb6d6cf172.png)
  
In-context learning on out-of-distribution prompts
(a) sampling prompt inputs from a non-isotropic Gaussian, (b)adding label noise to in-context examples, (c) restricting in-context examples to a single (random) orthant.
![image](https://user-images.githubusercontent.com/54510074/195228389-1e872b7a-c338-48f5-b932-63de3961f09c.png)
The research further shows similar results for prompts in a low-dimensional subspace, scaled prompts, and other out-of-distribution prompts.
  
Transformer in-context learning on more complex function classes
![image](https://user-images.githubusercontent.com/54510074/195229471-eb2a0a2b-d63d-4036-ba52-71ea707512bd.png)
For (d), the transformer was actually trained to predict the output of a 2-layer NN, but then evaluated on linear functions.

### What Does This Mean?
The authors use the data they have gathered to find what factors contribute to successful in-context learning. These include:
- **Problem Dimension and Capacity.** Error is reduced when model capacity increases and/or problem dimensionality decreases.
- **Curriculum Training.** This is when a model is trained initially on more simple data and then gradually increasing the complexity throughout the training. More specifically, this study began with a 5 dimensional subspace and length 11 prompts and then increased the subspace dimension by 1 and the prompt length by 2 every 2000 steps until the subspace dimension reaches the ambient dimension d and the prompt length reaches 2d+1. This drastically speeds up training and improves accuracy over less training.
- **Number of distinct prompts or functions seen during training.** Basically, the larger and more diverse your training set, the better your performance.

## Critical Analysis
While the results and novel findings in this study are incredible, the authors do very little to motivate their research or inform us of why we should care. It seems that much of their purpose in this research was to prove that transformers can learn in-context and find what improves that learning, but the AI community already knows that increasing model complexity, reducing problem complexity, and increasing training set size improves performance and other researchers have already shown the value of curriculum learning. 
  
Furthermore, the authors of the study did not share their models or datasets on huggingface, nor did they give a detailed (much less formal) description of their model and training methods. This makes it more difficult for the community to utilize thier research in any meaningful way.

## Discussion Questions
How do we know that the model doesn't just memorize training prompts?

The authors of this study recommend curriculum learning, where the complexity of the training data starts simple but is gradually increased, to speed up training. Why would curriculum learning help to speed up training?

## Resource Links
- [Research GitHub Repo](https://github.com/dtsip/in-context-learning)
- More on In-Context Learning: [An explanation of in-context learning as implicit bayesian inference](https://openreview.net/pdf?id=RdJVFCHjUMI)
- More on Transformers' Ability to Approximate Functions: [Transformers can do bayesian inference](https://openreview.net/pdf?id=KSugKcbNf9)
- How transformers learn to learn: [Meta-learning in neural networks: A survey](https://www.computer.org/csdl/journal/tp/2022/09/09428530/1twaJR3AcJW)
- Other Ways to Discover Algorithms for Different Learning Problems: [Data-driven algorithm design](https://arxiv.org/pdf/2011.07177.pdf)
