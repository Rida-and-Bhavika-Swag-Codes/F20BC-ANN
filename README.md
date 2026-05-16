# Deep Learning from First Principles

A from-scratch implementation of a feedforward neural network for binary classification built using only NumPy and Pandas.

The project focuses on understanding neural networks at a mechanistic level by implementing forward propagation, backpropagation, multiple optimization strategies and configurable architectures from first principles

---

### Features

- Fully configurable MLP (depth, width, activations)
- Backpropagation implemented from scratch
- Activation functions: Sigmoid, ReLU, Tanh
- Loss functions: BCE, MSE, Hinge
- Optimisation: Batch GD, SGD, Mini-batch GD
- Learning rate: constant + scheduled
- K-fold cross-validation
- Train/test evaluation pipeline
- Hyperparameter experimentation + analysis
  
---

### Dataset

Wisconsin Diagnostic Breast Cancer Dataset (UCI)

Binary classification task: malignant vs benign.

---
## Hyperparameter Effects

#### Epochs
- fast early gain in accuracy  
- saturation ~50 epochs  
- negligible gain >100–150 epochs  

#### Activation Functions
- Sigmoid ≈ ReLU (~97–98%)  
- Tanh ~74% (clear underfit)  
- Sigmoid: smoother early convergence  

#### Learning Rate
- high LR (0.5): fast convergence, unstable, overshoot risk  
- low LR (0.1): stable final accuracy  
- LR scheduling: no clear gain, ↑ training time  

#### Network Depth
- 1–2 layers: similar performance  
- deeper: ↓ accuracy, ↑ loss  
- likely optimisation difficulty (small tabular data)

#### Hidden Neurons
- 1 layer: low sensitivity to neuron count  
- multi-layer: instability at low neuron counts  
- moderate width: most stable  

#### Output Representation
- 2-output > 1-output (slight stability gain)  
- hinge loss pairing more consistent  

#### Optimisation
- Mini-batch GD:
  - best stability / accuracy / generalisation balance  
- Batch GD:
  - stable, slow  
- SGD:
  - high variance, noisy updates  


### Overall Behaviour
- optimisation + LR > architecture impact  
- shallow models sufficient for dataset  
- diminishing returns with complexity increase  


### Summary of Best Configuration
*based on model accuracy*
- Hidden Layers: 1
- Hidden Neurons: 4
- Activation: Sigmoid
- Output Nodes: 2
- Learning Rate: 0.1 (constant)

---

## Usage

Run training:
```bash
python main.py
```
Experiments and visualization:
```bash
model_testing.ipynb
plots.ipynb
```

