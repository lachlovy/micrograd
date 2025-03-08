# micrograd

This repository is inspired by Andrej Karpathy's Neural Networks course: Zero to Hero course. build an engine that can automatically compute the gradient and do the backward process through a computational graph.

Andrej Karpathy's microgard repository is [micrograd - Github](https://github.com/karpathy/micrograd).

## Installation

### Install dependencies

```bash
pip install -r requirements.txt
```

### Install in editable mode

```bash
pip install -e .
```

## Quick start

Example 1: use `Value` for automatic differentiation

```python
from micrograd.engine import Value

# Create scalar value
x = Value(2.0)
y = Value(3.0)

# Define computational graph
z = x * y + x

# Backpropagation
z.backward()

print(f"z = {z.data}")  # output: z = 8.0
print(f"dz/dx = {x.grad}")  # output: dz/dx = 4.0
print(f"dz/dy = {y.grad}")  # output: dz/dy = 2.0
```

Example 2: Build a simple neural network

```python
from micrograd.nn import Neuron, Layer

# Create a neuron (2 inputs)
neuron = Neuron(2)

# Input data
inputs = [Value(1.0), Value(2.0)]

# Forward propagation
output = neuron(inputs)
print(f"Neuron output: {output.data}")
```

## Project Structure

```
micrograd
├── README.md
├── micrograd
│   ├── __init__.py
│   ├── engine.py
│   └── nn.py
├── pyproject.toml
└── requirements.txt
```