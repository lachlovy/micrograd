import numpy as np

from .engine import Value


class Neuron:
    def __init__(self, num_inputs):
        self.w = [Value(np.random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.b = Value(np.random.uniform(-1, 1))

    def __call__(self, x):
        return (sum([w * x for w, x in zip(self.w, x)]) + self.b).tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, num_inputs, num_outputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_outputs)]

    def __call__(self, x):
        return (
            [neuron(x) for neuron in self.neurons]
            if len(self.neurons) > 1
            else self.neurons[0](x)
        )

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, num_inputs, num_outputs, hidden_size, num_hidden):
        self.hidden = [
            Layer(num_inputs if i == 0 else hidden_size, hidden_size)
            for i in range(num_hidden)
        ]
        self.output = Layer(hidden_size, num_outputs)

    def __call__(self, x):
        for layer in self.hidden:
            x = layer(x)
        return self.output(x)

    def parameters(self):
        return [p for layer in self.hidden + [self.output] for p in layer.parameters()]
