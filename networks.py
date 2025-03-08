import numpy as np

from microgard_engine import Value


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


if __name__ == "__main__":
    # create a single neuron
    x = [1, 2, 3]
    n = Neuron(3)
    print(f"Neuron output: {n(x)}")

    # create a mlp layer
    l = Layer(3, 3)
    print(f"Layer output: {l(x)}")

    # create a mlp
    mlp = MLP(3, 1, 3, 3)
    print(f"MLP output: {mlp(x)}")

    # traning a mlp
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]
    lr = 0.05
    num_epochs = 40

    # before training
    pred = [mlp(x) for x in xs]
    print(f"Predictions before training: {pred}")

    # training
    for epoch in range(num_epochs):
        pred = [mlp(x) for x in xs]
        loss = sum([(p - y) ** 2 for p, y in zip(pred, ys)])

        # zero the gradients
        for p in mlp.parameters():
            p.grad = 0.0

        # do backward pass
        loss.backward()

        # update the parameters
        for p in mlp.parameters():
            p.data -= lr * p.grad

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss {loss.data}")

    # after training
    pred = [mlp(x) for x in xs]
    print(f"Predictions after training: {pred}")
