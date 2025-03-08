import numpy as np


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * output.grad
            other.grad += 1.0 * output.grad

        output._backward = _backward
        return output

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return -1 * self

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        output._backward = _backward
        return output

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only support int or float currently"
        output = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * self.data ** (other - 1) * output.grad

        output._backward = _backward
        return output

    def __truediv__(self, other):
        return self * other**-1

    def exp(self):
        output = Value(np.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += output.data * output.grad

        output._backward = _backward
        return output

    def tanh(self):
        output = Value(np.tanh(self.data), (self,), "tanh")

        def _backward():
            self.grad += (1.0 - output.data**2) * output.grad

        output._backward = _backward
        return output

    def relu(self):
        output = Value(np.maximum(0, self.data), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * output.grad

        output._backward = _backward
        return output

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
