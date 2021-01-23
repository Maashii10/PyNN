import numpy as np
import Function


class Layer:
    def __init__(self, number, function="sigmoid", weight=np.empty([1, 1])):
        self.number = number
        self.function = function
        self.weight = weight
        self.previous_layer = None
        self.output = None
        if(self.function == "sigmoid"):
            self.function_activation = Function.sigmoid
            self.derivate_function_activation = Function.derivate_sigmoid

    def linkTo(self, layer):
        self.previous_layer = layer
        layer.next_layer = self

    def propagation(self):
        if(not self.previous_layer):
            raise Exception("Error, not linked layer. Link layers First")
        self.sum = np.dot(self.weight, self.previous_layer.output)
        self.output = self.function_activation(self.sum)

    def randomWeight(self):
        if(not self.previous_layer):
            raise Exception("Error, not linked layer. Link layers First")
        self.weight = np.random.rand(self.number, self.previous_layer.number)

    def describe(self):
        print("number : ", self.number, " function : ",
              self.function, "\nweight : \n", self.weight, "\noutput : \n", self.output)


class Network:
    def __init__(self, input_size, output_size, hidden_layer=1, hidden_layer_size=[1]):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer = hidden_layer
        self.hidden_layer_size = hidden_layer_size
        self.layers = [Layer(hidden_layer_size[i]) for i in range(
            self.hidden_layer)]  # hidden layers
        self.layers.insert(0, Layer(input_size))  # input layer
        self.layers.append(Layer(output_size))  # output layer
        for i in range(1, hidden_layer+2):
            self.layers[i].linkTo(self.layers[i-1])
            self.layers[i].randomWeight()

    def describe(self):
        for i in range(self.hidden_layer+2):
            self.layers[i].describe()

    def calculate(self, input):
        self.layers[0].output = input
        for i in range(1, self.hidden_layer+2):
            self.layers[i].propagation()


if __name__ == '__main__':
    nn = Network(9, 9, 1)
    nn.calculate([1, 1, 1, 1, 1, 1, 1, 1, 1])
    nn.describe()
