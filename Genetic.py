import Rnn
from random import randint
import numpy as np
import time


class Population:

    def __init__(self, number, input_size, output_size, max_hidden_layer=3, min_hidden_layer=1, max_hidden_layer_size=1, min_hidden_layer_size=1, random_function=False):
        self.number = number
        self.max_hidden_layer = max_hidden_layer
        self.min_hidden_layer = min_hidden_layer
        self.random_function = random_function
        self.min_hidden_layer_size = min_hidden_layer_size
        self.max_hidden_layer_size = max_hidden_layer_size if max_hidden_layer_size > 1 else output_size
        self.population = []
        for i in range(self.number):
            hidden_layer = randint(
                self.min_hidden_layer, self.max_hidden_layer)
            self.population.append(Rnn.Network(input_size, output_size, hidden_layer, np.random.randint(
                self.min_hidden_layer_size, self.max_hidden_layer_size, size=(hidden_layer))))

    def calculateAll(self, inputs):
        for i in range(self.number):
            self.population[i].calculate(inputs[i])

    def describe(self):
        for i in self.population:
            print("Population ", i, " :\n")
            i.describe()

    # def crossOver(self, father, mother):


"""if __name__ == '__main__':
    pp = Population(50, 4, 5)
    pp.calculateAll(np.ones(1000))"""
