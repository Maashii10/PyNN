import Rnn
from random import randint
from random import sample
import numpy as np
import time
import operator
from math import floor


class Population:

    def __init__(self, number, input_size, output_size, max_hidden_layer=3, min_hidden_layer=1, max_hidden_layer_size=1, min_hidden_layer_size=1, random_function=False):
        self.number = number
        self.max_hidden_layer = max_hidden_layer
        self.min_hidden_layer = min_hidden_layer
        self.random_function = random_function
        self.input_size = input_size
        self.output_size = output_size
        self.min_hidden_layer_size = min_hidden_layer_size
        self.max_hidden_layer_size = max_hidden_layer_size if max_hidden_layer_size > 1 else output_size
        self.population = []
        for i in range(self.number):
            # 2 hidden layer with 2/3 of hidden nodes (can be impriove in the futur)
            self.population.append(Rnn.Network(input_size, output_size, 2, [
                                   floor(2/3*input_size), floor(2/3*input_size)]))

    def calculateAll(self, inputs):
        for i in range(self.number):
            self.population[i].calculate(inputs[i])

    def describe(self):
        for i in self.population:
            print("Population ", i, " :\n")
            i.describe()

#test
    # mother is best, father is second best of previous population
    def crossOver(self, number):
        self.population.sort(key=operator.attrgetter('score'), reverse=True)
        mother = self.population[0]
        father = self.population[1]
        children = []
        for i in range(number):
            children.insert(0, Rnn.Network(mother.input_size, mother.output_size, 2, [
                            floor(2/3*mother.input_size), floor(2/3*mother.input_size)]))
            for l1, l2, cpt in zip(mother.layers, father.layers, range(len(mother.layers))):
                inter_w = []
                for w1, w2 in zip(l1.weight, l2.weight):
                    # same chance between both parents
                    inter_w.append(w1 if randint(0, 100) > 50 else w2)
                inter_w = np.array(inter_w)
                children[0].setLayer(cpt, inter_w)
        # We take residue to complete list
        residue = sample(self.population, len(self.population)-number-2)
        self.population = []
        self.population.append(mother)
        self.population.append(father)
        self.population += children
        self.population += residue
        for i in self.population:
            i.score = 0

    def mutation(self):
        # no mutation on parents
        for i in range(2, self.number):
            for j in self.population[i].layers[1:]:
                inter_w = np.random.ranf(
                    (np.size(j.weight, 0), np.size(j.weight, 1)))-0.5  # between 0.0-1.0
                # 30% of mutation, can be improve
                inter_w[(inter_w > 0.1)] = 0
                inter_w[inter_w < - 0.1] = 0
                j.weight += inter_w

    def shufflePopulation(self):
        np.random.shuffle(self.population)

    def resetScore(self):
        for i in self.population:
            i.score = 0

    def resetPopulation(self):
        self.population = []
        for i in range(self.number):
            # 2 hidden layer with 2/3 of hidden nodes (can be impriove in the futur)
            self.population.append(Rnn.Network(self.input_size, self.output_size, 2, [
                                   floor(2/3*self.input_size), floor(2/3*self.input_size)]))
