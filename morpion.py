import Genetic
import Rnn
import numpy as np
from math import floor
# Application example


class MorpionGame:
    def __init__(self, real_player=0):
        self.map = np.zeros(9)

    def describe(self):
        print(self.map.reshape(3, 3))

    def setIndice(self, indice, type=1):
        # bounty if the rnn choose an empty case, punition otherwise
        if self.map[indice] != 0:
            return -1000
        else:
            self.map[indice] = type
            return 15


def test():
    nb = 10000
    pp = Genetic.Population(nb, 9, 9, max_hidden_layer=10)
    max_score = -50000
    max_rnn = None
    max_map = None
    for i in range(0, nb, 2):
        mg = MorpionGame()
        for j in range(9):
            pp.population[i].calculate(mg.map)
            pp.population[i].score += mg.setIndice(
                floor(np.argmax(pp.population[i].output)), 1)
            pp.population[i+1].calculate(mg.map)
            pp.population[i+1].score += mg.setIndice(
                floor(np.argmax(pp.population[i+1].output)), -1)
        if(pp.population[i].score > max_score):
            max_score = pp.population[i].score
            max_rnn = pp.population[i]
            max_map = mg.map
        if(pp.population[i+1].score > max_score):
            max_score = pp.population[i+1].score
            max_rnn = pp.population[i+1]
            max_map = mg.map
    print(max_score)
    print(np.reshape(max_map, (3, 3)))
    max_rnn.save_model("model.pickle")


if __name__ == '__main__':
    test()
    #netText = Rnn.Network.open_model()
    # netText.describe()
