import Genetic
import Rnn
import numpy as np
from math import floor
from random import randint
# Application example


class MorpionGame:
    def __init__(self, real_player=0):
        self.map = np.zeros(9)

    def describe(self):
        print(self.map.reshape(3, 3))

    def setIndice(self, indice, type=1, takePlacement=False):
        # bounty if the rnn choose an empty case, punition otherwise
        if self.map[indice] != 0:
            return -1
        else:
            self.map[indice] = type
            if(takePlacement == True):
                if ((self.map[0] == self.map[1] and self.map[1] == self.map[2] and self.map[0])
                        or (self.map[3] == self.map[4] and self.map[4] == self.map[5] and self.map[3])
                        or (self.map[6] == self.map[7] and self.map[7] == self.map[8] and self.map[6])
                        or (self.map[0] == self.map[3] and self.map[3] == self.map[6] and self.map[6])
                        or (self.map[1] == self.map[4] and self.map[4] == self.map[7] and self.map[1])
                        or (self.map[2] == self.map[5] and self.map[5] == self.map[8] and self.map[2])
                        or (self.map[0] == self.map[4] and self.map[4] == self.map[8] and self.map[0])
                        or (self.map[2] == self.map[4] and self.map[4] == self.map[6] and self.map[2])):
                    return 100
            return 1

    def random_map(self):
        self.map = np.random.rand(9)
        self.map[self.map > 0.80] = 1
        self.map[self.map < 0.20] = -1
        self.map[(self.map < 1) & (self.map > -1)] = 0

# RNN is 10 inputs for 3x3 square map and indicator about cross or round (0->player 1, 1->splayer 2)


def find_placer():
    nb = 50
    pp = Genetic.Population(nb, 10, 9, max_hidden_layer=2)
    mg = MorpionGame()
    maps = [[0,  1,  1, -1,  0,  1,  0,  1, -1, ], [0,  0,  1, -1,  0,  1,  0,  0, -1, ], [0, -1,  0,  0,  0, -1,  0,  0,  1, ], [0,  0,  0,  0, -1,  0,  1,  0,  0, ], [0,  0,  0,  0,  0, -1, -1,  1,  0, ], [0, -1,  0,  1,  0, -1,  0,  0,  0, ], [0, 1, 0, 0, 0, 0, 0, 1, 0, ], [1,  1,  1, -1,  0,  1,  0,  0,  0, ], [0,  0,  0,  1, -1,  0,  0, -1,  0, ], [0, 0, 1, 0, 0, 0, 0, 0, 0, ], [1,  1,  0,  0,  0,  0,  1, -1, -1, ], [-1,  0, -1,  0,  0,  1,  0,  0,  0, ], [0,  0, -1, -1, -1, -1, -1,  0,  0, ], [-1,  0, -1, -1,  0,  0,  0,  0,  0, ], [0, 0, 0, 0, 0, 1, 0, 0, 0, ], [0, 0, 0, 0, 0, 1, 0, 0, 1, ], [0, -1,  0,  1,  0, -1,  0,  1,  0, ], [0,  0,  0,  0, -1, -1,  0, -1, -1, ], [0,  0, -1, -1,  0, -1,  0,  1, -1, ], [-1, -1,  0,  0,  0,  0,  0,  0,  0, ], [0,  0,  0,  0,  0, -1,  0,  0,  1, ], [0, 0, 0, 0, 0, 1, 0, 0, 0, ], [1,  0,  1,  1, -1, -1,  0, -1,  0, ], [0,  0, -1,  1,  0,  0,  0,  0,  0, ], [0, 0, 0, 0, 1, 0, 0, 1, 0, ], [0,  1,  0,  0,  0,  0, -1, -1,  0, ], [0,  0,  0,  0,  0, -1,  0,  1,  0, ], [0, -1,  0,  0,  1,  0,  0,  1,  0, ], [0, -1,  1, -1,  0,  0, -1,  1, -1, ], [0,  0,  0,  0,  0, -1,  0,  0,  1, ], [0,  0,  1,  0,  0,  0,  1,  1, -1, ], [0, 0, 1, 0, 0, 1, 0, 0, 0, ], [-1,  0,  0,  0,  0,  1,  0,  1,  1, ], [-1,  0,  0,  0,  0,  0,  0,  0,  0, ], [0,  1,  0,  1, -1,  1,  0,  0, -1, ], [1,  0,  1,  0, -1,  0,  0,  1,  0, ], [-1,  0, -1,  1,  0, -1, -1,  0,  1, ], [0,  0,  0,  1, -1,  0,  0,  1,  0, ], [0,  1,  0,  0,  0,  0,  1, -1,  0, ], [0,  0,  0,  0, -1,  0, -1,  0, -1, ], [0, 1, 0, 0, 1, 0, 0, 0, 1, ], [0,  1,  1,  0,  1,  0, -1,  0,  0, ], [0,  0,  0, -1,  0, -1,  0, -1,  1, ], [1,  1,  0,  1,  1, -1,  0, -1,  1, ], [-1,  0,  1,  0, -1,  0,  1,  1,  0, ], [0,  0,  0,  1,  0,  0, -1, -1,  0, ], [1,  0, -1,  0, -1,  1,  1,  0,  0, ], [0,  0,  1,  0,  0, -1,  0,  0,  0, ], [0, -1,  0, -1,  0, -1,  1,  1,  1, ], [-1,  0,  1,  0,  0, -1,  0,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           0,  0, ], [0,  0,  0,  0, -1,  0,  0,  0,  0, ], [0,  0,  0,  1,  0, -1,  1,  0,  0, ], [0,  0,  0,  0,  0,  0,  0, -1, -1, ], [1, -1,  0,  0,  0,  0,  0,  0,  0, ], [1, 0, 0, 1, 0, 0, 0, 0, 0, ], [-1,  0, -1,  1, -1,  0, -1,  1,  0, ], [0,  0,  0,  1,  0,  0,  0,  1, -1, ], [1, -1, -1,  0,  1,  1, -1,  0,  1, ], [0, 0, 0, 0, 1, 0, 0, 0, 0, ], [1, 0, 1, 0, 0, 0, 0, 1, 1, ], [0,  0, -1,  0,  0,  0,  0,  0,  0, ], [0,  0,  1,  0,  0, -1, -1,  0,  0, ], [0, -1,  1, -1, -1,  1, -1,  0,  0, ], [0, 0, 1, 0, 0, 1, 0, 1, 0, ], [0,  0, -1, -1,  0,  0,  0,  0,  0, ], [0,  0,  1, -1,  0,  0,  0,  1,  0, ], [1,  0,  0,  1,  1, -1,  0,  0, -1, ], [0,  0,  0, -1,  0,  0,  0,  0,  0, ], [0, 0, 0, 0, 0, 1, 0, 0, 0, ], [0,  0,  1,  1,  0, -1,  0,  0,  0, ], [0,  0,  0,  0, -1, -1,  0,  1,  0, ], [-1,  0,  0,  1,  0,  0, -1,  0,  0, ], [0,  0,  0,  1,  1,  1,  0,  1, -1, ], [-1,  0,  1, -1,  1, -1, -1,  1,  1, ], [0,  0, -1,  0,  1,  0,  1,  0,  0, ], [0,  1, -1,  0,  0,  1, -1, -1,  1, ], [0,  0,  0, -1,  0,  0,  0,  0, -1, ], [-1, -1,  1, -1,  1,  0,  0, -1,  0, ], [0, 1, 1, 1, 0, 0, 0, 1, 1, ], [-1,  0,  1,  1,  0,  0,  0,  0,  1, ], [1, 0, 0, 0, 0, 0, 1, 0, 0, ], [0,  0,  1,  0,  1, -1,  0,  0,  1, ], [-1, -1,  0,  1, -1,  0,  0,  1, -1, ], [0, 0, 1, 1, 0, 0, 0, 1, 0, ], [0,  1,  0,  0,  1, -1,  0,  0,  0, ], [0, 1, 0, 0, 0, 1, 1, 0, 0, ], [0,  1,  1,  0,  0, -1, -1,  0,  0, ], [0, -1,  1,  1,  0,  0, -1,  1, -1, ], [0,  0,  0, -1,  1, -1, -1,  0,  0, ], [0,  1,  1,  0,  0, -1,  1, -1,  0, ], [0,  0,  0,  0,  0,  0,  0,  0, -1, ], [-1,  0, -1,  1, -1,  1, -1,  0,  0, ], [0,  0,  0, -1,  0,  1,  0,  1,  1, ], [0, -1,  1,  0,  0,  0,  0,  0,  0, ], [1,  0,  0,  0,  1,  0,  0, -1,  0, ], [0, 0, 0, 0, 0, 0, 0, 0, 0, ], [-1,  0,  0,  0,  0,  0, -1,  0,  1, ], [0,  0,  1,  1, -1,  0,  0,  0,  0, ], [0, -1, -1,  1,  0, -1,  0,  0,  0, ], [0,  0, -1,  0,  1,  0,  0,  0,  0, ]]
#    pp.population[0] = pp.population[0].open_model("placer.pickle")
    for k in range(1500000):
        max_rnn = None
        max_map = None
        for i in range(0, nb):
            pp.population[i].score = 0
            pl = randint(0, 1)
            pl = -1 if pl == 0 else 1
            for j in range(100):  # Each RNN play vs himself
                mg.map = np.copy(maps[j])
                pp.population[i].calculate(np.concatenate((mg.map, [pl])))
                score = mg.setIndice(
                    np.argmax(pp.population[i].output), 1, False)
                if(score == -1):
                    #pp.population[i].score += -1
                    break
                pp.population[i].score += score
        print("Generation :", k, " best_score :",
              np.max([i.score for i in pp.population]), " mean :", np.mean([i.score for i in pp.population]))
        # print(np.reshape(max_map, (3, 3)))
        pp.crossOver(45)
        pp.mutation()
        pp.shufflePopulation()


def find_best():
    nb = 50
    pp = Genetic.Population(nb, 10, 9, max_hidden_layer=2)
    pp.population[0] = pp.population[0].open_model("placer.pickle")
    mg = MorpionGame()
    for k in range(1500000):
        max_rnn = None
        max_map = None
        for i in range(0, nb, 2):
            mg.map = np.zeros(9)
            pp.population[i].score = 0
            pp.population[i+1].score = 0
            for j in range(5):  # Each RNN play vs another RNN
                pp.population[i].calculate(np.concatenate((mg.map, [0])))
                score = mg.setIndice(
                    np.argmax(pp.population[i].output), 1, True)
                if(score == -1):
                    pp.population[i].score += -1
                    break
                pp.population[i].score += score
                if(j == 4 or score == 100):
                    break

                pp.population[i+1].calculate(np.concatenate((mg.map, [1])))
                score = mg.setIndice(
                    np.argmax(pp.population[i+1].output), -1, True)
                if(score == -1):
                    pp.population[i+1].score += -1
                    break
                pp.population[i+1].score += score
                if(score == 100):
                    break
            if(pp.population[i].score + pp.population[i+1].score >= 100):
                print(mg.map)
                input()
        print("Generation :", k, " best_score :",
              np.max([i.score for i in pp.population]), " mean :", np.mean([i.score for i in pp.population]))
        # print(np.reshape(max_map, (3, 3)))
        pp.crossOver(45)
        pp.mutation()
        pp.shufflePopulation()


if __name__ == '__main__':
    # find_placer()
    """mg = MorpionGame()
    print("[", end='')
    for i in range(100):
        mg.random_map()
        print(mg.map, ",", end='')
    print("]", end='')"""
    # find_best()
    # netText = Rnn.Network.open_model()
    # netText.describe()
