import math
import string
import random
import numpy as np
from Levenshtein import distance

pop_size = 128
params_len = 6 # s0, s1, s0s1, s0^2, s1^2, bias.
expected_output = 'sakalakapakapum'
alphabet = list(string.ascii_lowercase)

# TODO: Implement some kind of cost of having a kid.

sigmoid = lambda x: 1.0 / (1.0 + math.exp(-x))

class Agent:
    def __init__(self. parents=None, offer=None):
        self.gender = random.choice(['M', 'F'])

        if parents:
            return
        else:
            expected_len = len(expected_output)
            word_len = math.floor(abs(random.gauss(expected_len - 1, 5)) + 1)
            letters = [random.choice(alphabet) for _ in range(word_len)]
            self.word = ''.join(letters)

            self.params = [random.gauss(0, 2) for _ in range(params_len)]

    def fitness(self):
        x = -1 * distance(expected_output, self.word)
        return sigmoid(x)

    def make_offer(self, receiver):
        s0 = self.fitness()
        s1 = receiver.fitness()

        giver_input = [s0, s1, s0 * s1, s0 ** 2, s1 ** 2, 1]
        offer = sigmoid(np.dot(self.params, giver_input))

        receiver_input = [s1, s0, s1 * s0, s1 ** 2, s2 ** 2, 1]
        evaluation = sigmoid(np.dot(receiver.params, receiver_input))

        # Dado que si alguien te acepta la propuesta van a garchar,
        # les conviene aproximar con la mayor precisión posible la
        # evaluación de la otra persona.

        return evaluation > offer

    def __repr__(self):
        return self.word

def reproduce(agents):
    new agents = []

    for giver in agents:
        for receiver in agents:
            if giver.gender != receiver.gender:
                accepts = giver.make_offer(receiver)

                if accepts:
                    new_agents.append(Agent([giver, receiver]))

    return agents + new_agents

def select(agents):
    agents = sorted(agents, key=lambda x: x.fitness())
    return agents[-pop_size:]

def main():
    agents = [Agent() for _ in range(pop_size)]
    agents = select(agents)

    while len(agents):
        return
        agents = reproduce(agents)
        agents = select(agents)

main()
