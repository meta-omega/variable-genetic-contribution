import numpy as np
import string
import Levenshtein.distance as distance
from math import exp

params_size = 6
mutate_param_rate = 1e-2
mutate_word_rate = 10
initial_length = 10
collision_size = 33
pop_size = 100
expected_output = 'helloworld'
alphabet = list(string.ascii_lowercase)

def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))

def poly(param, (s1, s2)):
    p = np.dot(param, [1, s1, s2, s1 * s2, s1 ** 2, s2 ** 2])
    return sigmoid(x)

def main():
    agents = first_agents()

    while True:
        for i, (gender, param, word) in enumerate(agents):
            param = mutate_param(param)
            word = mutate_word(word)
            agents[i] = gender, param, word

        seratonin = [distance(expected_output, a[2]) for a in agents]

        for i in range(collision_size):
            agent1_i, agent2_i = np.random.randint(pop_size, (2,))
            agent1, agent2 = agents[agent1_i], agents[agent2_i]
            x = poly(agent1[1], (seratonin[agent1_i, agent2_i]))
            print('x:', x)

def first_agents():
    genders = np.random.randint(2, size=(pop_size, 1))
    params = np.random.randn(pop_size, params_size)
    words = np.random.choice(alphabet, (pop_size, initial_length))
    return list(zip(*(genders, params, words)))

def mutate_param(param):
    return param + np.random.randn(params_size) * mutate_param_rate

def mutate_word(string):
    n = np.random.randint(mutate_word_rate)
    char = np.random.choice(alphabet)
    pos = np.random.randint(len(string))
    if n == 0:
        string = f'{string[:pos]}{char}{string[pos:]}'
    elif n == 1:
        string = f'{string[:pos - 1]}{string[pos:]}'
    elif n == 2:
        string = f'{string[:pos - 1]}{char}{string[pos:]}'
    return string

main()

# ad + bs + c * ds + z d^2 + y * s ^ 2 + bias
