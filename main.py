import math
import string
import numpy as np
from Levenshtein import distance

params_size = 6
mutate_param_rate = 1e-2
mutate_word_rate = 10
initial_length = 10
collision_size = 33
pop_size = 100
margin_of_error = 1e-1
expected_output = 'helloworld'
alphabet = list(string.ascii_lowercase)

def sigmoid(x):
    try:
        power = math.exp(-x)
    except OverflowError:
        power = float('inf')

    return 1.0 / (1.0 + power)

def poly(param, s1, s2):
    p = np.dot(param, [1, s1, s2, s1 * s2, s1 ** 2, s2 ** 2])
    return sigmoid(p)

def main():
    agents = first_agents()

    while True:
        for i, (gender, param, word) in enumerate(agents):
            param = mutate_param(param)
            word = mutate_word(word)
            agents[i] = gender, param, word

        seratonin = [-distance(expected_output, ''.join(a[2])) for a in agents]
        total = sum([math.exp(x) for x in seratonin])
        seratonin = [math.exp(x) / total for x in seratonin]

        for i in range(collision_size):
            agent1_i, agent2_i = np.random.randint(pop_size, size=(2,))
            agent1, agent2 = agents[agent1_i], agents[agent2_i]

            offer = poly(agent1[1], seratonin[agent1_i], seratonin[agent2_i])
            accepts = poly(agent2[1], seratonin[agent1_i], seratonin[agent2_i])

            if (abs(1 - offer - accepts) < margin_of_error):
                new_agents = reproduce(agent1, agent2, offer)

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
        string = '{}{}{}'.format(string[:pos], char, string[pos:])
    elif n == 1:
        string = '{}{}'.format(string[:pos - 1], string[pos:])
    elif n == 2:
        string = '{}{}{}'.format(string[:pos - 1], char, string[pos:])
    return string

def get_child(agent1, agent2, offer):
    word1 = agent1[2]
    word2 = agent2[2]
    max_len = max([len(word1), len(word2)])

    while len(word1) < max_len:
        word1 += ' '

    while len(word2) < max_len:
        word2 += ' '

    pairs = [[word1[i], word2[i]] for i in range(max_len)]
    word = [np.random.choice(pair, p=[offer, 1 - offer]) for pair in pairs]
    word = ''.join(word)

    '''
    # Another implementation:
    word = np.random.choices([word1, word2], p=[offer, 1 - offer])
    '''

    

main()

# ad + bs + c * ds + z d^2 + y * s ^ 2 + bias
