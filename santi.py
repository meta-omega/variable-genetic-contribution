import math
import string
import numpy as np
from Levenshtein import distance

params_size = 6
mutate_param_rate = 1e-5
mutate_word_rate = 10
initial_pop_size = 500
min_pop_size = 1000
max_age = 100
expected_output = 'holamundo'
alphabet = list(string.ascii_lowercase)
offers = {}

#there is a problem with seratonin. it's maximized instead of minimized or the other
get_seratonin = lambda agent: -distance(expected_output, agent[2])
sigmoid = lambda x: 1.0 / (1.0 + np.exp(x))

def poly(param, s1, s2):
    p = np.dot(param, [1, s1, s2, s1 * s2, s1 ** 2, s2 ** 2])
    return sigmoid(p)

def main():
    agents = first_agents()
    seratonins = [get_seratonin(a) for a in agents]
    total = sum([math.exp(x) for x in seratonins])
    seratonin = [math.exp(x) / total for x in seratonins]

    while True:
        agents, seratonins = reproduce(agents, seratonins)
        debug(agents, seratonins)
        agents, seratonins = kill(agents, seratonins)

def first_agents():
    genders = np.random.randint(2, size=(initial_pop_size, 1))
    params = np.random.randn(initial_pop_size, params_size) * 1e-4
    words = np.random.choice(alphabet, (initial_pop_size, len(expected_output)))
    words = [''.join([letter for letter in word]) for word in words]
    ages = np.random.randint(max_age, size=(initial_pop_size,))
    return list(zip(*(genders, params, words, ages)))

def reproduce(agents, seratonins):
    while len(agents) < min_pop_size:
        agent1_i, agent2_i = np.random.randint(len(agents), size=(2,))
        agent1, agent2 = agents[agent1_i], agents[agent2_i]

        offer1 = poly(agent1[1], seratonins[agent1_i], seratonins[agent2_i])
        offer2 = poly(agent2[1], seratonins[agent2_i], seratonins[agent1_i])
        if offer1 + offer2 <= 1:
            offer1 = offer1 / (offer1 + offer2)
            gender, param, word = crossover(agent1, agent2, offer1)
            param = mutate_param(param)
            word = mutate_word(word)
            agent = gender, param, word, 0
            agents.append(agent)
            # seratonins.append(get_seratonin(agent))
            seratonins = [get_seratonin(a) for a in agents]
            total = sum([math.exp(x) for x in seratonins])
            seratonins = [math.exp(x) / total for x in seratonins]
    return agents, seratonins

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

def crossover(agent1, agent2, offer):
    param1, word1 = agent1[1:3]
    param2, word2 = agent2[1:3]

    param = offer * param1 + (1 - offer) * param2

    max_len = max([len(word1), len(word2)])

    while len(word1) < max_len:
        word1 += ' '

    while len(word2) < max_len:
        word2 += ' '

    pairs = [[word1[i], word2[i]] for i in range(max_len)]
    word = [np.random.choice(pair, p=[offer, 1 - offer]) for pair in pairs]
    word = ''.join(word)

    gender = np.random.randint(2)

    return gender, param, word

    '''
    # Another implementation:
    word = np.random.choices([word1, word2], p=[offer, 1 - offer])
    '''

def kill(agents, seratonins):
    new_agents, new_seratonins = [], []
    for i, agent in enumerate(agents):
        gender, param, word, age = agent
        if agent[3] < max_age:
            agent = (gender, param, word, age + 1)
            new_agents.append(agent)
            new_seratonins.append(seratonins[i])

    avg_fitness = np.mean([get_seratonin(a) for a in agents])
    agents, seratonins = new_agents, new_seratonins
    new_agents, new_seratonins = [], []
    for i, agent in enumerate(agents):
        if get_seratonin(agent) >= avg_fitness:
            new_agents.append(agent)
            new_seratonins.append(seratonins[i])

    return new_agents, new_seratonins

def debug(agents, seratonins):
    # print(len(agents), len(seratonins))
    rand_i = np.random.randint(len(agents))
    gender, param, word, age = agents[rand_i]
    avg_fitness = np.mean([get_seratonin(a) for a in agents])
    print(f'Word: {word}. Age: {age}. Fitness {avg_fitness}')

main()

# ad + bs + c * ds + z d^2 + y * s ^ 2 + bias
