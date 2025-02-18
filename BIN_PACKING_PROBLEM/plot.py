from bruteForce import KnapSack
import numpy as np
from random import randint
import time
import gc
import matplotlib.pyplot as plt


def time_measurement(profits, weights, capacity):
    gc_old = gc.isenabled()
    gc.disable()

    ks = KnapSack(profits, weights, capacity)

    start = time.process_time()
    result = ks.solve_knapsack_brute_force()
    stop = time.process_time()

    if gc_old:
        gc.enable()

    return stop - start


def time_mesurments_for_diffrent_values(profits, weights, capacity, how_many):

    brute_force_times = []
    for n in range(how_many):
        profits.append(randint(1, 100))
        weights.append(randint(1, 100))
        brute_force_times.append(time_measurement(profits, weights, capacity))

    return brute_force_times


def make_bar_plot(brute_force_times, lenght, how_many):
    y_pos = np.arange(len(brute_force_times))

    plt.figure(figsize=(10, 5))

    plt.bar(y_pos, brute_force_times, color='#7070ff')

    plt.xticks(y_pos, range(lenght, how_many + lenght))

    plt.xlabel('Number of elements', fontsize=12, color='#323232')
    plt.ylabel('Time [s]', fontsize=12, color='#323232')
    plt.title('Brute force', fontsize=16, color='#323232')

    plt.show()
    #  plt.savefig(name_of_algorith+'.png')


if __name__ == "__main__":
    weights = [8, 3, 5, 2]
    capacity = 9
    profits = [16, 8, 9, 6]
    lenght = len(weights)
    how_many = 10

    brute_force_times = time_mesurments_for_diffrent_values(profits, weights,
                                                            capacity, how_many)

    make_bar_plot(brute_force_times, lenght, how_many)
