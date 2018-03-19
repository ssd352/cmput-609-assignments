#!/usr/bin/env python

"""
 Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian, Zach Holland
 Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
"""
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    num_runs = 16
    num_episodes = 200
    run_matrix = np.zeros((num_episodes + 1, num_runs))
    for i in range(num_runs):
        episode = 0
        filename = '16/run.{:02d}.csv'.format(i+1)
        with open(filename) as f:
            for line in f:
                time = int(line.strip())
                run_matrix[episode][i] = time
                episode += 1

X = np.average(run_matrix, axis=1)
Y = np.arange(num_episodes + 1)
print(X, Y)
plt.figure()
plt.plot(X, Y)
plt.xlabel('Timesteps')
plt.ylabel('Episodes')
plt.savefig('perf.pdf')
# plt.show()

