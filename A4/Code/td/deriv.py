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
        filename = 'king/run.{:02d}.csv'.format(i+1)
        with open(filename) as f:
            for line in f:
                time = int(line.strip())
                run_matrix[episode][i] = time
                episode += 1

X = np.average(run_matrix, axis=1)
Y = np.arange(num_episodes + 1)
diff = np.fromiter((X[i + 1] - X[i] for i in range(X.size - 1)), np.float64)
# print(Y[100:-1], diff[5:])
print ((X[200] - X[200 - 5]) / 5)
plt.figure()
plt.plot(Y[:num_episodes], diff)
plt.xlabel('Episode #')
plt.ylabel('Length of episode')
plt.savefig('deriv.pdf')

plt.figure()
plt.plot(Y[100:num_episodes], diff[100:])
plt.xlabel('Episode #')
plt.ylabel('Length of episode')
plt.savefig('deriv.100.pdf')
# plt.show()

