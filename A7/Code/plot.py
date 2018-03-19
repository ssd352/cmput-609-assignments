import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import numpy as np

filename = 'steps.baseline.npy'

if os.path.exists(filename):
    data = np.load(filename)
    lmda = 0.90
    d = np.mean(data,axis=0)
    plt.plot(np.arange(1, d.shape[0]+1), d, label='Baseline Sarsa_lambda={}'.format(lmda))
    plt.ylim([100,500])
    plt.xlabel('Episode')
    plt.ylabel('Steps per episode \naveraged over {} runs'.format(data.shape[0]))

filename = 'steps.better.npy'
# filename = '/Users/ssd/Desktop/steps.npy'

if os.path.exists(filename):
    data = np.load(filename)
    lmda = 0.90
    d = np.mean(data, axis=0)
    plt.plot(np.arange(1, d.shape[0] + 1), d, label='Better Sarsa_lambda={}'.format(lmda))
    plt.ylim([100, 500])
    plt.xlabel('Episode')
    plt.ylabel('Steps per episode \naveraged over {} runs'.format(data.shape[0]))
    plt.legend()
    plt.show()

filename = 'Q.npy'
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

if os.path.exists(filename):
    data = np.load(filename)
    # lmda = 0.90
    # d = np.mean(data,axis=0)
    # plt.plot(np.arange(1,d.shape[0]+1),d,label='Sarsa_lambda={}'.format(lmda))
    xv, yv = np.meshgrid(np.linspace(-1.2, 0.5, 50), np.linspace(-0.07, 0.07, 50))
    # print X, Y, Z
    # ax.plot_wireframe(xv, yv, data, linewidth=0.5, color='k')
    surf = ax.plot_surface(xv, yv, data, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.ylim([100,500])
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.legend()
    plt.show()
