#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt


# def read_data_file(filepath):
#    lines = None
#    with open(filepath) as f:
#        lines = f.readlines
#    out = np.array(float(x) for x in lines)
#    return out

if __name__ == "__main__":
    x = np.arange(1, 1000 + 1, 1)
    ucb = np.fromfile("ucb.dat", sep="\n")
    
    greedy = np.fromfile("eps.greedy.dat", sep="\n")
    opt = np.fromfile("optimistic.dat", sep="\n")
    
    fig = plt.figure()
    
    plt.plot(x, ucb)
    plt.plot(x, greedy)
    plt.plot(x, opt)
    plt.ylabel('%')
    plt.title("Comparison")
    
    plt.legend(["UCB", "Realistic Epsilon Greedy", "Optimistic Epsilon Greedy"], loc="lower right")
    fig.savefig("Comparison.pdf")
    plt.show()
    
    fig = plt.figure()
    
    plt.plot(x, ucb)
    plt.title("UCB")
    plt.ylabel('%')
    fig.savefig("ucb.pdf")
    plt.show()
    
    fig = plt.figure()
    
    plt.plot(x, opt)
    plt.title("Optimistic Epsilon Greedy")
    plt.ylabel('%')
    fig.savefig("optimistic.pdf")
    plt.show()
    
    fig = plt.figure()
    
    plt.plot(x, greedy)
    plt.title("Realistic Epsilon Greedy")
    plt.ylabel('%')
    fig.savefig("greedy.pdf")
    plt.show()