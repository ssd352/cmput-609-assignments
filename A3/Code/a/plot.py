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
    x = np.arange(1, 99 + 1, 1)
    for ph in [0.25, 0.4, 0.55]:
 
        value = np.fromfile("value."+ str(ph) +".dat", sep="\n")
        
        # policy = np.fromfile("policy.dat", sep=", \n")
        policy = np.genfromtxt('policy.'+ str(ph) +'.dat', delimiter=',')
        
        fig = plt.figure()
        
        plt.plot(x, value)
        plt.title("Value ($p_h="+str(ph)+"$)")
        plt.ylabel('$v_*(s)$')
        plt.xlabel('State (Current capital)')
        fig.savefig("value." + str(ph) + ".pdf")
        # plt.show()

        # print('data from policy.dat is' +str(policy))
        fig = plt.figure()
        
        # plt.step(x, policy)
        state = policy[:, 0]
        action = policy[:, 1]
        plt.scatter(state, action, s=4, edgecolors='none')
        # plt.step(x, policy)
        plt.title("Policy ($p_h="+str(ph)+"$)")
        plt.ylabel('$\pi(s)$')
        plt.xlabel('State (Current capital)')
        fig.savefig("policy."+ str(ph) +".pdf")
        # plt.show()
