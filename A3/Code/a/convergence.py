from __future__ import division, print_function
import numpy as np

def reward(state):
    if state == 100:
        pass
        return 1
    return 0


def value_iteration(ph, threshold = 0.01, max_iterations = 10000, gamma = 1.0):
    V = np.zeros(101, dtype=np.float64)
    # V[100] = 1.0

    err = err_1 = err_2 = None
    for n in range(max_iterations):
        old_V = np.copy(V)
        delta = 0.0
        for state in range(1, 100):
            v = V[state]
            mx = V[state]
            for action in range(min(state + 1, 100 + 1 - state)):
                sm = (1 - ph) * (reward(state - action) + gamma * V[state - action]) + ph * (reward(state + action) + gamma * V[state + action])
                if sm > mx + np.finfo(np.float64).eps:
                    mx = sm  # np.maximum(mx, sm)
            V[state] = mx
            delta = np.maximum(delta, np.absolute(v - V[state]))
        err_2 = err_1
        err_1 = err
        err = np.dot(old_V - V, old_V - V)
        if err_1 and err_2 and err:
            print(np.log(err / err_1) / np.log(err_1 / err_2))
        if delta < threshold:
            break
    return V


def optimal_policy(V, ph, gamma = 1.0):
    policy = np.zeros(101)
    maximals = [[] for i in range(100)]
    
    for state in range(1, 100):
        mx = V[state]
        argmax = -1
        for action in range(min(state + 1, 100 + 1 - state)):
            sm = (1 - ph) * (reward(state - action) + gamma * V[state - action]) + (ph) * (reward(state + action) + gamma * V[state + action])
            if sm + np.finfo(np.float64).eps >= V[state]:
            # if np.absolute(sm - V[state]) <= np.finfo(np.float64).eps:
            # if sm > mx:
                # print('At state ' + str(state) + ', action ' + str(action) + ' is optimal')
                # print(sm - V[state])
                # mx = sm
                # argmax = action
                maximals[state].append(action)
        policy[state] = argmax
        # policy[state] = maximals[state][-1]

        
##        if len(maximals[state]) == 1:
##            policy[state] = maximals[state][0]
##        else:
##            policy[state] = np.random.choice(maximals[state][1:])
        
    # print(policy)
    # print(maximals)
    return maximals


def two_in_one(ph, threshold = 0.01, max_iterations = 10000, gamma = 1.0):
    V = np.zeros(101, dtype=np.float64)
    policy = [0] * 101
    # V[100] = 1.0
    for n in range(max_iterations):
        delta = 0.0
        for state in range(1, 100):
            v = V[state]
            mx = V[state]
            argmax = 0
            for action in range(1, min(state + 1, 100 + 1 - state)):
                sm = (1 - ph) * (reward(state - action) + gamma * V[state - action]) + ph * (reward(state + action) + gamma * V[state + action])
                mx = np.maximum(mx, sm)
                argmax = action
            policy[state] = action
            V[state] = mx
            delta = np.maximum(delta, np.absolute(v - V[state]))
        if delta < threshold:
            break
    return V, policy

    
def gambler(ph, threshold=1e-10 / 2, max_iterations = 10000000, gamma = 1.0):
    V = value_iteration(ph, threshold, max_iterations, gamma)
    policy = optimal_policy(V, ph, gamma)
    # V, policy = two_in_one(ph, threshold, max_iterations, gamma)
    return V, policy

    
if __name__ == "__main__":
    V, policy = gambler(0.55)
    with open('value.dat', 'w') as f:
        for item in V[1:-1]:
            f.write(str(item) + '\n')
    with open('policy.dat', 'w') as f:
        for state, actions in enumerate(policy[1:]):
            for action in actions:
                f.write(str(state + 1) + ' , ' + str(action) + '\n')
