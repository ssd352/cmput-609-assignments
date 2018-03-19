from __future__ import division, print_function
import numpy as np


def expected(ph=0.55, threshold=0.001):
    N = np.zeros(101)
    delta = 1
    while delta > threshold:
        delta = 0
        for cnt in range(1, 100):
            n = N[cnt]
            N[cnt] = 1 + (1 - ph) * N[cnt - 1] + ph * N[cnt + 1]
            delta = np.maximum(delta, np.absolute(n - N[cnt]))
    return N


def prob(ph=0.55):
    prob_arr = np.zeros((100 + 1, 10000 + 1))
    prob_arr[0][0] = 1
    prob_arr[100][0] = 1
    for n in range(1, 10000 + 1):
        for s in range(1, 100):
            prob_arr[s][n] = ph * prob_arr[s + 1][n - 1] + (1 - ph) * prob_arr[s - 1][n - 1]
    return prob_arr


# def number_of_ways():
#     num = np.zeros((100 + 1, 10000 + 1))
#     num[0][0] = 1
#     num[100][0] = 1
#     for n in range(1, 10000 + 1):
#         for s in range(1, 100):
#             num[s][n] = num[s + 1][n - 1] + num[s - 1][n - 1]
#     return num

# prob_dict = {}
# Num_runs = 0
#
#
# def prob2(n, s, ph=0.55):
#     global prob_dict, Num_runs
#     Num_runs += 1
#     print(n, s, Num_runs)
#     if (n, s) in prob_dict:
#         return prob_dict[(n, s)]
#     if s == 0 or s == 100:
#         if n == 0:
#             return 1
#         else:
#             return 0
#     pr = 0
#     if n == 0:
#         return 0
#         # raise ValueError()
#     if s > 0:
#         pr += (1 - ph) * prob2(n - 1, s - 1)
#     if s < 100:
#         pr += ph * prob2(n - 1, s + 1)
#
#     prob_dict[(n, s)] = pr
#     return pr


if __name__ == "__main__":
    ph = 0.55
    arr = prob(ph)
    min_prob = np.min(np.sum(arr[1:-1:1, 0:-1:1], 1))
    print("Probability of reaching a terminal state in 10000 steps or less is", min_prob)
    print("The expected number of steps before termination is", np.max(expected(ph)))
