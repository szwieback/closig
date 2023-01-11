'''
Created on Jan 5, 2023

@author: simon
'''
N = 5

for n0 in range(N):
    for n1 in range(n0 + 1, N):
        ind = (n0 + 1) * N - n0 * (n0 + 1) // 2 + (n1 - n0 - N - 1)
        print(n0, n1, ind)