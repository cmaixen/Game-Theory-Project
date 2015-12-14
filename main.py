#!/usr/bin/env python

#NATHANIEL WAS HIER

import numpy as np
import pylab as pl
import random

# 0,False - cooperate
# 1,True  - defect


PD = np.array([[1, -0.5],[1.5, 0]]) #prisonner's dilemma
H = np.array([[1, 0.5],[0.5, 0]]) #harmony
SD = np.array([[1, 0.5],[1.5, 0]]) #snowdrift
SH = np.array([[1, -0.5], [0.5, 0]]) #stag-hunt


# grid size
N=20;

# player's strategies
#p strategy

def Decide_with_prob(player_strategy, player_payoff, strategy,neighbor_payoff, payoff_system):
#     print "payoff_player " + str(player_payoff)
#     print "player_strategy " + str(player_strategy)
#     print "neigbors_strategy " + str(strategy)
#     print "neighbor_payoff " + str(neighbor_payoff)
    length = strategy.shape[0]
    prob = np.zeros(length)

    #calculates the probability
    max_payoff = max(max(p[0:]) for p in payoff_system)
    min_payoff = min(min(p[0:]) for p in payoff_system)

    for i in range(length):
            wi = player_payoff
            wj = neighbor_payoff[i]
            prob[i] = calcprob = (1 + (wj - wi)/ (4* (max_payoff - min_payoff)))/2

    #pick random number and decide to change strategy or not
    random_array = range(length)
    random_neighbor = np.random.choice(random_array)
    strategy_neigbor = strategy[random_neighbor]
    Pij = prob[random_neighbor]
    if player_strategy == strategy_neigbor:
       return player_strategy
    else:
       return np.random.choice([player_strategy, strategy_neigbor], p=[1-Pij,Pij])



def getneighbours( M, i, j ):
    """
    returns the 4 neighbours of i,j from matrix M with
    boundary restrictions
    """
    nh= [];
    (Lenght_X,Lenght_Y)=M.shape; #geeft de lengtematen van matrix
    for ix in [-1,0,1]:
        for jx in [-1,0,1]:
            if abs(ix)==abs(jx) : continue # with  this you will only calculate the neighbour on the left, right, up and down.
            if valid_neighbour(i+ix, j+jx, Lenght_X, Lenght_Y):
                nh = np.append(nh,M[i+ix, j+jx])
            #grenzen worden overschreden
            elif 0 > i+ix:
                l = Lenght_X - 1
                nh = np.append(nh, M[l, j+jx])
            elif i+ix >= Lenght_X:
                 nh = np.append(nh, M[0, j+jx])
            elif i+jx >= Lenght_Y:
                 nh = np.append(nh, M[i+ix, 0])
            #0 > i+jx
            else:
                 l = Lenght_Y - 1
                 nh = np.append(nh, M[i+ix,l])

    return nh.astype(int)

def valid_neighbour(i,j, Lenght_X, Lenght_Y):
    #check if index is valid
#     print i,j
    if (0 <= i < Lenght_X) and (0 <= j < Lenght_Y):
        return True
    else:
        return False


def run( initial, nruns, payoff ):
    """
    - initial holds the initial choice of strategy
    - strat   holds numbers symbolizing the strategy (mapped by num2strat)
    - nruns   is the number of iterations
    """
    S = np.zeros( (N,N,nruns),dtype=np.int ); # strategy array , maakt N op N matrixen n keer aan
    P = np.zeros( (N,N,nruns),dtype=np.int ); # payoff   array
    S[:,:,0]=initial; #initial strategies
    for t in range(nruns-1):
        #for all_players: interact_with_neighbors, give_payoff
        for i in range(N):
            for j in range(N):
                #get neighbours strategy
                nh = getneighbours(S[:,:,t],i,j); # get neighbour strategies
                no = getneighbours(P[:,:,t],i,j); # get neighbour payoffs
                #calculate player payoff = sum of game with his neighbours
                P[i,j,t]=np.sum(payoff[np.zeros(nh.shape[0],dtype=np.int)+S[i,j,t], nh ] );
                no = getneighbours(P[:,:,t],i,j); # get neighbour payoffs
        #for all_pllayers: choose_random_neighbor, change_strategy?
        for i in range(N):
            for j in range(N):
                #get neighbours strategy
                nh = getneighbours(S[:,:,t],i,j); # get neighbour strategies
                no = getneighbours(P[:,:,t],i,j); # get neighbour payoffs
                S[i,j,t+1]= Decide_with_prob(S[i,j,t],  P[i,j,t], nh, no, payoff)

    return (S,P);





def loop(Fc, Payoff, special=False):
    amount_of_runs = 100
    result = np.zeros((amount_of_runs,amount_of_runs))
    initial = np.array([np.random.choice([True,False], p=[Fc,1-Fc]) for i in range(N*N)]).reshape(N,N); # random initial strategies
    if special:
        initial = np.array([0 for i in range(N*N)]).reshape(N,N);
        initial[9,9] = 0
    for k in range(amount_of_runs):
        print k
        (S,P) = run( initial, 100, Payoff);
        result[k] = fraction_cop(S,N)
    return average_of_all_runs(result)



%matplotlib inline
var = PD
PD_results = [loop(0.3,var),loop(0.5,var),loop(0.7,var), loop(1,var,True)]
