#!/usr/bin/env python
import numpy as np
import pylab as pl
import random
import math

# 0,False - cooperate
# 1,True  - defect


PD = np.array([[1, -0.5],[1.5, 0]]) #prisonner's dilemma
H = np.array([[1, 0.5],[0.5, 0]]) #harmony
SD = np.array([[1, 0.5],[1.5, 0]]) #snowdrift
SH = np.array([[1, -0.5], [0.5, 0]]) #stag-hunt

# degree = amount of neighbours
degree = 8
# lattice network size
N=20 #4900 players in a 70x70 grid

generations = 1000

# player's strategies
#p strategy

def payoff_calc(p,q,Dg,Dr):
    """
    Returns the payoff for a player using strategy "p" against an opponent
    using strategy "q", or the array of payoffs when playing against
    opponents using the strategies in array "q".
    """
    R = 1.0       #reward
    S = -Dr     #sucker's payoff
    T = 1.0 + Dg  #temptation
    P = 0.0       #punishment

    payoff = (R-S-T+P)*p*q + (S-P)*p + (T-P)*q + P
    return payoff


def shuffle_two_alligned_lists(list1,list2):
    # Given list1 and list2
    list1_shuf = []
    list2_shuf = []
    index_shuf = range(len(list1))
    random.shuffle(index_shuf)
    for i in index_shuf:
        list1_shuf.append(list1[i])
        list2_shuf.append(list2[i])
    return list1_shuf, list2_shuf



#This function should eventually replace Decide_with_prob
def strat_update(player_strategy, player_payoff, neighbour_strategies, neighbour_payoffs, updaterule=1, tau=.2):
    """
    Returns updated strategy of player p depending on payoffs
    and strategies of neighbors n.
    """
    if updaterule == 1: #strategy Imitation Max
        payoffs = np.concatenate(([player_payoff],neighbour_payoffs), axis=0)
        strategies = np.concatenate(([player_strategy],neighbour_strategies), axis=0)
        #We shuffle because when all payoffs are equal a random strategy needs to be returned,
        #with argmax if all are equal only the first will be returned.
        payoffs, strategies = shuffle_two_alligned_lists(payoffs,strategies)
        #get index of the max
        stratind = np.argmax(payoffs)
        new_strategy = strategies[stratind]
        return new_strategy
    elif updaterule ==2: #strategy Fermi-PW
        #select random neighbour
        index_shuf = range(len(neighbour_payoffs))
        random.shuffle(index_shuf)
        nb = index_shuf[1]
        #calculate probability
        prob = 1/(1+math.exp( (player_payoff-neighbour_payoffs[nb])/tau ))
        #update strategy
        change = np.random.binomial(1,prob)
        new_strategy = player_strategy*abs(change-1) + neighbour_strategies[nb]*change
        return new_strategy


def getneighbours( M, i, j ):

    """
    returns the 8 neighbours of i,j from matrix M with
    boundary restrictions
    """
    nh= [];
    (Lenght_X,Lenght_Y)=M.shape; #geeft de lengtematen van matrix
    for ix in [-1,0,1]:
        for jx in [-1,0,1]:
            if ix == 0 and jx == 0: continue # with  this we don't calculate ourselves
            #Deze if-test is enkel juist als beide binnen de grenzen vallen
            #Deze test mag ook verwijderd worden want dit wordt ook getest in het else-gedeelte
            if valid_neighbour(i+ix, j+jx, Lenght_X, Lenght_Y):
                nh = np.append(nh,M[i+ix, j+jx])
                #grenzen worden overschreden
            else:
                #xn x-value neighbor
                #yn y-value neighbor

                #We kijken of x binnen de grenzen valt
                #X < 0
                if 0 > i+ix:
                    xn = Lenght_X - 1
                    #nh = np.append(nh, M[l, j+jx])
                #X >= Lenght_X
                elif i+ix >= Lenght_X:
                    xn = 0
                    #nh = np.append(nh, M[0, j+jx])
                #0 < X < Lenght_X
                else:
                    xn = i+ix
                #We kijken of y binnen de grenzen valt
                # Y >= Lenght_Y
                if i+jx >= Lenght_Y:
                    yn = 0
                    #nh = np.append(nh, M[i+ix, 0])
                #Y < 0
                elif 0 > i+jx:
                    yn = Lenght_Y - 1
                    #nh = np.append(nh, M[i+ix,l])
                # 0 < Y < Lenght_Y
                else:
                    yn = i+jx

                nh = np.append(nh, M[xn, yn])
    return nh.astype(float)



#check if neighbor is not out of the lattice
def valid_neighbour(i,j, Lenght_X, Lenght_Y):
    #check if index is valid
    if (0 <= i < Lenght_X) and (0 <= j < Lenght_Y):
        return True
    else:
        return False

def accumulate_payoff(str_p,str_n,Dg,Dr):
    payoff = 0
    for i in range(len(str_n)):
        payoff += payoff_calc(str_p, str_n[i], Dg, Dr)
    return payoff

def calculate_mean_of_matrix(M):
    (Lenght_X,Lenght_Y)=M.shape
    mean = 0
    for i in range(Lenght_Y):
        for j in range(Lenght_X):
            mean += M[j,i]

    mean = mean / (Lenght_X*Lenght_Y)
    return mean

def run(initial,Dg, Dr,strategy=1, generations=1000, N=20): #3000 generations met 70x70 grid

    """
    - initial holds the initial choice of strategy
    - strat   holds numbers symbolizing the strategy (mapped by num2strat)
    """

    eps = 0.0005
    S = np.zeros( (N,N,generations),dtype=np.float ); # strategy array , maakt N op N matrixen n keer aan
    P = np.zeros( (N,N,generations),dtype=np.float ); # payoff   array
    S[:,:,0]=initial; #initial strategy voor de eerste matrix
    x = calculate_mean_of_matrix(initial)
    x_old = x + 10 * eps

    for t in range(generations-1):

        if(abs(x-x_old)<eps and t > 100):
            print "converged at generation: " + str(t)
            return S, P, t
#         print "generation " + str(t)
        #for all_players: interact_with_neighbors, give_payoff
        for i in range(N):
            for j in range(N):
                #get neighbours strategy
                nh = getneighbours(S[:,:,t],i,j); # get neighbour strategies
                #no = getneighbours(P[:,:,t],i,j); # get neighbour payoffs
                #calculate player payoff = sum of game with his neighbours
                str_p = S[i,j,t] #strategy of player at [i,j,t]
                P[i,j,t]=accumulate_payoff(str_p,nh,Dg,Dr)#update payoff
                #P[i,j,t]=np.sum(payoff[np.zeros(nh.shape[0],dtype=np.int)+S[i,j,t], nh ] );
                #no = getneighbours(P[:,:,t],i,j); # get neighbour payoffs
        #for all_pllayers: choose_random_neighbor, change_strategy?
        for i in range(N):
            for j in range(N):
                #get neighbours strategy
                nh = getneighbours(S[:,:,t],i,j); # get neighbour strategies
                no = getneighbours(P[:,:,t],i,j); # get neighbour payoffs
                str_p = S[i,j,t] #strategy of player at [i,j,t]
                po_p = P[i,j,t] #payoff of player at [i,j,t]
                S[i,j,t+1]= strat_update(str_p,po_p,nh,no,strategy)

        x_old = x
        x = calculate_mean_of_matrix(S[:,:,t])
        #print "X::" + str(x) + "||Xold::" + str(x_old)


    return S,P,t;


#Makes initial discrite strategy matrix with C = 0.5
def init_discrete():
    initial = np.array([np.random.choice([True,False], p=[0.5,0.5]) for i in range(N*N)]).reshape(N,N); # random initial strategies
    return initial

#Makes initial uniform distributed continuous matrix
#interval: [0.0;1.0[
#returns a matrix of (N, N)
def init_continuos():
    uniform = np.random.uniform(0.0,1.0,(N,N))
    for i in range(N):
        for j in range(N):
            uniform[i,j] = round(uniform[i,j], 2)
    return uniform

#Metagrid of 11 x 11 with parameters Dr and Dg
#20 runs
def run_experiment():
    axis = np.arange(0.0,1.1,0.1) #11 points
    grid_results = []
    for i in range(1):
        print 'NEW ITERATION: ' + str(i)
        initial = init_continuos() # random initial strategies
        grid = np.zeros((11, 11))
        for x in range(11):
            Dr = axis[x]
            for y in range(11):
                Dg = axis[y]
                print "Dg Value= " + str(Dg) + "Dr Value= " + str(Dr)
                S,P,t = run(initial,Dg,Dr, generations = generations, N=N)
                if t == generations - 1:
                    mean = 0
                    for i in range(100):
                        result = S[:,:,-i]
                        m = np.mean(result)
                        mean += m
                    grid[y,x] = mean
                    print "mean: " + str(mean)

                else:
                    last_index = generations-t
                    result=S[:,:,-last_index] #take the last strategies
                    mean = np.mean(result) #if we save do  grid[y,x] we can print the grid and respect the axisses
                    grid[y,x] = mean
                    print "mean: " + str(mean)

        grid_results.append(grid)
        mean = calculate_mean_grid(grid_results)
        variance = calculate_variance_grid(grid_results)
        return mean, variance

def calculate_mean_grid(grids):
    amount_of_grids,rows,columns = np.array(grids).shape
    mean_grid = np.zeros((11, 11))
    for i in range(rows):
        for j in range(columns):
            total = 0.0
            for grid in grids:
                total += grid[i,j]
            mean_value = total / amount_of_grids
            mean_grid[i,j] = mean_value
    return mean_grid

def calculate_variance_grid(grids):
    amount_of_grids, rows, columns = np.array(grids).shape
    variance_grid = np.zeros((11,11))
    var = np.var(grids, axis = 2)
    return var_grid



### START EXPERIMENT ###

result, variance = run_experiment()


### SAVE RESULT ###
import pickle

output = open('mean_frac_cop_continous.pkl', 'wb')
mean_frac_cop_continous = result
pickle.dump(mean_frac_cop_continous, output)
output.close()


#### PLOT CODE #####

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
# %matplotlib inline

plt.title('Avarage of Fraction of cooperation with continous values')
# set the limits of the plot to the limits of the data
X = Y = np.arange(0.0,1.1,0.1)
plt.pcolormesh(X,Y,result,cmap=plt.cm.jet, vmin=0.0, vmax=1.0, norm = colors.Normalize() )
# plt.axis([0.0, 1.0, 0.0, 1.0])
plt.xlabel('Dr')
plt.ylabel('Dg')
plt.colorbar()
plt.savefig('mean_frac_cop_continous')
plt.show()
plt.close()


### END PLOT CODE ###
