# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:43:55 2020

@author: Theodor
"""

#%%
def binomial_price_amer_conv_call(S0,sigma,T,r,N,lbd,call_value,no_shares,R=0.4,FV=100,q=0): 
    r"""
    S0 = initial stock price
    
    sigma = annualized vol of the stock
    
    T = option_lifetime
    
    N = number of periods
    
    r = risk-free interest rate
    
    payoff = callable function of 1/more parameters
    """
    import numpy as np
    import math
    a = np.exp((r-q)*T/N)
    u = np.exp(np.sqrt((sigma**2-lbd)*T/N))
    d = np.exp(-np.sqrt((sigma**2-lbd)*T/N))
    pu = (np.exp((r-q)*T/N)-d*np.exp(-lbd*T/N))/(u-d)
    pd = (u*np.exp(-lbd*T/N)-a)/(u-d)
    
    def comb(n,k):
        return math.factorial(n)/(math.factorial(k)*math.factorial(n-k))
    def bin_tree():
        tree = [[S0]]
        for i in range(1,N+1):
            tree.append([S0*u**(i-j)*d**j for j in range(0,i+1)])
        return tree
    binom_tree = bin_tree()[::-1]
    
    values = []
    for i in range(N+1):
        if i==0:
            values.append([max([x*no_shares,FV]) for x in binom_tree[i]])
        elif 1<=i and i<N:
            values.append([max([min([call_value,(values[-1][j]*pu+values[-1][j+1]*pd + R*FV*(1-pu-pd))*np.exp(-r*T/N)]),\
                                no_shares * binom_tree[i][j]]) for j in range(len(values[-1])-1)])
        else:
            values.append([(pu*values[-1][0]+pd*values[-1][1]+(1-pu-pd)*R*FV)*np.exp(-r*T/N)])
    
    return values[-1][0]

def binomial_price_amer_conv(S0,sigma,T,r,N,lbd,no_shares,R=0.4,FV=100,q=0):
    r"""
    S0 = initial stock price
    
    sigma = annualized vol of the stock
    
    T = option_lifetime
    
    N = number of periods
    
    r = risk-free interest rate
    
    payoff = callable function of 1/more parameters
    
    R = recovery rate of the bond. 
    
    lbd = default intensity of the bond. 
    """
    import numpy as np
    import math
    a = np.exp((r-q)*T/N)
    u = np.exp(np.sqrt((sigma**2-lbd)*T/N))
    d = np.exp(-np.sqrt((sigma**2-lbd)*T/N))
    pu = (np.exp((r-q)*T/N)-d*np.exp(-lbd*T/N))/(u-d)
    pd = (u*np.exp(-lbd*T/N)-a)/(u-d)
    def comb(n,k):
        return math.factorial(n)/(math.factorial(k)*math.factorial(n-k))
    def bin_tree():
        tree = [[S0]]
        for i in range(1,N+1):
            tree.append([S0*u**(i-j)*d**j for j in range(0,i+1)])
        return tree
    binom_tree = bin_tree()[::-1]
    values = []
    for i in range(N+1):
        if i==0:
            values.append([max([x*no_shares,FV]) for x in binom_tree[i]])
        elif 1<=i and i<N:
            values.append([max([(values[-1][j]*pu+values[-1][j+1]*pd + R*FV*(1-pu-pd))*np.exp(-r*T/N),\
                                no_shares * binom_tree[i][j]]) for j in range(len(values[-1])-1)])
        else:
            values.append([(pu*values[-1][0]+pd*values[-1][1]+(1-pu-pd)*R*FV)*np.exp(-r*T/N)])
    return values[-1][0]
#%%
def binomial_price_amer_conv2(S0,sigma,T,r,N,lbd,no_shares,R=0.4,FV=100,q=0):
    r"""
    S0 = initial stock price
    
    sigma = annualized vol of the stock
    
    T = option_lifetime
    
    N = number of periods
    
    r = risk-free interest rate
    
    payoff = callable function of 1/more parameters
    
    R = recovery rate of the bond. 
    
    lbd = default intensity of the bond. It is a callable
    """
    import numpy as np
    import math
    times = np.linspace(0,T,N+1,endpoint = True)
    import scipy.integrate as integ
    avg_lbds = [N/T * integ.quad(lbd,times[i],times[i+1])[0] for i in range(len(times)-1)]
    a = np.exp((r-q)*T/N)
    ups = [np.exp(np.sqrt((sigma**2-avg_lbds[i])*T/N)) for i in range(len(times)-1)]
    downs = [np.exp(-np.sqrt((sigma**2-avg_lbds[i])*T/N)) for i in range(len(times)-1)]
    p_ups = [(np.exp((r-q)*T/N)-downs[i]*np.exp(-avg_lbds[i]*T/N))/(ups[i]-downs[i]) \
             for i in range(len(times)-1)]
    p_downs = [(ups[i]*np.exp(-avg_lbds[i]*T/N)-a)/(ups[i]-downs[i]) for i in range(len(times)-1)]
    def comb(n,k):
        return math.factorial(n)/(math.factorial(k)*math.factorial(n-k))
    def bin_tree():
        tree = [[S0]]
        for i in range(1,N+1):
            tree.append([S0*ups[i-1]**(i-j)*downs[i-1]**j for j in range(0,i+1)])
        return tree
    binom_tree = bin_tree()[::-1]
    values = []
    for i in range(N+1):
        if i==0:
            values.append([max([x*no_shares,FV]) for x in binom_tree[i]])
        elif 1<=i and i<N:
            values.append([max([(values[-1][j]*p_ups[-i]+values[-1][j+1]*p_downs[-i] + \
                                 R*FV*(1-p_ups[-i]-p_downs[-i]))*np.exp(-r*T/N),\
                                no_shares * binom_tree[i][j]]) for j in range(len(values[-1])-1)])
        else:
            values.append([(p_ups[-1]*values[-1][0]+p_downs[-1]*values[-1][1]+\
                            (1-p_ups[-1]-p_downs[-1])*R*FV)*np.exp(-r*T/N)])
    return values[-1][0]

    
