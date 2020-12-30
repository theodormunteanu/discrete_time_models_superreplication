# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 01:47:04 2020

@author: Theodor
"""

def trinomial(n,p1,p2,p3,m):
    r"""
    Inputs:
    ---------
    n = number of periods
    p1,p2,p3 = the down, middle and up probabilities
    m = the value between -n and + n
    
    Output:
    ---------
    P($X_1+X_2+...+X_n = m$)
    """
    from functools import reduce
    a = max([0,m])
    b = int((m+n)/2)
    x = p1 * p2**(-2) *p3
    def factorial(k):
        if k==0:
            return 1
        else: 
            return reduce((lambda x,y:x*y),range(1,k+1))
    return sum([x**k * factorial(n)/(factorial(k-m)*factorial(n+m-2*k)*\
                                 factorial(k)) for k in range(a,b+1)])*\
                                 (p2/p1)**m * p2**n 


#%%
def boyle_price(S0,f,n,sig,T,lbd,r):
    r"""
    Inputs:
    --------------
    S0: the current underlying asset price
    n:  number of periods
    sig: the free parameter
    T: the option lifetime
    r: risk-free interest rate
    """
    import numpy as np
    R = np.exp(r*T/n)
    W = np.exp((2*r+sig**2)*T/n)
    u = np.exp(sig*lbd*np.sqrt(T/n))
    p3 = ((W-R)*u-(R-1))/((u-1)*(u**2-1))
    p1 = ((W-R)*u**2-(R-1)*u**3)/((u-1)*(u**2-1))
    p2 = 1-p1-p3
    term_values = [S0*u**i for i in range(n,-n-1,-1)]
    probs = [trinomial(n,p1,p2,p3,i) for i in range(n,-n-1,-1)]
    return np.exp(-r*T)*sum([f(term_values[i])*probs[i] for i in range(len(term_values))])

#%%
def trinomial_option_price(S0,f,n,u,T,r,lbd):
    import numpy as np
    if lbd<=0 or lbd>=1:
        raise ValueError('lbd is the probability of going up so it must be in [0,1]')
    p1 = (lbd*(u-1)-r)/(1-1/u)
    p2 = (1+r-1/u-lbd*(u-1/u))/(1-1/u)
    term_values = [S0*u**i for i in range(n,-n-1,-1)]
    probs = [trinomial(n,p1,p2,lbd,i) for i in range(n,-n-1,-1)]
    return np.exp(-r*T)*sum([f(term_values[i])*probs[i] for i in range(len(term_values))])


