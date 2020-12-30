# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 19:56:20 2020

@author: Theodor
"""
import sys
sys.path.append(r'C:\Users\Theodor\Documents\python army of functions2\future_options')
from future_option import future_option
from binomial_tree import binomial_model,binomial_tree 
#%%
def CRR_tree(S0,sig,T,r,N,f=None):
    import numpy as np
    u = np.exp(sig*np.sqrt(T/N))
    return binomial_tree(S0,N,u,1/u,f)

def Tian_tree(S0,sig,T,r,N,f=None):
    import numpy as np
    R = np.exp(r*T/N)
    Q = np.exp(sig**2 * T/N)
    u = R*Q/2 * (Q+1+np.sqrt(Q**2+2*Q-3))
    d = R*Q/2 * (Q+1-  np.sqrt(Q**2+2*Q-3))
    return binomial_tree(S0,N,u,d,f)

def Jarrow_Rudd_tree(S0,sig,T,r,N,f=None):
    import numpy as np
    R = np.exp(r*T/N)
    u = R*(1+np.sqrt(np.exp(sig**2*T/N)-1))
    d = R*(1-np.sqrt(np.exp(sig**2*T/N)-1))
    return binomial_tree(S0,N,u,d,f)
    
def price_CRR(S0,sig,T,r,N,f):
    import numpy as np
    u = np.exp(sig*np.sqrt(T/N))
    bin_tree = binomial_model(S0,r,T,f)
    return bin_tree.european_option_price(N,u,1/u)

def price_Tian_model(S0,sig,T,r,N,f):
    import numpy as np
    R = np.exp(r*T/N)
    Q = np.exp(sig**2 * T/N)
    u = R*Q/2 * (Q+1+np.sqrt(Q**2+2*Q-3))
    d = R*Q/2 * (Q+1-  np.sqrt(Q**2+2*Q-3))
    p = (R-d)/(u-d)
    bin_tree = binomial_model(S0,r,T,f)
    return bin_tree.european_option_price(N,u,d,p)

def price_Jarrow_Rudd_model(S0,sig,T,r,N,f):
    import numpy as np
    R = np.exp(r*T/N)
    p=1/2
    u = R*(1+np.sqrt(np.exp(sig**2*T/N)-1))
    d = R*(1-np.sqrt(np.exp(sig**2*T/N)-1))
    bin_tree = binomial_model(S0,r,T,f)
    return bin_tree.european_option_price(N,u,d,p)

#%%
