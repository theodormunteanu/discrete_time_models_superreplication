# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 07:43:10 2020

@author: Theodor
"""
from binomial_models import price_CRR,price_Tian_model,price_Jarrow_Rudd_model
def test_prices():
    S0,K,r,T,N = 100,100,0.05,1,5
    f = lambda S:max([S-K,0])
    import numpy as np
    lst = np.linspace(0.1,0.3,21) # list of scenarios for volatilities
    prices_CRR = [price_CRR(S0,sig,T,r,N,f) for sig in lst]
    prices_Tian = [price_Tian_model(S0,sig,T,r,N,f) for sig in lst]
    prices_JR = [price_Jarrow_Rudd_model(S0,sig,T,r,N,f) for sig in lst]
    import matplotlib.pyplot as plt
    plt.plot(lst,prices_CRR,'r-',label = 'CRR')
    plt.plot(lst,prices_Tian,'g-',label = 'Tian')
    plt.plot(lst,prices_JR,'b-',label = 'Jarrow-Rudd')
    plt.title('European call price when S = 100, K = 100,r = 5%, N = 5 periods \n depending on $\sigma$')
    plt.xlabel('$\sigma$')
    plt.ylabel('Call price')
    plt.grid(True)
    plt.legend()
    plt.show()
test_prices()
#%%
def test_prices2():
    S0,K,r,sig,T = 100,100,0.05,0.2,1
    f = lambda S:max([S-K,0])
    lst = range(1,21)
    prices_CRR = [price_CRR(S0,sig,T,r,N,f) for N in lst]
    prices_Tian = [price_Tian_model(S0,sig,T,r,N,f) for N in lst]
    prices_JR = [price_Jarrow_Rudd_model(S0,sig,T,r,N,f) for N in lst]
    import matplotlib.pyplot as plt
    plt.plot(lst,prices_CRR,'r-',label = 'CRR')
    plt.plot(lst,prices_Tian,'g-',label = 'Tian')
    plt.plot(lst,prices_JR,'b-',label = 'Jarrow-Rudd')
    plt.title('European call price when S = 100, K = 100,r = 5% \n depending \
              on the number of periods')
    plt.xlabel('No of periods')
    plt.ylabel('Call price')
    plt.grid(True)
    plt.legend()
    plt.show()
test_prices2()
#%%