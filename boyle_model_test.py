# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 02:21:02 2020

@author: Theodor
"""
from boyle_model import boyle_price
from binomial_models import price_CRR


#%%
def option_price_BS(S0,T,r,sigma,K,q=0,opt = "call",n = 0):
    import numpy as np 
    import scipy.stats as scp
    d1= (np.log(S0/K)+(r-q+sigma**2/2)*T)/(sigma*np.sqrt(T)) 
    d2 = d1-sigma*np.sqrt(T)
    def call_price(S0,T,r,sigma,K,q=0):
        return S0*np.exp(-q*T)*scp.norm.cdf(d1)-K*np.exp(-r*T)*scp.norm.cdf(d2)
    def put_price(S0,T,r,sigma,K,q=0):
        return K*np.exp(-r*T)*scp.norm.cdf(-d2)-S0*np.exp(-q*T)*scp.norm.cdf(-d1)
    if opt=="call":
        return call_price(S0,T,r,sigma,K,q)
    else:
        return put_price(S0,T,r,sigma,K,q)

def test_boyle_price():
    f = lambda S:max([S-90,0])
    import numpy as np
    S0,n,sig,T,lbd,r = 100,6,0.2,1,np.sqrt(3),0.05
    print('Boyle price',boyle_price(S0,f,n,sig,T,lbd,r))
    print("CRR model:",price_CRR(S0,sig,T,r,6,f)[0][0])
    print("BS price",option_price_BS(S0,T,r,sig,90))

def test_boyle_price2():
    import numpy as np
    S0,sig,T,lbd,r,K = 100,0.2,1,np.sqrt(3),0.05,90
    f = lambda S:max([S-K,0])
    prices_Boyle = [boyle_price(S0,f,i,sig,T,lbd,r) for i in range(1,21)]
    prices_CRR  = [price_CRR(S0,sig,T,r,i,f)[0][0] for i in range(1,21)]
    prices_BS  = [option_price_BS(S0,T,r,sig,K) for i in range(1,21)]
    import matplotlib.pyplot as plt
    plt.plot(range(1,21),prices_Boyle,'r-',label = 'Boyle')
    plt.plot(range(1,21),prices_CRR,'bo',label = 'CRR')
    plt.plot(range(1,21),prices_BS,'g',label = 'BS')
    plt.xlabel('No of periods')
    plt.ylabel('Option price')
    plt.title('Call option price with strike K = {0}'.format(90))
    plt.legend()
    plt.grid(True)
    plt.show()
#%%
def test_boyle_price3():
    import numpy as np
    S0,sig,T,lbd,r,K = 100,0.2,1,np.sqrt(3),0.05,110
    f = lambda S:max([K-S,0])
    prices_Boyle = [boyle_price(S0,f,i,sig,T,lbd,r) for i in range(1,21)]
    prices_CRR  = [price_CRR(S0,sig,T,r,i,f)[0][0] for i in range(1,21)]
    prices_BS  = [option_price_BS(S0,T,r,sig,K,opt = "put") for i in range(1,21)]
    import matplotlib.pyplot as plt
    plt.plot(range(1,21),prices_Boyle,'r-',label = 'Boyle')
    plt.plot(range(1,21),prices_CRR,'bo',label = 'CRR')
    plt.plot(range(1,21),prices_BS,'g',label = 'BS')
    plt.xlabel('No of periods')
    plt.ylabel('Option price')
    plt.title('Put option price with strike K = {0}'.format(K))
    plt.legend()
    plt.grid(True)
    plt.show()
test_boyle_price3()
#%%
def boyle_test2():
    import numpy as np
    S0,n,sig,T,lbd,r = 100,4,0.2,1,np.sqrt(3),0.05
    f = lambda S:max([S-100,0])
    print(boyle_price(S0,f,n,sig,T,lbd,r)[1])
boyle_test2()
#%%
from binomial_tree import binomial_model
from boyle_model import trinomial_option_price
def trinomial_test():
    S0,r,T = 100,0.1,1
    f = lambda S:max([S-100,0])
    mdl = binomial_model(S0,r,T,f)
    n,u,lbd = 4,1.2,1/3
    print(trinomial_option_price(S0,f,n,u,T,r,lbd))
    print(mdl.european_option_price(45,u)[0])
trinomial_test()
#%%