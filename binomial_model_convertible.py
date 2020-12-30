# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:43:55 2020

@author: Theodor
"""
import sys
sys.path.append(r'C:\Users\Theodor\Documents\python_libs\credit')
sys.path.append(r'C:\Users\Theodor\Documents\python_libs\american_options')
sys.path.append(r'C:\Users\Theodor\Documents\python_libs\binomial_model')
from debt_problem import recovery_rate,surv
from piecewise_expo import piecewise_exponential
from american_pricer_lst_sqr2 import american_price_ls3_alt
from binomial_model import binomial_price_amer
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


#%%
def conv_bond_test2():
    A0,sigma,r,FV,T = 12.1773 * 10**9,0.0911,0.02,10**10,4
    MktCap,ns,sig_E,N = 3*10**9,6*10**6,0.35,20
    S0 = MktCap/ns
    import numpy as np
    g = surv(A0,r,sigma,FV,T,r,FV)
    times3 = np.linspace(0,4,9,endpoint = True)
    survivals4 = [g(x) for x in times3]
    lbds3 = [2*(np.log(survivals4[i]/survivals4[i+1])) for i in range(len(survivals4)-1)]
    lbd = lambda t:sum([lbds3[i]*(times3[i]<t and t<=times3[i+1]) for i in range(len(lbds3))])
    R = recovery_rate(A0,r,sigma,FV,T,r)
    print(binomial_price_amer_conv2(S0,sig_E,T,r,N,lbd,ns,R,FV))
conv_bond_test2()
#%%
def test_binomial_price_amer_conv():
    S0,sigma,T,r,N,lbd,call_value,no_shares = 50,0.3,0.75,0.05,3,0.01,113,2
    print(binomial_price_amer_conv_call(S0,sigma,T,r,N,lbd,call_value,no_shares))
    print(binomial_price_amer_conv(S0,sigma,T,r,N,lbd,no_shares))
    
test_binomial_price_amer_conv()
#%%

r"""
I borrow from the Weibull calibration and Matlab example here: """

def conv_bond_test():
    A0,sigma,r,FV,T = 12.1773 * 10**9,0.0911,0.02,10**10,4
    import numpy as np
    g = surv(A0,r,sigma,FV,T,r,FV)
    times = np.linspace(0,4,101,endpoint = True)
    times2 = np.linspace(0,4,17,endpoint = True)
    times3 = np.linspace(0,4,9,endpoint = True)
    times4 = np.linspace(0,4,49,endpoint = True)
    survivals1 = [g(x) for x in times]
    survivals2 = [g(x) for x in [0,1,2,3,4]]
    survivals3 = [g(x) for x in times2]
    survivals4 = [g(x) for x in times3]
    survivals5 = [g(x) for x in times4]
    lbd = np.dot([-np.log(survivals1[i]) for i in range(1,len(times))],\
                          times[1::])/np.dot(times[1::],times[1::])
    lbds = [np.log(survivals2[i]/survivals2[i+1]) for i in range(len(survivals2)-1)]
    lbds2 = [4*(np.log(survivals3[i]/survivals3[i+1])) for i in range(len(survivals3)-1)]
    lbds3 = [2*(np.log(survivals4[i]/survivals4[i+1])) for i in range(len(survivals4)-1)]
    lbds4 = [12 * np.log(survivals5[i]/survivals5[i+1]) for i in range(len(survivals5)-1)]
    print("the survival probabilities at every 3 months is given by:",survivals2)
    """
    We calibrate the Weibull distribution.
    
    ys = the resulting log(-log) survivals, the explained variables. 
    
    xs = the log of times, the explanatory variables. 
    
    a,b = least squares coefficients
    
    lbd2,k are the Weibull resulting parameters.
    """
    ys = [np.log(-np.log(x)) for x in survivals1[1::]]
    xs = [np.log(x) for x in times[1::]]
    n = len(ys)
    a = (n*np.dot(xs,ys)-sum(xs)*sum(ys))/(n*np.dot(xs,xs)-(sum(xs))**2)
    b = (np.dot(xs,xs)*sum(ys)-sum(xs)*np.dot(xs,ys))/(n*np.dot(xs,xs)-sum(xs)**2)
    lbd2 = np.exp(-b/a)
    k = a
    """
    We need the survival and density functions of Weibull distribution. 
    """
    
    def survival(lbd,k,u):
        return np.exp(-(u/lbd)**k)
    def density(lbd,k,u):
        np.exp(-(u/lbd)**k) * (u/lbd)**(k-1) * k/lbd
    f = lambda u:np.exp(-(u/lbd2)**k)*(u/lbd2)**(k-1) * k/lbd2 #Weibull dens.
    fp = lambda u:f(u)*np.exp(-r*u)#integrator: Weibull density+discount fact
    import scipy.integrate as integ
    R = recovery_rate(A0,r,sigma,FV,T,r,FV)
    price = FV*np.exp(-r*T)*survival(lbd2,k,T)+integ.quad(fp,0,T)[0]*FV*R
    price2 = FV*np.exp(-(r+lbd)*T)+R*FV*(1-np.exp(-(r+lbd)*T))/(r+lbd)*lbd
    """
    Now we pass also to the piecewise exponential models. \\
    """
    obj = piecewise_exponential([1,2,3],lbds)
    obj2 = piecewise_exponential(times2[1:-1],lbds2)
    obj3 = piecewise_exponential(times3[1:-1],lbds3)
    obj4 = piecewise_exponential(times4[1:-1],lbds4)
    f2 = lambda u:np.exp(-r*u)*obj.pdf2(u) #the density for exponential r.v.
    f3 = lambda u:np.exp(-r*u)*obj2.pdf2(u)
    f4 = lambda u:np.exp(-r*u)*obj3.pdf2(u)
    f5 = lambda u:np.exp(-r*u)*obj4.pdf2(u)
    price3 = FV*np.exp(-r*T)*np.exp(-sum(lbds)) + R * FV * integ.quad(f2,0,T)[0]
    price4 = FV*np.exp(-r*T)*np.exp(-sum(lbds2)/2) + R*FV*integ.quad(f3,0,T)[0]
    price5 = FV*np.exp(-r*T)*np.exp(-sum(lbds3)/2) + R*FV*integ.quad(f4,0,T)[0]
    price6 = FV*np.exp(-r*T)*np.exp(-sum(lbds4)/2) + R*FV*integ.quad(f5,0,T)[0]
    print("The exponential model parameter is",lbd)
    print("The parameters lambda and k of Weibull distribution are",lbd2,k)
    print("The bond price under Weibull distribution is ",price)
    print("The bond price under exponential distribution is ",price2)
    print("Price under piecewise exponential with 3 knots at years 1,2,3 is:",price3)
    print("Price under piecewise exponential with knots at every 3 months is:",price4)
    print("Price under piecewise exponential with knots at every 6 months is ",price5)
    print("Price under piecewise exponential with knots at every month is ",price6)
    payoff = lambda x:max([FV-x,0])
    payoff2 = lambda x:min([FV,x])
    regressors = [lambda x:1,lambda x:x,lambda x:x**2]
    print("The american price with least squares approach is: ",\
          FV*np.exp(-r*T) - american_price_ls3_alt(A0,r,sigma,T,payoff,regressors,r,4,80000))
    print("The american price with binomial approach is:",\
          FV*np.exp(-r*T) - binomial_price_amer(A0,sigma,T,payoff,r,100)[-1][0])
    print("The american price with least squares approach is, using min payoff: ",\
          american_price_ls3_alt(A0,r,sigma,T,payoff2,regressors,r,4,80000))
    print("The american price with binomial approach is, using min payoff:",\
          binomial_price_amer(A0,sigma,T,payoff2,r,100)[-1][0])
    
conv_bond_test()
#%%
def conv_bond_test2bis():
    A0,sigma,r,FV,T = 12.1773 * 10**9,0.0911,0.02,10**10,4
    MktCap,ns,sig_E,N = 3*10**9,6*10**6,0.35,20
    S0 = MktCap/ns
    import numpy as np
    g = surv(A0,r,sigma,FV,T,r,FV)
    times = np.linspace(0,4,101,endpoint = True)
    times2 = np.linspace(0,4,17,endpoint = True)
    times3 = np.linspace(0,4,9,endpoint = True)
    times4 = np.linspace(0,4,49,endpoint = True)
    survivals1 = [g(x) for x in times]
    survivals2 = [g(x) for x in [0,1,2,3,4]]
    survivals3 = [g(x) for x in times2]
    survivals4 = [g(x) for x in times3]
    survivals5 = [g(x) for x in times4]
    lbd = np.dot([-np.log(survivals1[i]) for i in range(1,len(times))],times[1::])/np.dot(times[1::],times[1::])
    lbds = [np.log(survivals2[i]/survivals2[i+1]) for i in range(len(survivals2)-1)]
    lbds2 = [4*(np.log(survivals3[i]/survivals3[i+1])) for i in range(len(survivals3)-1)]
    lbds3 = [2*(np.log(survivals4[i]/survivals4[i+1])) for i in range(len(survivals4)-1)]
    lbds4 = [12 * np.log(survivals5[i]/survivals5[i+1]) for i in range(len(survivals5)-1)]
    obj = piecewise_exponential([1,2,3],lbds)
    obj2 = piecewise_exponential(times2[1:-1],lbds2)
    obj3 = piecewise_exponential(times3[1:-1],lbds3)
    obj4 = piecewise_exponential(times4[1:-1],lbds4)
    """
    We calibrate the Weibull distribution.
    
    ys = the resulting log(-log) survivals, the explained variables. 
    
    xs = the log of times, the explanatory variables. 
    
    a,b = least squares coefficients
    
    lbd2,k are the Weibull resulting parameters.
    """
    ys = [np.log(-np.log(x)) for x in survivals1[1::]]
    xs = [np.log(x) for x in times[1::]]
    n = len(ys)
    a = (n*np.dot(xs,ys)-sum(xs)*sum(ys))/(n*np.dot(xs,xs)-(sum(xs))**2)
    b = (np.dot(xs,xs)*sum(ys)-sum(xs)*np.dot(xs,ys))/(n*np.dot(xs,xs)-sum(xs)**2)
    lbd2,k = np.exp(-b/a),a
    print(len(ys))
    print(lbd2,k)
    """
    We need the survival and density functions of Weibull distribution. 
    """
    def surv_Weibull(lbd,k,u):
        return np.exp(-(u/lbd)**k)
    def dens_Weibull(lbd,k,u):
        np.exp(-(u/lbd)**k) * (u/lbd)**(k-1) * k/lbd
    err = sum([(g(x)-np.exp(-lbd*x))**2 for x in times[1::]])/100
    errs = sum([(g(x)-obj.survival2(x))**2 for x in times[1::]])/100
    errs2 = sum([(g(x)-obj2.survival2(x))**2 for x in times[1::]])/100
    errs3 = sum([(g(x)-obj3.survival2(x))**2 for x in times[1::]])/100
    errs4 = sum([(g(x)-obj4.survival2(x))**2 for x in times[1::]])/100
    err2 = sum([(g(x)-surv_Weibull(lbd2,k,x))**2 for x in times[1::]])/100
    indexes = ['Exponential','Piecewise Exponential: \n 4 knots','PE - quartely',\
               'PE: semi-annually','PE-monthly','Weibull']
    import pandas as pd
    data = [err,errs,errs2,errs3,errs4,err2]
    cols = ['mean square error']
    print(pd.DataFrame(data,index = indexes,columns = cols))
    lbd = lambda t:sum([lbds4[i]*(times4[i]<t and t<=times4[i+1]) for i in range(len(lbds4))])
    R = recovery_rate(A0,r,sigma,FV,T,r)
    payoff = lambda x:max([FV-x,0])
    print("With conversion, the american price with binomial approach is:",\
          binomial_price_amer_conv2(S0,sig_E,T,r,N,lbd,ns,R,FV))
    print("Without conversion, the american price with binomial approach is:",\
          FV*np.exp(-r*T) - binomial_price_amer(A0,sigma,T,payoff,r,N)[-1][0])
conv_bond_test2bis()
#%%
    