# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 06:49:06 2020

@author: Theodor
"""

from binomial_models import price_CRR,price_Tian_model,price_Jarrow_Rudd_model
from binomial_models import CRR_tree, Tian_tree,Jarrow_Rudd_tree
#%%
def test_price():
    S0,sig,T,r,N = 309,0.14,1/2,0,2
    f = lambda S:max([S-310,0])
    print("CRR model:",price_CRR(S0,sig,T,r,N,f))
    print("Tian model:",price_Tian_model(S0,sig,T,r,N,f))
    print("Jarrow-Rudd model:",price_Jarrow_Rudd_model(S0,sig,T,r,N,f))
    print("CRR tree",CRR_tree(S0,sig,T,r,N))
    print("Tian tree:",Tian_tree(S0,sig,T,r,N))
    print("Jarrow-Rudd tree:",Jarrow_Rudd_tree(S0,sig,T,r,N))
test_price()
#%%
def test_price2():
    S0,sig,T,r,N = 309,0.14,1/2,0,3
    f = lambda S:max([S-310,0])
    tree_asset_CRR = CRR_tree(S0,sig,T,r,N)
    tree_asset_JR = Jarrow_Rudd_tree(S0,sig,T,r,N)
    tree_asset_Tian = Tian_tree(S0,sig,T,r,N)
    tree_option_JR = price_Jarrow_Rudd_model(S0,sig,T,r,N,f)
    tree_option_CRR = price_CRR(S0,sig,T,r,N,f)
    tree_option_Tian = price_Tian_model(S0,sig,T,r,N,f)
    delta_JR = [[(tree_option_JR[i][j]-tree_option_JR[i][j-1])/(tree_asset_JR[i][j]-\
             tree_asset_JR[i][j-1]) for j in range(1,i+1)] for i in range(1,len(tree_option_JR))]
    delta_Tian = [[(tree_option_Tian[i][j]-tree_option_Tian[i][j-1])/(tree_asset_Tian[i][j]-\
             tree_asset_Tian[i][j-1]) for j in range(1,i+1)] for i in range(1,len(tree_option_Tian))]
    delta_CRR = [[(tree_option_CRR[i][j]-tree_option_CRR[i][j-1])/(tree_asset_CRR[i][j]-\
             tree_asset_CRR[i][j-1]) for j in range(1,i+1)] for i in range(1,len(tree_option_CRR))]
    print(delta_Tian)
    print("CRR tree:",tree_asset_CRR)
    print("option CRR tree",tree_option_CRR)
    print(tree_option_CRR[0][0],tree_option_Tian[0][0],tree_option_JR[0][0])
test_price2()
#%%
def trinomial(n,p1,p2,p3,m):
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

def trin_term_dist(S0,u,n,p1,p2,p3):
    return [S0*u**i for i in range(n,-n-1,-1)],[trinomial(n,p1,p2,p3,i) for i in \
            range(n,-n-1,-1)]
    
def test_trin():
    S0,u,n = 1,1.5,3
    #print(sum(trin_term_dist(S0,u,n,1/3,1/3,1/3)[1]))
    print([trinomial(3,1/3,1/3,1/3,k) for k in range(-3,4)])
    print(sum([trinomial(3,1/3,1/3,1/3,k) for k in range(-3,4)]))

test_trin()
#%%
def test_trinomial():
    n,p1,p2,p3 = 2,1/3,1/3,1/3
    print(trinomial(n,p1,p2,p3,-2))
    print(trinomial(n,p1,p2,p3,-1))
    print(trinomial(n,p1,p2,p3,0))
test_trinomial()
#%%
def test_sum():
    a = [[1,2,3],[4,5,6]]
    print([[x**2 for x in a[i]] for i in range(len(a))])
    print(sum([sum(a[i]) for i in range(len(a))]))
    y = [11,12,90,12,10,13]
    print([y[i] for i in range(len(y)) if i%2==0 and i<3 and i>1])
test_sum()
#%%
def test_optimisation():
    import scipy.optimize as opt
    fun = lambda x:x[0]+x[1]-x[0]*x[1]
    bnds = ((0,1/2),(0,1.2))
    print(opt.minimize(fun,(1/2,1/2),method = 'SLSQP',bounds = bnds))
test_optimisation()
#%%
def test_optimisation2():
    import scipy.optimize as opt
    fun = lambda x:(-5*x[0]-5*x[1]-5*x[0]*x[1]+6*x[0]*x[2])
    bnds = ((0,1/2),(0,1/4),(0,1/2))
    print(opt.minimize(fun,(0,0,0),method = 'SLSQP',bounds = bnds))
test_optimisation2()
#%%
def test_optimisation3():
    import scipy.optimize as opt
    fun = lambda x:-3*(1/4+x[0]/2)*(2/3+x[1]/2)-3/5*(1/4+x[0]/2)*(1/3-2*x[2])-16/25*(3/4-\
                     3*x[0]/2)*(2/3+x[1])
    bnds = ((0,1/2),(0,1/6),(0,1/2))
    print(opt.minimize(fun,(0,0,0),method = 'SLSQP',bounds = bnds))
test_optimisation3()
