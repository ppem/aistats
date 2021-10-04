import torch
from torch.nn import Module
import torch.autograd
from kernel import LinearRBF
import numpy as np

class LinearKSD():

    def __init__(self, E, k):
        self.E = E
        self.k = k

    def score(self,X):
        Ex = -self.E(X)
        S = torch.autograd.grad(Ex,X,grad_outputs=torch.ones_like(Ex),retain_graph=True)[0]
        return S

    def test_stat(self,X):
        n = int(X.shape[0]/2)
        X1 = X[:n,:]
        X2 = X[n:int(2*n)]
        s1 = self.score(X1)
        s2 = self.score(X2)
        self.k.set_med_dist(X1,X2)
        K = self.k.pair_eval(X1,X2)
        A = (s1*s2).sum(-1)*K
        B = (s2*self.k.pair_gradXY(X1,X2,True)).sum(-1)
        C = (s1*self.k.pair_gradYX(X1,X2,True)).sum(-1)
        D = self.k.pair_gradXYsum(X1,X1,True)
        H = A+B+C+D
        stat = (H.mean() / torch.sqrt((H**2).mean()))**2
        return stat



class KSD():

    def __init__(self, E, k, device='cpu'):
        self.E = E
        self.k = k
        self.device = device

    def bootstrapper_rademacher(self,n):
        return 2.0*torch.randint(0, 2, (n,1)).to(self.device)-1

    def score(self,X):
        Ex = self.E(X)
        S = torch.autograd.grad(Ex,X,grad_outputs=torch.ones_like(Ex),retain_graph=True)[0]
        return S

    def test_stat(self,X,set_bw=True):
        n = X.shape[0]
        d = X.shape[1]
        s = self.score(X)
        s_gram = s.mm(s.transpose(0,1)) 
        if set_bw:
            self.k.set_med_dist(X)
        K = self.k.eval(X,X)

        B = torch.zeros([n,n]).to(self.device)
        C = torch.zeros([n,n]).to(self.device)
        for i in range(d):
            B += self.k.gradXY(X,X,i,True)*s[:,[i]]
            C += self.k.gradYX(X,X,i,True).transpose(0,1)*(s[:,[i]].transpose(0,1))

        H = K*s_gram + B + C + self.k.gradXYsum(X,X,True).transpose(0,1)

        stat = H.mean()

        return stat, H

    def perform_test(self,X,nboot=100):
        n=X.shape[0]
        stat,H = self.test_stat(X)

        boots = torch.zeros(nboot).to(self.device)
        for i in range(nboot):
            W = self.bootstrapper_rademacher(n)
            boots[i]=W.transpose(0,1).mm(H.mm(W/100)).squeeze()

        pval = (boots>(n*stat)).sum().float()/nboot
        return stat, pval
