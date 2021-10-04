import torch
import numpy as np

class LinearRBF():

    def __init__(self):
        self.sigma = 1
        self.l2_last = None
        self.K_last = None

    def pair_eval(self,X,Y):
        self.l2_last = ((X-Y)**2).sum(-1)
        self.K_last = torch.exp(-self.l2_last / (2*self.sigma))
        return self.K_last

    def set_med_dist(self,X,Y):
        self.sigma = torch.median(((X-Y)**2).sum(-1))

    def pair_gradXY(self,X,Y,reuse=False):
        if not reuse:
            self.pair_eval(X,Y)
        return -self.K_last.unsqueeze(-1)*(X-Y)/self.sigma

    def pair_gradYX(self,X,Y,reuse=False):
        return -self.pair_gradXY(X,Y,reuse)
    
    def pair_gradXYsum(self,X,Y,reuse=False):
        if not reuse:
            self.pair_eval(X,Y)
        return (self.K_last/self.sigma)*(X.shape[1]-(self.l2_last/self.sigma))

class RBF():

    def __init__(self):
        self.sigma = 1
        self.l2_last = None
        self.K_last = None

    def eval(self,X,Y):
        self.l2_last = torch.cdist(X,X)
        self.K_last = torch.exp(-self.l2_last / (2*self.sigma))
        return self.K_last

    def set_med_dist(self,X):
        self.sigma = torch.median(torch.cdist(X,X))

    def gradXY(self,X,Y,i,reuse=False):
        if not reuse:
            self.eval(X,Y)
        return -self.K_last*((X[:,[i]]-Y[:,[i]].transpose(0,1)))

    def gradYX(self,X,Y,i,reuse=False):
        return -self.gradXY(X,Y,i,reuse)

    def gradXYsum(self,X,Y,reuse=False):
        if not reuse:
            self.eval(X,Y)
        return (self.K_last/self.sigma)*(X.shape[1]-(self.l2_last/self.sigma))
