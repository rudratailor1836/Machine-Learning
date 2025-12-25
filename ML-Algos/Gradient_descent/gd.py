class GDRegressor:
    
    def __init__(self,learning_rate, epochs):
        self.m = 100
        self.b = -120
        self.lr = learning_rate
        self.epochs = epochs 
        
    def fit(self,X,y):
        
        for i in range(self.epochs):
            loss_slope_intercept = -2*np.sum(y - self.m*X.ravel() - self.b)
            loss_slope_slope = -2*np.sum((y - self.m*X.ravel() - self.b)*X.ravel())
            
            self.b = self.b - (self.lr * loss_slope_intercept)
            self.m = self.m - (self.lr * loss_slope_slope)
        parameters = (self.m,self.b)
        return parameters
    
    
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt 
import numpy as np 
X,y = make_regression(n_samples=100,n_features=1, n_informative=1, n_targets=1, noise=20, random_state=13)

gd = GDRegressor(0.01,100)
print(gd.fit(X,y))
