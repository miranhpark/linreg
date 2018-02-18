import numpy as np

class SimpleLinReg:
    def __init__(self):
        self.b1 = None
        self.b0 = None

    def fit(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.x_mean = np.mean(self.x)
        self.y_mean = np.mean(self.y)
        self.b1 = np.sum((self.x - self.x_mean) * (self.y - self.y_mean)) / \
                    (np.sum((self.x - self.x_mean)**2))
        self.b0 = self.y_mean - self.b1 * self.x_mean
        
    @property
    def coef_(self):
        return np.array([self.b1])
    
    @property
    def intercept_(self):
        return self.b0

    def predict(self, x):
        return self.b1 * x + self.b0

    def score(self, x, y):
        u = np.sum((y - self.predict(x))**2)
        v = np.sum((y - np.mean(y))**2)
        return 1 - u/v

