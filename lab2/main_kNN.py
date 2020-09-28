from numpy import *
import matplotlib.pyplot as plt
from numpy.linalg import norm

def kNN(k, X, y, x):
    '''
    kNN classification of x
    -----------------------
        Input: 
        k: number of nearest neighbors
        X: training data           
        y: class labels of training data
        x: test instance

        return the label to be associated with x

        Hint: you may use the function 'norm' 
    '''

    m,n = X.shape

    # Take the distances
    distances = norm(X - x, axis=1)
        
    # Sort distances and re-arrange labels based on the distance of the instances
    idx = distances.argsort()
    labels = labels[idx]
    
    c = zeros(max(labels)+1)
    
    # Compute the class labels of the k nearest neigbors
    for i in range(k):
        c[labels[i]] += 1

    # Return the label with the largest number of appearances
    label = argmax(c)
    
    return label

