import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    ''' sigmoid fn '''
  g = 1./(1. + np.exp(-z))
  g = np.clip(g,0.001,0.999)
  return 1/(1 + np.exp(-z))


def cost(w, X, y): 
    ''' Computes the cost using w as the parameters for logistic regression. '''
    E = 0
    N = X.shape[0]
    for i in range(N):
      E = E + (-(1-y[i])*np.log( (1-sigmoid(np.dot(W.T,y[i]))) )- y[i]*np.log(sigmoid(np.dot(W.T,y[i])))) 
    E /= N
    return E
    
def compute_grad(w, X, y):
    ''' Computes the gradient of the cost with respect to the parameters. '''
  dE = np.zeros_like(w) # initialize gradient
  dE = X.T@(sigmoid(np.dot(W.T,y))-y)) 
  return dE

def predict(w, X):
    # Predict whether each label is 0 or 1 using learned logistic regression parameters w. The threshold is set at 0.5

    N = X.shape[0] # number of examples
    yp = np.zeros(N) # predicted classes of examples
    
    # TODO <! --

    
    for i in range(N):
        yp[i] = (sigmoid(np.dot(X[i],w)) > 0.5) * 1
            
    # -->
    return yp



#======================================================================
# Load the dataset
data = np.loadtxt('./data/data.csv', delimiter=',')
 
#Add intercept term to 
data_1 = np.ones((data.shape[0], 4))
data_1[:, 1:] = data

# Standardize the data N.B. This line was missing, but it's quite important:
# (It will still work without standardization, but may behave eratically)
data_1[:,1:3] = (data_1[:,1:3] - np.mean(data_1[:,1:3],axis=0)) / np.std(data_1[:,1:3],axis=0)

n_test = 20
X = data_1[:-n_test, 0:3]
y = data_1[:-n_test, -1]
X_test = data_1[-n_test:, 0:3]
y_test = data_1[-n_test:, -1]


# Plot data 
pos = np.where(y == 1) # instances of class 1
neg = np.where(y == 0) # instances of class 0
plt.scatter(X[pos, 1], X[pos, 2], marker='o', c='b')
plt.scatter(X[neg, 1], X[neg, 2], marker='x', c='r')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Admitted', 'Not Admitted'])
plt.show()


N = X.shape[0]

# Initialize fitting parameters 
w = np.random.randn(3,1) * 0.05

# Gradient Descent
# TODO  
  
# Plot the decision boundary
plt.figure()
plot_x = np.array([min(X[:, 1]), max(X[:, 2])])
plot_y = (- 1.0 / w[2,0]) * (w[1,0] * plot_x + w[0,0])
plt.plot(plot_x, plot_y)
plt.scatter(X[pos, 1], X[pos, 2], marker='o', c='b')
plt.scatter(X[neg, 1], X[neg, 2], marker='x', c='r')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Decision Boundary', 'Admitted', 'Not Admitted'])
plt.show()

# Compute accuracy on the training set
accuracy = np.mean(predict(w, X) - y)
print("\nAccuracy: %4.3f" % accuracy)

# TODO Plot error curves 
