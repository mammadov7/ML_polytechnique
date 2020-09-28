import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
np.set_printoptions(precision=3)

# Load the data

data = np.loadtxt('data/data_train.csv', delimiter=',')

# Prepare the data

X = data[:,0:-1]
y = data[:,-1]
# 

# Inspect the data
# plt.figure()
# plt.hist(X[:,0], 10)
# plt.savefig("fig/histX1.pdf")

# plt.figure()
# plt.hist(X[:,1], 10)
# plt.savefig("fig/histX2.pdf")

# plt.figure()
# plt.hist(X[:,2], 10)
# plt.savefig("fig/histX3.pdf")

# # TODO 

# plt.figure()
# plt.plot(X[:,1],X[:,2], 'o')
# plt.xlabel('$x_2$')
# plt.ylabel('$x_3$')
# plt.savefig("fig/data.pdf")

# plt.figure()
# plt.plot(X[:,0],y, 'o')
# plt.xlabel('$x_2$')
# plt.ylabel('$x_3$')
# plt.savefig("fig/bef_dataX1toY.pdf")

# TODO 

# Standardization
m = np.mean(X,axis=0)
s = np.std(X,axis=0) 
X = (X - m)/s
# TODO 

# plt.figure()
# plt.plot(X[:,0],y, 'o')
# plt.xlabel('$x_2$')
# plt.ylabel('$x_3$')
# plt.savefig("fig/after_dataX1toY.pdf")
# plt.show()
# 

# Feature creation

def phi(X, degree):
    N,D = X.shape
    for d in range(2,degree+1):
        X = np.column_stack([X,X[:,0:D]**d])
    X = np.column_stack([np.ones(len(X)), X])
    return X

def MSE(y_test,y_pred):
  return np.mean( (y_test-y_pred)**2 )

# Polynomial degree
# degree = 2
mses=[]
mses1=[]
for degree in range(1,10):
  Z = phi(X,degree)

  # Estimating the coefficients
  # TODO 
  B = inv(Z.T@Z)@Z.T@y
  # Evaluation 
  # TODO 

  test = np.loadtxt('data/data_test.csv', delimiter=',')

  X_test = test[:,0:-1]
  y_test = test[:,-1]

  X_test = (X_test - m)/s


  Z_test = phi(X_test,degree)
  y_hat = Z_test@B
  y_hat1 = Z@B
  baseline = np.ones( y_test.size ) * np.mean(y)
  mses.append( MSE(y_test,y_hat)) 
  mses1.append(MSE(y,y_hat1))
  print( MSE(y_test,y_hat) )
  print( MSE(y,y_hat1) )
  print()

plt.plot(range(1,10),mses)
plt.plot(range(1,10),mses1)
plt.show()