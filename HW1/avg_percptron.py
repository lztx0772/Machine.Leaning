import numpy as np
# 
# X = np.array([
#     [-2,4,-1],
#     [4,1,-1],
#     [1, 6, -1],
#     [2, 4, -1],
#     [6, 2, -1],

# ])
# 
#y = np.array([-1,-1,1,1,1])

def perceptron_avg(X, Y):
    w = np.zeros(len(X[0]))
    rate = 1
    epochs = 5

    for t in range(epochs):
        for i, x in enumerate(X):
            if (np.dot(X[i], w)*Y[i]) <= 0:
                w = w + rate*X[i]*Y[i]

    return w
    
w = perceptron_avg(X,y)
print(w)