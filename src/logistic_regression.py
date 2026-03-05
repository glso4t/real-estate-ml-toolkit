import numpy as np
import copy, math

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost_logistic_reg(X, y, w, b, lambda_=0.0):
    m, n = X.shape
    cost = 0.0
    for i in range(m):
        z = np.dot(X[i], w) + b
        y_hat = sigmoid(z)
        cost += -y[i]*np.log(y_hat) - (1-y[i])*np.log(1-y_hat)
    
    cost = cost / m
    reg_cost = 0.0    # Προσθήκη του Regularization Cost
    for j in range(n):
        reg_cost += (w[j]**2)
    reg_cost = (lambda_ / (2 * m)) * reg_cost
    
    return cost + reg_cost


def compute_gradient_logistic_reg(X, y, w, b, lambda_=0.0):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.0
    
    for i in range(m):
        z = np.dot(X[i], w) + b
        y_hat = sigmoid(z)
        error = y_hat - y[i]
        
        dj_db += error
        for j in range(n):
            dj_dw[j] += error * X[i, j]
            
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_ / m) * w[j]
        
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, alpha, num_iters, lambda_=0.0): 
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistic_reg(X, y, w, b, lambda_)   

        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        if i<100000:
            J_history.append(compute_cost_logistic_reg(X, y, w, b, lambda_))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
        
    return w, b, J_history



def predict_probability(X, w, b):
    z = np.dot(X, w) + b
    p = sigmoid(z)
    
    return p

def predict_class(X, w, b):
    p = predict_probability(X, w, b)
    y_pred = (p >= 0.5).astype(int)
    
    return y_pred