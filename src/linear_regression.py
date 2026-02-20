import numpy as np
import math


def compute_cost(X, y, w, b, lambda_=0.0):
    """
    cost για Linear Regression με regularization

    """
    m = X.shape[0]

    cost_sum = 0.0
    for i in range(m):
        y_hat_i = np.dot(X[i], w) + b        # fwbx(i) = w·x(i) + b
        error_i = y_hat_i - y[i]
        cost_sum = cost_sum + (error_i ** 2)

    mse_cost = cost_sum / (2 * m)

    # regularization: (lambda_/(2m)) * Σ w_j^2
    reg_sum = 0.0
    for j in range(w.shape[0]):
        reg_sum = reg_sum + (w[j] ** 2)

    reg_cost = (lambda_ * reg_sum) / (2 * m)

    return mse_cost + reg_cost

def compute_gradient(X, y, w, b, lambda_=0.0):
    """
    gradient της Linear Regression με regularization

    """
    m, n = X.shape

    dj_dw = np.zeros(n)
    dj_db = 0.0

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]

        dj_db = dj_db + err

        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]

    dj_db = dj_db / m

    for j in range(n):
        dj_dw[j] = (dj_dw[j] / m) + (lambda_ / m) * w[j]

    return dj_dw, dj_db


def gradient_descent(X, y, w_in, b_in, alpha, num_iters, lambda_=0.0):
    """
    gradient descent για linear regression regularized

    σε κάθε επανάληψη dj_dw, dj_db = compute_gradient(...)
    και κάνουμε update
            w = w - alpha * dj_dw
            b = b - alpha * dj_db
    """
    w = w_in.copy()
    b = b_in

    J_history = []    # τιμές cost για να βλέπουμε ότι μειώνεται σταδιακά


    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b, lambda_=lambda_)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i % math.ceil(num_iters/10) == 0 or i == num_iters - 1:      #χωρίζουμε τις επαναλήψεις σε 10 parts και save cost 10 φορές συνολικά, κ την last iteration.
            cost = compute_cost(X, y, w, b, lambda_=lambda_)
            J_history.append((i, cost))

    return w, b, J_history


def zscore_normalize_features(X):
    """
    z-score σε κάθε feature του X

    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    sigma = np.where(sigma == 0, 1, sigma)    # αποφυγή διαίρεσης με 0

    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma