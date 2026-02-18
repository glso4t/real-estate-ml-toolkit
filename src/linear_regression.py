import numpy as np


def predict_single(x, w, b):
    """
    Predict για 1 παράδειγμα (ένα σπίτι), θα το χρησιμοποιήσω στην κάτω συναρτηση

    x: (n,)  vector με features πχ [size_m2, bedrooms, age]
    w: (n,)  vector με weights
    b: scalar bias

    Επιστρέφει:
      y_hat: πρόβλεψη τιμής
    """

    # Υπολογίζω y_hat = w1*x1 + w2*x2 + ... + wn*xn + b
    
    y_hat = 0.0

    n = x.shape[0]

    for j in range(n):
        y_hat = y_hat + w[j] * x[j]

    y_hat = y_hat + b
    return y_hat


def predict(X, w, b):
    """
    Predict για ΠΟΛΛΑ παραδείγματα (πολλά σπιτια)

    X: (m, n) πίνακας features
       m = πόσα σπίτια
       n = πόσα features
    w: (n,)
    b: scalar

    Επιστρέφει:
      y_hat: (m,) προβλέψεις για όλα τα σπίτια
    """
    m = X.shape[0]
    y_hat = np.zeros(m)

    for i in range(m):
        y_hat[i] = predict_single(X[i], w, b)

    return y_hat
