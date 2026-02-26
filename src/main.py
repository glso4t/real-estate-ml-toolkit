from data_utils import load_or_create_csv
from linear_regression import compute_cost, gradient_descent, zscore_normalize_features
import numpy as np


def main():
    #loading data
    df = load_or_create_csv("data/houses.csv", m=200, seed=161)
    X = df[["size_m2", "bedrooms", "age"]].to_numpy()
    y = df["price"].to_numpy()
    
    # training/testing split 80/20 (training=εκπαιδεύω τον αλγόριθμο πάνω στα data, testing= κρατάω το 20% hidden για να δώ αν κανει καλές προβλέψεις)
    train_size = int(0.8 * len(X))
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # feature scaling    
    X_train_norm, mu, sigma = zscore_normalize_features(X_train)
    
    # εφαρμόζω (x-mu)/sigma του training και στο testing set
    X_test_norm = (X_test - mu) / sigma

    # παράμετροι training
    n = X_train_norm.shape[1]
    w_init = np.zeros(n)
    b_init = 0.0
    
    alpha = 0.1       
    iters = 2000
    lambda_ = 1.0 #regularization
    
    # Training
    print(f"Training on {len(X_train)} samples...")
    w_final, b_final, hist = gradient_descent(
        X_train_norm, y_train, w_init, b_init, alpha, iters, lambda_
    )
    
    # evaluation: κόστος w/o regularization για να βρω το καθαρό σφάλμα πρόβλεψης
    train_cost = compute_cost(X_train_norm, y_train, w_final, b_final, lambda_=0) #λ=0, μας ενδιαφέρει το κόστος χωρίς την ποινή στο testing
    test_cost = compute_cost(X_test_norm, y_test, w_final, b_final, lambda_=0)
    
    
    print("-" * 30)
    print(f"Final Train Cost: {train_cost:,.2f}")
    print(f"Final Test Cost:  {test_cost:,.2f}")
    print("-" * 30)
    print(f"Learned Weights: {w_final}")
    print(f"Learned Bias:    {b_final:,.2f}")
    
if __name__ == "__main__":
    main()
