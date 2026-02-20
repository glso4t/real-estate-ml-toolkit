from data_utils import load_or_create_csv
from linear_regression import compute_cost, gradient_descent, zscore_normalize_features
import numpy as np


def main():
    df = load_or_create_csv("data/houses.csv", m=200, seed=42)

    X = df[["size_m2", "bedrooms", "age"]].to_numpy()
    y = df["price"].to_numpy()
    
    #feature scaling    
    X_norm, mu, sigma = zscore_normalize_features(X)

    
    n = X_norm.shape[1]
    w = np.zeros(n)
    b = 0.0
    
    alpha = 0.1       
    num_iters = 2000
    lambda_ = 1.0
    
    #cost before training
    print("Initial cost:", compute_cost(X_norm, y, w, b, lambda_=lambda_))

    #training
    w, b, hist = gradient_descent(X_norm, y, w, b, alpha, num_iters, lambda_=lambda_)

    #cost after training
    print("Final cost:", compute_cost(X_norm, y, w, b, lambda_=lambda_))
    print("Learned w:", w)
    print("Learned b:", b)

    print("\nCost history:")
    for it, c in hist[:10]:
        print(f"iter {it}: cost {c}")

    if len(hist) > 10:
        it, c = hist[-1]
        print(f"... iter {it}: cost {c}")


if __name__ == "__main__":
    main()
