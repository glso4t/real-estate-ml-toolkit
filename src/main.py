from data_utils import load_or_create_csv
from linear_regression import compute_cost, compute_gradient
import numpy as np


def main():
    df = load_or_create_csv("data/houses.csv", m=200, seed=42)
    print(df.head())
    
    X = df[["size_m2", "bedrooms", "age"]].to_numpy()
    y = df["price"].to_numpy()
    
    n = X.shape[1]
    w = np.zeros(n)
    b = 0.0

    
    cost_no_reg = compute_cost(X, y, w, b, lambda_=0.0)
    cost_reg = compute_cost(X, y, w, b, lambda_=1.0)

    print("Cost (no reg):", cost_no_reg)
    print("Cost (reg, lambda=1):", cost_reg)



if __name__ == "__main__":
    main()
