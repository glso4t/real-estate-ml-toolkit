import numpy as np
import pandas as pd

from linear_regression import predict


def main():
    # φορτώνω dataset από το .csv
    df = pd.read_csv("data/houses.csv")

    # παιρνω features κ target
    X = df[["size_m2", "bedrooms", "age"]].to_numpy()
    y = df["price"].to_numpy()

    # αρχικοποίηση w,b
    n = X.shape[1]
    w = np.zeros(n)
    b = 0.0

    # κάνω προβλεψεις
    y_hat = predict(X, w, b)

    print("First 5 predictions:", y_hat[:5])
    print("First 5 real prices:", y[:5])


if __name__ == "__main__":
    main()
