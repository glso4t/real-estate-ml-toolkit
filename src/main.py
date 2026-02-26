from data_utils import load_or_create_csv
from sklearn.linear_model import SGDRegressor
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
    
    # inference for random house (120m2, 3 υπνοδωμάτια, 10 ετών)
    x_new = np.array([120, 3, 10])
    
    # normalization (χρήση mu&sigma από training set)
    x_new_norm = (x_new - mu) / sigma
    
    prediction = np.dot(x_new_norm, w_final) + b_final
    print(f"\nPredicted price for 120m2, 3br, 10y: {prediction:,.2f}€")

    # σφάλμα σε ευρω (δεν παίρνω το τετράγωνο για το κόστος, αλλά abs)
    y_test_pred = np.dot(X_test_norm, w_final) + b_final
    errors = np.abs(y_test_pred - y_test)
    mae = np.mean(errors)
    
    print(f"Average error in test set: {mae:,.2f}€")
    
    
    
    # συγκρίνω με scikit
    sk_alpha = lambda_ / len(X_train)
    sgdr = SGDRegressor(max_iter=2000, alpha=sk_alpha, penalty='l2')
    sgdr.fit(X_train_norm, y_train)

    print("\n--- Scikit-Learn Comparison ---")
    print(f"Sklearn Weights: {sgdr.coef_}")
    print(f"Sklearn Bias:    {sgdr.intercept_[0]:,.2f}")
    
    # Σύγκριση προβλέψεων στο ίδιο σπίτι
    sk_prediction = sgdr.predict(x_new_norm.reshape(1, -1))
    print(f"Sklearn Prediction: {sk_prediction[0]:,.2f}€")
    print(f"Difference: {abs(prediction - sk_prediction[0]):,.2f}€")
    
if __name__ == "__main__":
    main()
