from data_utils import load_or_create_csv
from sklearn.linear_model import SGDRegressor, LogisticRegression
from linear_regression import (
    compute_cost,
    gradient_descent,
    zscore_normalize_features
)
from logistic_regression import (
    compute_cost_logistic_reg,
    gradient_descent as gradient_descent_logistic,
    predict_probability,
    predict_class
)
import numpy as np



def main():
    #loading data
    df = load_or_create_csv("data/houses.csv", m=200, seed=161)
    X = df[["size_m2", "bedrooms", "age"]].to_numpy()
    y = df["price"].to_numpy()
    

    #CLASSIFICATION: 1 = good deal, 0 = not worth it
    true_price = 2500*df["size_m2"] + 15000*df["bedrooms"] - 1200*df["age"] + 20000  # θεωρητικοί παράμετροι για ένα σπίτι
    y_class = (y < 0.75 * true_price).astype(int).to_numpy() #αν η πραγματική τιμή είναι 25% φθηνότερη από τη θεωρητική, τότε είναι ευκαιρία
    
    
    # training/testing split 80/20 (training=εκπαιδεύω τον αλγόριθμο πάνω στα data, testing= κρατάω το 20% hidden για να δώ αν κανει καλές προβλέψεις)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    y_train_class, y_test_class = y_class[:train_size], y_class[train_size:]
    
    
    # feature scaling    
    X_train_norm, mu, sigma = zscore_normalize_features(X_train)
    # εφαρμόζω (x-mu)/sigma του training και στο testing set
    X_test_norm = (X_test - mu) / sigma


    # LINEAR REGRESSION παράμετροι training
    print("\n--- Training Linear Regression ---")
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
    
    
    
    #LINEAR REGRESSION evaluation: κόστος w/o regularization για να βρω το καθαρό σφάλμα πρόβλεψης
    train_cost = compute_cost(X_train_norm, y_train, w_final, b_final, lambda_=0) #λ=0, μας ενδιαφέρει το κόστος χωρίς την ποινή στο testing
    test_cost = compute_cost(X_test_norm, y_test, w_final, b_final, lambda_=0)
    
    print("-" * 30)
    print("\n--- Linear Regression Results ---")
    print(f"Final Train Cost: {train_cost:,.2f}")
    print(f"Final Test Cost:  {test_cost:,.2f}")
    print("-" * 30)
    print(f"Learned Weights: {w_final}")
    print(f"Learned Bias:    {b_final:,.2f}")
    
    
    
    # LINEAR REGRESSION inference for random house (120m2, 3 υπνοδωμάτια, 10 ετών)
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
    
    
    
    
    # LOGISTIC REGRESSION παράμετροι training
    print("\n--- Training Logistic Regression ---")
    w_log = np.zeros(n)
    b_log = 0.0

    alpha_log = 0.1
    iters_log = 2000
    lambda_=1.0

    w_log, b_log, hist_log = gradient_descent_logistic(
        X_train_norm,
        y_train_class,
        w_log,
        b_log,
        alpha_log,
        iters_log,
        lambda_
    )
    
    
    #LOGISTIC REGRESSION evaluation
    y_pred_test = predict_class(X_test_norm, w_log, b_log)
    accuracy = np.mean(y_pred_test == y_test_class)

    print("\n--- Logistic Regression Results ---")

    print(f"Classifier accuracy: {accuracy:.3f}")

    
    #LOGISTIC REGRESSION inference
    prob = predict_probability(x_new_norm.reshape(1,-1), w_log, b_log)
    decision = predict_class(x_new_norm.reshape(1,-1), w_log, b_log)

    print("\nProbability good deal:", prob[0])

    if decision[0] == 1:
        print("Prediction: GOOD DEAL")
    else:
        print("Prediction: NOT WORTH IT")
        
    test_cost_log = compute_cost_logistic_reg(X_test_norm, y_test_class, w_log, b_log, lambda_=0)
    print(f"Logistic Test Cost: {test_cost_log:.4f}")

    
    
    
    
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
    
    # Scikit-learn Logistic Regression
    lr_model = LogisticRegression(penalty='l2', C=1.0) # C = 1/lambda
    lr_model.fit(X_train_norm, y_train_class)
    sk_log_acc = lr_model.score(X_test_norm, y_test_class)
    print(f"Sklearn Logistic Accuracy: {sk_log_acc:.3f}")
    
if __name__ == "__main__":
    main()
