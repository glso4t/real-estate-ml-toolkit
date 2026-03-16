from data_utils import load_or_create_csv
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
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

class CustomLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    όταν καλώ .fit() καλείται ο gradient_descent_logistic και 
    όταν καλώ .predict() καλείται η predict_class() που έφτιαξα
    για να καταλαβαίνει η scikit και να χρησιμοποιηθει το cross_val_score της βιβλιοθήκης
    """
    def __init__(self, alpha=0.1, iters=2000, lambda_=1.0, flag_for_prints=False):
        self.alpha = alpha
        self.iters = iters
        self.lambda_ = lambda_
        self.flag_for_prints = flag_for_prints
        self.w_ = None
        self.b_ = None

    def fit(self, X, y):
        n = X.shape[1]
        w_init = np.zeros(n)
        b_init = 0.0
        
        self.w_, self.b_, _ = gradient_descent_logistic(
            X, y, w_init, b_init, self.alpha, self.iters, self.lambda_, flag_for_prints=self.flag_for_prints
        )
        return self

    def predict(self, X):
        return predict_class(X, self.w_, self.b_)





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



    # =======LINEAR REGRESSION παράμετροι training=======
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



    #=======LOGISTIC REGRESSION παράμετροι training=======
    print("\n--- Training Logistic Regression ---")
    
    """
    Εδώ τρέχω την gradient_descent_logistic "γυμνή":
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
    """
    #χρησιμοποιώ την κλάση που έφτιαξα CustomLogisticRegression
    alpha_log = 0.1
    iters_log = 2000
    lambda_log = 1.0
    my_log_model = CustomLogisticRegression(alpha=alpha_log, iters=iters_log, lambda_=lambda_log, flag_for_prints=False)
    
    #training the training set
    my_log_model.fit(X_train_norm, y_train_class)
    
    #5-fold cross validation (αυτόματα απο scikit):
    # Το cross_val_score χωρίζει αυτόματα το X_train_norm σε 5 κομμάτια, 
    # κάνει train στα 4 και test στο 1, και το επαναλαμβάνει 5 φορές!
    print("Running 5-Fold Cross Validation on Training Set...")
    cv_scores = cross_val_score(my_log_model, X_train_norm, y_train_class, cv=5)
    
    
    #LOGISTIC REGRESSION evaluation
    print("\n--- Logistic Regression Results ---")
    print(f"CV Accuracies across 5 folds: {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})")
    
    y_pred_test = my_log_model.predict(X_test_norm)
    accuracy = np.mean(y_pred_test == y_test_class)
    print(f"Logistic Test Accuracy: {accuracy:.3f}")
    
    precision = precision_score(y_test_class, y_pred_test)
    recall = recall_score(y_test_class, y_pred_test)
    f1 = f1_score(y_test_class, y_pred_test)
    print(f"Precision:               {precision:.3f}")
    print(f"Recall:                  {recall:.3f}")
    print(f"F1 Score:                {f1:.3f}")
    
    test_cost_log = compute_cost_logistic_reg(X_test_norm, y_test_class, my_log_model.w_, my_log_model.b_, lambda_=0)
    print(f"Logistic Test Cost: {test_cost_log:.4f}")
    
    
    #LOGISTIC REGRESSION inference
    prob = predict_probability(x_new_norm.reshape(1,-1), my_log_model.w_, my_log_model.b_)
    decision = my_log_model.predict(x_new_norm.reshape(1,-1))
    print("\nProbability good deal:", prob[0])
    if decision[0] == 1:
        print("Prediction: GOOD DEAL")
    else:
        print("Prediction: NOT WORTH IT")
        




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
    lr_model = LogisticRegression(l1_ratio=0, C=1.0) # C = 1/lambda
    lr_model.fit(X_train_norm, y_train_class)
    sk_log_acc = lr_model.score(X_test_norm, y_test_class)
    print(f"Sklearn Logistic Accuracy: {sk_log_acc:.3f}")
    
if __name__ == "__main__":
    main()
