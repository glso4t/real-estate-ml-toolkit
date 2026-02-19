
import os
import numpy as np
import pandas as pd


def make_dataset(m=200, seed=161):
    """
    creates ένα απλό dataset για σπίτια
    
    m=200 σημαίνει έστω 200 σπίτα για dataset
    seed=161 κάνει τα random αποτελέσματα σταθερα 
    
    Features:
      - size_m2: τετραγωνικά σπιτιού
      - bedrooms: υπνοδωμάτια
      - age: ηλικία σπιτιού (σε έτη)

    Target:
      - price: τιμή σπιτιού (σε euros)
    """
    np.random.seed(seed)    #κάθε φορά να βγαίνει το ίδιο dataset
    

    # -----------------------------
    # X features
    # -----------------------------
    
    size_m2 = np.random.uniform(40, 180, m)      # απο 40 έως 180 τετραγωνικα, uniform=δεκαδικοί
    bedrooms = np.random.randint(1, 6, m)        # απο 1 έως 5 υπνοδωματια, randint=ακέραιοι
    age = np.random.uniform(0, 50, m)            # απο 0 έως 50 χρόνια

    # -----------------------------
    # price = w1*size + w2*bedrooms + w3*age + b + noise
    # -----------------------------
    
    w1 = 2500
    w2 = 15000
    w3 = -1200
    b = 20000

    noise = np.random.normal(0, 20000, m)

    price = w1 * size_m2 + w2 * bedrooms + w3 * age + b + noise     #noise= τυχαίο σφάλμα πραγματικών δεδομένων πχ θέα, γειτονια κλπ

    df = pd.DataFrame({
        "size_m2": size_m2,
        "bedrooms": bedrooms,
        "age": age,
        "price": price
    })

    return df


def load_or_create_csv(path="data/houses.csv", m=200, seed=161):
    """
    load or create csv
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        return pd.read_csv(path)

    df = make_dataset(m=m, seed=seed)
    df.to_csv(path, index=False)
    return df