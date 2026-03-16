# Real Estate ML Toolkit (From Scratch)

## Project Description
This project is a machine learning pipeline designed to estimate house prices and identify "Good Deal" investment opportunities. 

The core of this project is the **manual implementation** of fundamental algorithms. Instead of just calling library functions, I wrote the Gradient Descent and Regularization logic from scratch using NumPy to fully understand the optimization process.

---

## Features & Implementation

### 1. Synthetic Data Generation (`data_utils.py`)
To test the models, I built a custom data generator that creates a realistic house dataset.
* **Features**: Square footage ($m^2$), Number of Bedrooms, and House Age.
* **Target**: House Price (generated via a linear formula + Gaussian noise).
* **Why?**: It allows for controlled testing of the algorithm's convergence before moving to real-world "messy" data.

### 2. Linear Regression (`linear_regression.py`)
Used for predicting the exact price of a property.
* **Gradient Descent**: Manual implementation of the weight update rule.
* **Z-score Normalization**: Crucial step to scale features (e.g., mapping 150 $m^2$ and 2 bedrooms to the same scale) so the algorithm converges efficiently.
* **L2 Regularization**: Prevents the model from "over-memorizing" the noise in the data.

### 3. Logistic Regression & Classification (`logistic_regression.py`)
Used to classify if a house is a "Good Deal" based on a price threshold.
* **Logic**: If the actual price is < 75% of the theoretical market value, it's labeled as a `Good Deal (1)`.
* **Wrapper Strategy**:I implemented a Custom Wrapper Class to make my "from scratch" code compatible with Scikit-Learn’s professional evaluation tools.
---

## Results & Performance Analysis

### Why is the Accuracy 1.000?
The Classification model typically achieves **100% accuracy** (or very close to it).
* **The Reason**: Since the "Good Deal" label is generated using Linear Relationships (simple additions and multiplications) within our synthetic data, the Logistic Regression model is able to find the perfect **Decision Boundary** that separates the data points.
* **Validation**: The high score confirms that the mathematical implementation of Gradient Descent is working exactly as intended for linearly separable data.
* **Real-world Note**: In real datasets, accuracy would be lower due to human factors and missing information. Here, it simply proves that the mathematical implementation of Gradient Descent is working correctly.

### Robustness: 5-Fold Cross-Validation
To ensure the model isn't just "lucky" with a single split, I implemented **5-Fold Cross-Validation**. The data is split into 5 different parts, and the model is trained and tested 5 times. 
* **Result**: The Mean CV Accuracy remains around **98-100%**, proving the model is highly stable.

### Precision, Recall, and F1 Score (The 0.000 Result)
I added advanced metrics (Precision, Recall, F1) to monitor if the model misses deals or gives false alarms.
* **The "Zero" Issue**: If these metrics appear as **0.000** or trigger `UndefinedMetricWarnings`, it means that in that specific random Test Set, **there were zero "Good Deals" present**. This is a common "Imbalanced Data" scenario—if the sample has no bargains, the metrics cannot be calculated.

---

### Comparison with Scikit-Learn
To validate my "from scratch" code, I compared it against the industry-standard `sklearn` library:
* **Price Prediction**: My model's prediction was within **~13€** of Scikit-Learn's `SGDRegressor` prediction (on a 350,000€ house).
* **Consistency**: Both models reached identical conclusions on the classification test set, confirming the custom sigmoid and cost functions are accurate.

---

## Future Improvements
To keep the project focused on fundamentals, I left out the following, which would be next steps for a production-level tool:
1.  **Polynomial Features**: To capture non-linear trends (e.g., when price per $m^2$ increases for luxury houses).
2.  **Real-World Datasets**: Testing on Kaggle data (e.g., Ames Housing Dataset).
3.  **Cross-Validation**: Splitting data multiple times to ensure the model isn't biased by one specific train/test split.

---

## How to Run
1.  **Activate Environment**: `.\.venv\Scripts\activate`
2.  **Install Dependencies**: `pip install numpy pandas scikit-learn`
3.  **Execute**: `python src/main.py`
