# Part 2: Supervised Machine Learning Model

## 1. Label Definitions
- **Regression Label (`y_reg`)**: We are predicting `days_to_forward` — a continuous numeric variable representing the number of days it takes for the CFPB to forward a consumer complaint to the respective company.
- **Classification Label (`y_clf`)**: A binary label derived by thresholding `y_reg` at its 75th percentile. `y_clf = 1` indicates an "Extreme Delay" (taking longer than 75% of all complaints), while `y_clf = 0` indicates a normal/fast forwarding time.

## 2. Categorical Encoding Strategies
### Label Encoding Justification
For the `Submitted via` column, categories have a natural operational speed order. We applied Label Encoding as follows:
- `Web` (1) - Fastest, instantly digitized.
- `Phone` / `Referral` (2) - Medium speed, requires manual transcription.
- `Postal mail` / `Fax` (3) - Slowest, requires physical sorting and scanning.
This mapping preserves the natural ordinal relationship regarding processing friction.

### One-Hot Encoding Justification
For columns like `Product` and `State`, there is no natural mathematical or ranked order (e.g., a "Mortgage" is not mathematically "greater than" a "Credit Card"). Using Label Encoding here would assign arbitrary integers (1, 2, 3), tricking the algorithm into assuming a false-ordinal relationship (e.g., assuming category 3 is mathematically 3x larger than category 1). **One-Hot Encoding** avoids this by creating independent binary columns for each category, dropping the first column to avoid multicollinearity (the dummy variable trap).

## 3. Data Leakage Prevention
When scaling our features using `StandardScaler`, we must call `.fit()` **only on the training set** (`X_train`), and then apply `.transform()` to both `X_train` and `X_test`. 
**Why fitting on the full dataset is Data Leakage:** The scaler calculates the mean and standard deviation of the data. If we fit it on the entire dataset, information about the *test set's* distribution (its mean/std) "leaks" into the transformation of the training set. The model would therefore be trained on data that has implicitly "seen" the test set, leading to overly optimistic and invalid performance estimates.

## 4. Regression Analysis: Linear vs Ridge
### Coefficient Interpretation
- **Large Positive Coefficient:** Associated with an *increase* in `days_to_forward`. For example, a coefficient of `+2.5` means that a one standard-deviation increase in that scaled feature is associated with forwarding the complaint 2.5 days *slower*.
- **Large Negative Coefficient:** Associated with a *decrease* in `days_to_forward`. A coefficient of `-1.8` means a one standard-deviation increase in the feature is associated with forwarding the complaint 1.8 days *faster*.

*(Refer to the Notebook output for the top 3 feature coefficients and MSE/R² values).*

### Ridge vs OLS (Linear) Regression Comparison
Ridge regression applies an **L2 penalty** to the model, which shrinks the magnitude of the coefficients toward zero to prevent overfitting. The `alpha` parameter controls the strength of this regularization: higher `alpha` = stronger penalty. Ridge often produces a different coefficient profile than OLS because it suppresses features that are highly collinear or weakly correlated with the target, distributing the weight more evenly across features to create a more robust model.

## 5. Classification Analysis: Logistic Regression
### Addressing Class Imbalance
By defining our classification target at the 75th percentile, the positive class ("Extreme Delay") only constitutes ~25% of the data. We addressed this using **`class_weight='balanced'`** in the Logistic Regression constructor. This automatically adjusts the algorithm's loss function to penalize misclassifications of the minority class more heavily, inverse to their frequency.

### Evaluation Metrics
- **Precision:** `TP / (TP + FP)` (Out of all complaints we *predicted* as delayed, how many were actually delayed?)
- **Recall:** `TP / (TP + FN)` (Out of all *actual* delayed complaints, how many did we successfully catch?)

**Which is more important?** For this specific operational task, **Recall** is more important. A False Negative (FN) means a severely delayed complaint slips through the cracks undetected, potentially violating regulatory SLA times. A False Positive (FP) just means an analyst reviews a complaint that was actually fast. Missing a regulatory deadline is much more costly than a brief manual review.

### AUC Explanation
The **AUC (Area Under the Receiver Operating Characteristic Curve)** value represents the probability that the model will rank a randomly chosen positive instance (Extreme Delay) higher than a randomly chosen negative instance. An AUC of 0.5 is random guessing; an AUC closer to 1.0 indicates strong separation ability between the two classes.

## 6. Decision-Threshold Sensitivity
*(Refer to the Notebook output table for Precision, Recall, and F1 across thresholds).*

- **F1-Maximizing Threshold:** *[See Notebook Output]*
- **Optimizing for Recall:** Because Recall is our most important metric, we would **lower the decision threshold** (e.g., from 0.50 down to 0.30). 
- **The Cost of doing so:** Lowering the threshold makes the model more aggressive at predicting "Delayed". This dramatically increases Recall (catching almost all delays) but the *cost* is a severe drop in Precision—the operational team will be flooded with False Positives, wasting time reviewing complaints that don't actually need urgent escalation.

## 7. Regularization Experiment (C Parameter)
In Logistic Regression, the `C` parameter is the inverse of regularization strength (`C = 1 / lambda`). 
- A smaller `C` (like `0.01`) means **stronger regularization**, heavily penalizing complex models and shrinking coefficients closer to zero. 
- A larger `C` (like `1.0`) allows the model to fit the training data more tightly.
*(Refer to Notebook output to see if reducing C improved or worsened performance—typically, if the model was overfitting, a smaller C improves test AUC; if it was underfitting, it worsens it).*

## 8. Bootstrap Confidence Interval
We drew 500 bootstrap samples with replacement to compute the 95% Confidence Interval for the difference in AUC between the `C=1.0` and `C=0.01` models.
- **Does it exclude zero?** *(Refer to Notebook output).*
- **Interpretation:** If the 95% CI excludes zero (e.g., [0.02, 0.05]), we can be highly confident that the `C=1.0` model's advantage is statistically significant and consistent. If it includes zero (e.g., [-0.01, 0.03]), the performance difference between the two models is likely due to random sample variance, meaning the simpler `C=0.01` model is just as reliable.
