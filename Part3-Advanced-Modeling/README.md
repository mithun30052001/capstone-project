# Part 3: Advanced Modeling — Ensembles, Tuning, and Full ML Pipeline

## 1. Decision Tree Baseline (Unconstrained vs. Controlled)
### High Variance & Overfitting in Decision Trees
An unconstrained Decision Tree (`max_depth=None`) is highly prone to overfitting, showing very high (often 100%) training accuracy and significantly lower test accuracy.
*   **Why they are High-Variance:** Decision Trees are non-parametric models that grow greedily by making splits that maximize information gain at each node. Without constraints, the tree will keep splitting until every leaf node is pure (containing only one class). This results in a highly complex decision boundary that memorizes the noise and outliers in the training data rather than learning general patterns.

### Controlled Decision Tree Optimization
We trained a second tree with `max_depth=5` and `min_samples_split=20` to control this variance:
*   **`max_depth`:** Limits the maximum depth of the tree. Growing shallower trees reduces model complexity (variance) at the cost of some bias (underfitting), preventing the model from capturing overly specific rules.
*   **`min_samples_split`:** Dictates that a node will not split if it contains fewer than 20 samples. This prevents the tree from creating splits that only respond to small, noisy subsets of data.

*(Refer to the Notebook output to compare the unconstrained vs. controlled tree's train/test gap).*

---

## 2. Split Criteria: Gini vs. Entropy
We compared two trees under `max_depth=5` using different split criteria:
*   **Gini Impurity Formula:**
    $$\text{Gini} = 1 - \sum_{i=1}^{C} p_i^2$$
*   **Entropy Formula:**
    $$\text{Entropy} = -\sum_{i=1}^{C} p_i \log_2(p_i)$$
*   **Meaning of Gini = 0:** A Gini impurity of exactly 0 represents a **pure node**. This means all samples falling into that leaf node belong to a single class (e.g., they are all 100% "Normal Speed" or 100% "Extreme Delay"). No uncertainty exists.

---

## 3. Random Forest Ensembles & Feature Importance
### Bagging Concept
Random Forest is built on the concept of **Bagging (Bootstrap Aggregating)**. 
- **Bootstrap Sampling:** Each tree in the ensemble is trained on a random sample of the training data selected *with replacement*. This means some samples are repeated, and about 36.8% of the data (out-of-bag samples) is unseen by any individual tree.
- **Feature Subspace Sampling:** At each split in each tree, only a random subset of features (typically $\sqrt{\text{total features}}$) is considered. 
- **Ensemble Averaging:** By averaging the predictions of many deep, diverse, and uncorrelated trees, Random Forest significantly reduces variance compared to a single deep decision tree without increasing bias.

### Feature Importance Calculation
*   **Random Forest Feature Importance:** Computed as the **mean decrease in impurity (MDI)**. It measures the average reduction in Gini impurity or Entropy that a feature contributes across all splits, weighted by the number of samples reaching those nodes, averaged across all 100 trees.
*   **Difference from Linear Regression Coefficients:** Linear regression coefficients represent the change in the target variable for a unit change in the feature, assuming all other features remain constant. They are highly sensitive to multicollinearity and scale. Random Forest feature importance is non-linear, scale-invariant, always positive, and represents how useful a feature is for partitioning the classes, regardless of linear relationships.

---

## 4. Feature Ablation Study & Production Trade-off
We identified the 5 lowest-importance features using our Random Forest and evaluated a reduced model with these features removed:
*   **Analysis:** *(Refer to Notebook outputs for Full vs. Reduced AUC).*
*   **Production Trade-off:** Deploying a simpler, lower-dimensional model reduces computational inference cost and maintenance burden (e.g., fewer features to query, ingest, clean, and monitor in production). However, this simplification is only acceptable if the degradation in AUC is below a tolerable business threshold (typically $< 1\%$). If the AUC drops significantly, the more complex model must be retained.

---

## 5. Cross-Validated Comparison
A single train-test split can be misleading due to random sampling variance (the split might happen to contain an unusually easy or hard test set). 
**Why Cross-Validation (CV) is more reliable:** Stratified 5-Fold Cross-Validation splits the dataset into 5 equal folds. The model is trained on 4 folds and tested on the 5th, repeating this process 5 times so every single data point is used for testing exactly once. The mean and standard deviation of these 5 runs provide a robust estimate of both the model's true performance and its stability across different data samples.

---

## 6. Hyperparameter Tuning (GridSearchCV)
We tuned our Random Forest pipeline using `GridSearchCV`:
*   **Total Model Configurations Evaluated:** 2 (n_estimators) $\times$ 2 (max_depth) $\times$ 1 (min_samples_leaf) $\times$ 3 (CV folds) = **12 total fits** (executed on a stratified 20% sample of the training set to optimize search speed, then refitted on 100% of the training data).
*   **Grid Search vs. Randomized Search:** Grid Search is exhaustive and guarantees finding the absolute best combination within the specified grid, but it is highly computationally expensive. Randomized Search samples a fixed number of parameter settings from specified distributions, which is far faster and typically finds a near-optimal solution with a fraction of the computational budget.

---

## 7. Learning Curve Analysis
We evaluated the best pipeline on subsets of the training data from 20% to 100%:
*   **Training AUC Trend:** Typically decreases as the dataset grows because it is harder to perfectly overfit a larger, more diverse dataset.
*   **Test AUC Trend:** Typically increases as the dataset grows because the model has more examples from which to learn generalizable rules.
*   **Data-Limited vs. Capacity-Limited Conclusion:** *(Refer to your notebook output table. If the test AUC continues to rise steeply at 100%, the model is data-limited and collecting more data will help. If it plateaus, the model is capacity-limited, and we need a more complex model architecture).*

---

## 8. Summary Model Comparison & Recommendation

| Model | 5-Fold CV Mean AUC | 5-Fold CV Std AUC | Test-Set AUC |
|---|---|---|---|
| Logistic Regression (Part 2) | *[Notebook Output]* | *[Notebook Output]* | *[Notebook Output]* |
| Controlled Decision Tree | *[Notebook Output]* | *[Notebook Output]* | *[Notebook Output]* |
| Random Forest (Baseline) | *[Notebook Output]* | *[Notebook Output]* | *[Notebook Output]* |
| Gradient Boosting | *[Notebook Output]* | *[Notebook Output]* | *[Notebook Output]* |
| Tuned Random Forest (GridSearchCV) | *[Notebook Output]* | *[Notebook Output]* | *[Notebook Output]* |

### Final Client Recommendation
*(Justify your recommended model in 3-5 sentences based on the CV Mean AUC and standard deviation, explaining why it is the most stable and accurate choice for their FinTech billing pipeline).*
