# FinTech Consumer Complaints: ML & LLM Pipeline
**IIT AI/ML Capstone Project**

## Part 1: Data Acquisition, Cleaning, and Exploratory Analysis

### Dataset Description & Justification
For this project, I selected the **Consumer Financial Protection Bureau (CFPB) Consumer Complaints Database**. This is a publicly available, uncleaned government dataset detailing financial complaints filed by consumers against institutions. 
This dataset is ideal because it bridges traditional tabular Machine Learning (predicting dispute likelihood or resolution time) with modern Large Language Model applications (extracting structured JSON from messy, emotional consumer narrative texts).

### Data Download Instructions
Due to the massive size of the CFPB database (>5 Million rows, several gigabytes), the raw dataset is not hosted directly in this repository to comply with GitHub's file size limits. 

To replicate this project and run the notebook, please download the dataset directly from the U.S. government portal using one of the following methods:

**Method 1: Direct Web Download (Recommended)**
1. Navigate to: [http://consumerfinance.gov/data-research/consumer-complaints/#get-the-data](http://consumerfinance.gov/data-research/consumer-complaints/#get-the-data)
2. Click **Download the full dataset**.
3. Choose the **CSV** format.
4. Compress the downloaded file to a zip file if you are using the provided notebook (e.g., `complaints.csv.zip`).
5. Place the zipped file in the same directory as the Jupyter notebook, or upload it to your Colab workspace.

**Method 2: Programmatic Download via API (Filtered Sample)**
If you prefer to download a smaller, targeted sample (e.g., 20,000 rows) directly via Python, you can use the official API:
```python
import requests
import pandas as pd

url = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
params = {
    "date_received_min": "2023-01-01",
    "product": "Credit card or prepaid card",
    "size": 20000,
    "format": "csv"
}
response = requests.get(url, params=params)
with open('complaints.csv', 'wb') as f:
    f.write(response.content)
```

### Imputation Strategy Justification (Task 5 & 8a)
To satisfy the rubric, numeric features were engineered from the dataset: `days_to_forward` (difference between date received and sent to company) and `narrative_length`.
- **Highest Skewness Column:** `narrative_length` exhibited extreme **positive skewness** (a long right tail). 
- **Meaning of Positive Skew:** This means the majority of complaints are relatively short, but a small handful of angry consumers wrote massive, multi-page essays, pulling the mean significantly upward.
- **Why Median over Mean:** Because the mean is heavily distorted by these extreme high values, it ceases to represent the "typical" complaint. The **median** is resistant to extreme outliers, making it a much more accurate measure of central tendency. Therefore, the median was used for imputing any remaining nulls in highly skewed numeric columns.

### Outlier Detection with IQR (Task 6)
Two numeric columns were analyzed using the Interquartile Range (IQR) method:
1. **`narrative_length`:** A large number of outliers were detected above the upper bound. These represent highly detailed complaints. 
   - *Action:* **Retain them.** In NLP/LLM tasks (Part 3), long narratives are incredibly valuable for extracting structured insights. Capping them would destroy vital contextual data.
2. **`days_to_forward`:** Outliers were detected where the CFPB took an unusually long time to forward the complaint to the company.
   - *Action:* **Cap them.** For predictive modeling in Part 2, extreme delays (e.g., > 100 days) are likely administrative errors or extreme edge cases that will add noise to a regression model predicting resolution time. I will cap these at the 99th percentile in Part 2.

### Visualizations Interpretation (Task 7)
- **Histogram (`narrative_length`):** The distribution is heavily right-skewed (positive skew). Most complaints cluster around 0-500 characters, trailing off into a long tail of very lengthy complaints.
- **Scatter Plot (`narrative_length` vs `days_to_forward`):** The plot shows a weak, near-zero correlation. The length of a consumer's complaint does not visibly impact how quickly the CFPB forwards it to the company.
- **Box Plot (`days_to_forward` split by `Submitted via`):** There is a visible difference in medians. Complaints submitted via 'Web' have a tighter spread and lower median forwarding time compared to 'Postal mail' or 'Fax', which show larger variances and higher medians due to manual processing.
- **Correlation Heatmap & Alternative Explanation:** The highest absolute correlation observed was between `narrative_length` and `days_to_forward` (though still relatively weak). 
   - *Causal or Third Variable?* This is unlikely to be causal (a long complaint doesn't mechanically take longer for an automated system to route). A plausible **third variable (confounder)** is the *complexity of the complaint type*. Complex products like 'Mortgages' likely require both longer explanations from the consumer and manual legal review by the CFPB, increasing both variables simultaneously.

### Spearman Rank Correlation (Task 8b)
The Spearman correlation captures monotonic (consistent but not necessarily linear) relationships, while Pearson captures strictly proportional linear relationships. 
- **Highest |Spearman - Pearson| differences:**
  1. For the top pair, `|Spearman| > |Pearson|`. This indicates a non-linear but monotonic relationship (e.g., as X increases, Y consistently increases, but at an accelerating or decelerating rate).
  2. *Feature Selection Choice:* Moving into Part 2, I will rely on **Spearman correlation**. Because financial and text-derived features are rarely perfectly normally distributed (often highly skewed), a rank-based measure like Spearman provides a more robust signal for feature selection against non-linear tree-based models (like Random Forest or XGBoost).

### Grouped Aggregation (Task 8c)
- **Group Analysis:** Grouping by `Submitted via` against `days_to_forward`.
- **Highest Mean & Standard Deviation:** 'Postal mail' or 'Fax' typically exhibits the highest mean and highest standard deviation.
- **Implication of High Within-Group Variance:** High within-group standard deviation is a concern. It means that just knowing a complaint was submitted by mail is not enough to accurately predict its forwarding time—some mails are processed in 2 days, others in 30 days. The feature is noisy.
- **Predictive Signal:** The ratio of the highest group mean to the lowest group mean is calculated in the notebook. Since the ratio is notable (e.g., > 1.5), it suggests that despite the noise, the categorical feature still carries a valid predictive signal and should be included in the ML model.
