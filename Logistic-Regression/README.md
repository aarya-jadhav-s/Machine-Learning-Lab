# Heart Disease Prediction using Logistic Regression

Part 3 of my supervised machine learning series: a **classification** project that predicts whether a patient has a higher chance of a heart attack using **Logistic Regression**.

## Dataset

The dataset is the [Health care: Heart attack possibility] from Kaggle has **303 rows and 14 columns**, with no missing values. The features are numeric or already label-encoded:

| Column | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting ECG results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels |
| `thal` | Thalassemia type |
| `target` | **Target variable**: 0 = lower chance of heart attack, 1 = higher chance |

## Tech Stack

- Python
- NumPy, Pandas
- Seaborn
- scikit-learn (`LogisticRegression`, `StandardScaler`, train/test split, metrics)

## What the Notebook Covers

1. **Data loading and inspection**: `head()`, `info()`, column names, and checking that the target has exactly 2 classes.
2. **Feature / target split**: `X` = all columns except `target`, `y` = `target`.
3. **Train-test split**: 80/20 with `random_state=42` (242 training rows, 61 test rows).
4. **Class balance check**: the training set has 133 rows of class 1 and 109 of class 0, which is balanced enough that the model won't be heavily biased toward one class.
5. **Model training**: `LogisticRegression(max_iter=10000)`. The iteration limit is raised so the solver can converge.
6. **Prediction and evaluation**: predicted on the test set and compared with the actual labels (accuracy and precision).
7. **Feature scaling**: standardized the features with `StandardScaler`. The scaler is **fit on the training set only** and then used to transform the test set, which prevents data leakage. The model was retrained and re-evaluated.
8. **Confusion matrix**: built the confusion matrix to see exactly which kinds of mistakes the model makes.
9. **Full classification metrics**: accuracy, precision, **recall**, F1-score and the classification report.

## Results

### Effect of Standard Scaling
 
| Metric | Without scaling | With `StandardScaler` |
|---|---|---|
| Accuracy | **88.52%** | 85.25% |
| Precision | **87.88%** | 87.10% |
 
On this particular split, scaling did not improve the model. The accuracy difference is only about 2 test rows out of 61, so it is within noise for such a small test set.
 
### Confusion Matrix (model without scaling)
 
|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | TN = 25 | FP = 4 |
| **Actual 1** | FN = 3 | TP = 29 |
 
- **FP (False Positive) = Type 1 error**: a healthy patient is flagged as high risk.
- **FN (False Negative) = Type 2 error**: a patient who is at risk is missed.
### Final Evaluation Metrics (model without scaling)
 
| Metric | Score |
|---|---|
| Accuracy | **88.52%** |
| Precision | **87.88%** |
| Recall | **90.63%** |
| F1-score | **89.23%** |
 
Per-class report:
 
| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 (lower chance) | 0.89 | 0.86 | 0.88 | 29 |
| 1 (higher chance) | 0.88 | 0.91 | 0.89 | 32 |
 
### Why Recall Matters Here
 
This is heart data, i.e. a medical diagnosis problem. In that setting the costly mistake is a **False Negative (Type 2 error)**: telling a patient they are fine when they are actually at risk. Recall measures exactly this, so it is the metric to prioritize. With a recall of **90.63%** (29 of 32 at-risk patients caught), the model is working well on this data, and it missed only 3 at-risk patients in the test set.


## Key Learnings

- Logistic Regression is used for **classification**, not regression. It predicts the probability of a class and then converts it to 0 or 1.
- Check the class distribution after splitting, because a heavily imbalanced target makes accuracy misleading.
- **Accuracy** is the share of all predictions that are correct. **Precision** is the share of predicted positives that are truly positive. **Recall** is the share of actual positives that the model finds. **F1-score** is the balance between precision and recall.
- Use **precision** when False Positives (Type 1 errors) are costly, and **recall** when False Negatives (Type 2 errors) are costly, as in medical diagnosis.
- Fit the scaler on the training data only (`fit_transform` on train, `transform` on test) to avoid data leakage.
- Scaling does not always improve results. Logistic Regression with the default `lbfgs` solver already handled this data well, though scaling mainly helps convergence speed and is important for regularization and distance-based models.
- `max_iter` may need to be increased if the model gives a convergence warning.


## How to Run

1. Clone this repository and place `Heart Data.csv` in the same folder as the notebook.
2. Install the dependencies:
   ```bash
   pip install numpy pandas seaborn scikit-learn jupyter
   ```
3. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook Logistic_Regression.ipynb
   ```

## Project Structure

```
├── Logistic_Regression.ipynb
├── Heart Data.csv
└── README.md
```

## Possible Next Steps

- Plot the **ROC-AUC curve** and a heatmap of the confusion matrix (Seaborn)
- Report recall and the confusion matrix for the scaled model too, for a fair side-by-side comparison
- Use cross-validation for a more reliable score, since the test set is only 61 rows
- Tune the decision threshold to push recall even higher if missing a positive case is unacceptable
- Compare with other classifiers (KNN, Decision Tree, Random Forest)
