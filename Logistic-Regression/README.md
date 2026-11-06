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
- scikit-learn (`LogisticRegression`, train/test split, metrics)

## What the Notebook Covers

1. **Data loading and inspection**: `head()`, `info()`, column names, and checking that the target has exactly 2 classes.
2. **Feature / target split**: `X` = all columns except `target`, `y` = `target`.
3. **Train-test split**: 80/20 with `random_state=42` (242 training rows, 61 test rows).
4. **Class balance check**: the training set has 133 rows of class 1 and 109 of class 0, which is balanced enough that the model won't be heavily biased toward one class.
5. **Model training**: `LogisticRegression(max_iter=10000)`. The iteration limit is raised so the solver can converge.
6. **Prediction and evaluation**: predicted on the test set and compared with the actual labels.

## Results

| Metric | Score |
|---|---|
| Accuracy | **88.52%** |
| Precision | **87.88%** |

## Key Learnings

- Logistic Regression is used for **classification**, not regression. It predicts the probability of a class and then converts it to 0 or 1.
- Check the class distribution after splitting, because a heavily imbalanced target makes accuracy misleading.
- **Accuracy** is the share of all predictions that are correct. **Precision** is the share of predicted positives that are truly positive.
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

- Add **recall**, **F1-score**, a confusion matrix, and the ROC-AUC curve. Recall is especially important in medical problems, where missing a positive case is costly.
- Scale the features with `StandardScaler`
- Use cross-validation for a more reliable score, since the test set is only 61 rows
- Compare with other classifiers (KNN, Decision Tree, Random Forest)
