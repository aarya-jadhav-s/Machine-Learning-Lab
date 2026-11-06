# Insurance Charges Prediction using Linear Regression

My first supervised machine learning project: predicting individual medical insurance charges with **Linear Regression** in Python (scikit-learn).

## Dataset

`Insurance Data.csv` has **1,338 rows and 7 columns**:

| Column | Description |
|---|---|
| `age` | Age of the person |
| `sex` | Female / male |
| `bmi` | Body Mass Index |
| `children` | Number of children / dependents |
| `smoker` | Whether the person smokes (yes / no) |
| `region` | Residential region (northeast, northwest, southeast, southwest) |
| `charges` | Medical insurance cost (**target variable**) |

## Tech Stack

- Python
- NumPy, Pandas
- Seaborn (visualization)
- scikit-learn (model, train/test split, metrics)

## What the Notebook Covers

1. **Data loading and exploration**: scatter plot of BMI vs. charges, coloured by smoker status. Charges rise with BMI, and smokers pay noticeably more.
2. **Baseline model**: dropped `region`, encoded `sex` and `smoker` as 0/1, split the data 80/20 (`random_state=42`), and trained `LinearRegression`.
3. **Evaluation**: R² and Adjusted R².
4. **Feature engineering**:
   - One-hot encoding of `region`
   - Avoiding the **dummy variable trap** with `drop_first=True`
   - **Interaction features**: `age × smoker` and `bmi × smoker`
5. **Overfitting / underfitting check**: compared training vs. testing R².

## Results

| Model | Test R² |
|---|---|
| Baseline (without `region`) | 0.781 (Adjusted R²: 0.777) |
| + One-hot encoded `region` | 0.784 |
| + `region` with `drop_first=True` | 0.784 |
| + Interaction features (`age_smoker`, `bmi_smoker`) | **0.865** |

**Final model:** Training R² = 0.834, Testing R² = 0.865. The two scores are close, so the model is neither overfitting nor underfitting.

## Key Learnings

- Categorical columns must be converted to numbers before modelling (binary mapping for `sex` and `smoker`, one-hot encoding for `region`).
- `drop_first=True` avoids the dummy variable trap (perfect multicollinearity between dummy columns).
- Interaction features gave the biggest improvement (R² 0.78 → 0.87), because the effect of age and BMI on charges is much stronger for smokers.
- Comparing training and testing R² is a simple way to detect overfitting or underfitting.
- Mapping a column twice returns `NaN` values, because the original text values are already overwritten. Re-run from the data-loading cell if this happens.

## How to Run

1. Clone this repository and place `Insurance Data.csv` in the same folder as the notebook.
2. Install the dependencies:
   ```bash
   pip install numpy pandas seaborn scikit-learn jupyter
   ```
3. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook Linear_Regression.ipynb
   ```

## Project Structure

```
├── Linear_Regression.ipynb
├── Insurance Data.csv
└── README.md
```

## Possible Next Steps

- Add other metrics such as MAE and RMSE
- Scale features and try Ridge / Lasso regression
- Try polynomial features or tree-based models (Random Forest, Gradient Boosting)
- Add cross-validation for more reliable scores
