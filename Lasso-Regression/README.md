# Insurance Charges Prediction using Lasso Regression

Part 2 of my supervised machine learning series: predicting medical insurance charges with **Lasso Regression** (L1 regularization) and tuning the regularization strength `alpha` in two ways: **manually with `Lasso`** and automatically with **`LassoCV`**.

## Dataset

The dataset is the [Medical Cost Personal Datasets] from Kaggle, and it has **1,338 rows and 7 columns**:

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
- Pandas
- Seaborn (visualization)
- scikit-learn (`Lasso`, `LassoCV`, train/test split, metrics)

## What the Notebook Covers

### Preprocessing (same as Part 1)
- Binary encoding: `sex` (female = 1, male = 0) and `smoker` (yes = 1, no = 0)
- One-hot encoding of `region` with `drop_first=True` (avoids the dummy variable trap)
- Interaction features: `age_smoker` and `bmi_smoker`
- 80/20 train-test split (`random_state=42`)

### Model 1: Lasso (manual alpha tuning)
- Trained `Lasso(alpha=0.5)` as a starting point: **MSE = 20,918,649**
- Looped over `alpha = [0.001, 0.1, 1, 2, 5, 10, 20, 30, 40, 50, 100]`, recorded the test MSE for each, and plotted alpha vs. MSE.
- Best alpha in this loop was **10** (MSE = 20,872,845). Beyond that, MSE rises as the model is over-penalized (22.4M at alpha = 100).

### Model 2: LassoCV (automatic alpha tuning)
- `LassoCV` with the same alpha list, `cv=5` (5-fold cross-validation), `max_iter=1000`, `random_state=42`
- **Best alpha: 0.001**
- **MSE: 20,922,600**
- **R² score: 0.8652**

## Results

| Model | Alpha | Test MSE |
|---|---|---|
| Lasso (starting value) | 0.5 | 20,918,649 |
| Lasso (best from manual loop) | 10 | 20,872,845 |
| LassoCV (5-fold CV) | 0.001 | 20,922,600 |

R² of the final LassoCV model: **0.865**, the same as plain Linear Regression in Part 1.

## Key Learnings

- Lasso adds an L1 penalty that shrinks coefficients and can push unimportant ones to exactly zero, which acts as feature selection.
- `alpha` controls the penalty strength. Too high and the model underfits; too low and it behaves like ordinary linear regression.
- Manually looping over alphas works but is slow. `LassoCV` does the search with cross-validation automatically.
- The manual loop picked alpha = 10 and LassoCV picked 0.001, because the loop judges each alpha on a single test split while `LassoCV` averages over 5 folds of the training data. Choosing alpha by looking at the test set can overfit to that particular test split, so the cross-validated choice is the more trustworthy one.
- The MSE differences between alphas are tiny (under 0.3%). With these features, regularization barely helps, because the data has only a few features and little noise.

## How to Run

1. Clone this repository and place `Insurance Data.csv` in the same folder as the notebook.
2. Install the dependencies:
   ```bash
   pip install pandas seaborn scikit-learn jupyter
   ```
3. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook Lasso_Regression.ipynb
   ```

## Project Structure

```
├── Lasso_Regression.ipynb
├── Insurance Data.csv
└── README.md
```

## Possible Next Steps

- Compare with Ridge / RidgeCV and ElasticNet / ElasticNetCV
- Inspect `lasso_model.coef_` to see which features Lasso shrinks to zero
- Scale features with `StandardScaler` before applying regularization
- Report RMSE and MAE alongside MSE
