# Medical Insurance Cost Prediction

A complete machine-learning project that predicts individual annual medical insurance
charges from age, sex, BMI, number of children, smoking status and region.

## 1. Problem Statement
Insurers price policies from customer risk factors. Given a person's profile, predict
the expected insurance **charges** (a regression problem) and understand which factors drive cost.

## 2. Dataset
`data/insurance.csv` – 1,338 rows, 7 columns (`age, sex, bmi, children, smoker, region, charges`),
same schema as the public Kaggle *Medical Cost Personal Datasets*.

> The bundled file is **synthetic** (created by `generate_data.py`) so the project runs offline.
> To use the real data, download Kaggle's `insurance.csv`, put it in `data/`, and re-run `train.py`.
> Do not submit synthetic results as real-world findings.

## 3. Project Structure
```
generate_data.py   create the sample dataset
train.py           EDA, feature engineering, model comparison, tuning, evaluation, saving
predict.py         command-line prediction
app.py             Streamlit web app
data/              insurance.csv
models/            insurance_model.joblib
outputs/           plots + model_comparison.csv + final_metrics.json
```

## 4. Methodology
1. **EDA** – distributions, boxplots by category, BMI/age scatter plots, correlation heatmap.
2. **Feature engineering** – `bmi_obese` (BMI ≥ 30), `smoker_obese` (smoker × obese interaction), `age_sq`.
3. **Preprocessing** – StandardScaler for numeric, One-Hot Encoding for categorical, inside one sklearn `Pipeline` (no data leakage).
4. **Models compared (5-fold CV)** – Linear Regression, Ridge, Random Forest, Gradient Boosting.
5. **Tuning** – GridSearchCV on Gradient Boosting.
6. **Evaluation** – R², MAE, RMSE on a held-out 20% test set; residual and actual-vs-predicted plots.

## 5. Results (on the bundled synthetic data)
| Model | Test R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.916 | 2998 | 4534 |
| Ridge | 0.916 | 3007 | 4537 |
| Gradient Boosting | 0.916 | 3098 | 4545 |
| Random Forest | 0.914 | 3182 | 4591 |
| **Tuned Gradient Boosting (saved)** | 0.914 | 3099 | 4603 |

Models score closely here because the synthetic data is mostly linear. On the real Kaggle
dataset, tree-based models usually beat linear ones (R² ≈ 0.86–0.88) thanks to the smoker × BMI interaction.

## 6. Key Insights
- **Smoking status** is by far the strongest cost driver.
- **Age** increases cost steadily; **BMI ≥ 30** raises cost sharply, especially for smokers.
- Sex and region have small effects; children have a modest effect.

## 7. How to Run
```bash
pip install -r requirements.txt
python generate_data.py          # skip if you use the real insurance.csv
python train.py                  # trains, saves model and plots
python predict.py --age 35 --sex male --bmi 28.5 --children 2 --smoker no --region northeast
streamlit run app.py             # web app
```

## 8. Limitations & Future Work
- Small dataset, few features (no medical history, income, or plan type).
- Add SHAP explanations, try XGBoost/LightGBM, and deploy the app online.
- Predictions are estimates, not real insurance quotes.
