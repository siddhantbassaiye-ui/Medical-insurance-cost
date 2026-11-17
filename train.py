"""
Medical Insurance Cost Prediction - full training pipeline
EDA -> preprocessing -> model comparison (CV) -> tuning -> evaluation -> save
"""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
SEED = 42

# ---------------------------------------------------------------- 1. Load
df = pd.read_csv("data/insurance.csv")
print("Shape:", df.shape)
print("Missing values:\n", df.isnull().sum(), "\n")
print(df.describe().round(2), "\n")

# ---------------------------------------------------------------- 2. EDA
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["charges"], kde=True, ax=ax[0]).set_title("Distribution of Charges")
sns.histplot(np.log1p(df["charges"]), kde=True, ax=ax[1]).set_title("Log(Charges)")
plt.tight_layout(); plt.savefig("outputs/01_charges_distribution.png", dpi=130); plt.close()

fig, ax = plt.subplots(2, 2, figsize=(12, 9))
sns.boxplot(data=df, x="smoker", y="charges", ax=ax[0, 0]).set_title("Charges by Smoker")
sns.boxplot(data=df, x="region", y="charges", ax=ax[0, 1]).set_title("Charges by Region")
sns.boxplot(data=df, x="sex", y="charges", ax=ax[1, 0]).set_title("Charges by Sex")
sns.boxplot(data=df, x="children", y="charges", ax=ax[1, 1]).set_title("Charges by Children")
plt.tight_layout(); plt.savefig("outputs/02_categorical_vs_charges.png", dpi=130); plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="bmi", y="charges", hue="smoker", alpha=0.7)
plt.title("BMI vs Charges (by Smoker)")
plt.savefig("outputs/03_bmi_vs_charges.png", dpi=130); plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="age", y="charges", hue="smoker", alpha=0.7)
plt.title("Age vs Charges (by Smoker)")
plt.savefig("outputs/04_age_vs_charges.png", dpi=130); plt.close()

enc = df.copy()
enc["sex"] = enc["sex"].map({"male": 1, "female": 0})
enc["smoker"] = enc["smoker"].map({"yes": 1, "no": 0})
plt.figure(figsize=(8, 6))
sns.heatmap(enc.drop(columns="region").corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout(); plt.savefig("outputs/05_correlation_heatmap.png", dpi=130); plt.close()

# ---------------------------------------------------------------- 3. Features
def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["bmi_obese"] = (d["bmi"] >= 30).astype(int)
    d["smoker_obese"] = ((d["smoker"] == "yes") & (d["bmi"] >= 30)).astype(int)
    d["age_sq"] = d["age"] ** 2
    return d

X = add_features(df.drop(columns="charges"))
y = df["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

num_cols = ["age", "bmi", "children", "age_sq"]
cat_cols = ["sex", "smoker", "region"]
passthrough = ["bmi_obese", "smoker_obese"]

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ("pass", "passthrough", passthrough),
])

# ---------------------------------------------------------------- 4. Compare models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=SEED),
    "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
}
kf = KFold(5, shuffle=True, random_state=SEED)
rows = []
for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("model", m)])
    cv_r2 = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rows.append({
        "Model": name,
        "CV R2 (mean)": cv_r2.mean(),
        "Test R2": r2_score(y_test, pred),
        "Test MAE": mean_absolute_error(y_test, pred),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, pred)),
    })
results = pd.DataFrame(rows).sort_values("Test R2", ascending=False).round(4)
print("\n=== Model comparison ===\n", results.to_string(index=False))
results.to_csv("outputs/model_comparison.csv", index=False)

# ---------------------------------------------------------------- 5. Tune best tree model
grid = GridSearchCV(
    Pipeline([("pre", pre), ("model", GradientBoostingRegressor(random_state=SEED))]),
    param_grid={
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [2, 3],
        "model__subsample": [0.8, 1.0],
    },
    cv=kf, scoring="r2", n_jobs=-1,
)
grid.fit(X_train, y_train)
best = grid.best_estimator_
print("\nBest params:", grid.best_params_)

pred = best.predict(X_test)
metrics = {
    "model": "Tuned Gradient Boosting",
    "best_params": grid.best_params_,
    "R2": round(r2_score(y_test, pred), 4),
    "MAE": round(mean_absolute_error(y_test, pred), 2),
    "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),
}
print("Final test metrics:", metrics)
json.dump(metrics, open("outputs/final_metrics.json", "w"), indent=2)

# ---------------------------------------------------------------- 6. Diagnostics
plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred, alpha=0.6)
lims = [y_test.min(), y_test.max()]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual charges"); plt.ylabel("Predicted charges")
plt.title(f"Actual vs Predicted (R² = {metrics['R2']})")
plt.tight_layout(); plt.savefig("outputs/06_actual_vs_predicted.png", dpi=130); plt.close()

plt.figure(figsize=(7, 4))
sns.histplot(y_test - pred, kde=True)
plt.title("Residual Distribution"); plt.xlabel("Residual")
plt.tight_layout(); plt.savefig("outputs/07_residuals.png", dpi=130); plt.close()

names = best.named_steps["pre"].get_feature_names_out()
imp = pd.Series(best.named_steps["model"].feature_importances_, index=names).sort_values()
plt.figure(figsize=(8, 5))
imp.plot(kind="barh")
plt.title("Feature Importance (Gradient Boosting)")
plt.tight_layout(); plt.savefig("outputs/08_feature_importance.png", dpi=130); plt.close()

# ---------------------------------------------------------------- 7. Save
joblib.dump(best, "models/insurance_model.joblib")
print("\nSaved model -> models/insurance_model.joblib")
