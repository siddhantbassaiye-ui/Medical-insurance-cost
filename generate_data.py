"""
Generates a synthetic dataset with the SAME columns as the well-known Kaggle
'Medical Cost Personal Datasets' (insurance.csv):
age, sex, bmi, children, smoker, region, charges

If you have the real insurance.csv, just place it in data/insurance.csv and
skip this script.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1338

age = rng.integers(18, 65, N)
sex = rng.choice(["male", "female"], N)
bmi = np.clip(rng.normal(30.6, 6.1, N), 15.9, 53.1).round(2)
children = rng.choice([0, 1, 2, 3, 4, 5], N, p=[0.43, 0.24, 0.18, 0.12, 0.02, 0.01])
smoker = rng.choice(["yes", "no"], N, p=[0.205, 0.795])
region = rng.choice(["southwest", "southeast", "northwest", "northeast"], N)

is_smoker = (smoker == "yes").astype(int)
charges = (
    2500
    + 260 * (age - 18)
    + 330 * (bmi - 25).clip(min=0)
    + 480 * children
    + 23000 * is_smoker
    + 19500 * is_smoker * (bmi > 30)       # smoker + obese interaction
    + 300 * (region == "southeast")
    - 250 * (region == "southwest")
)
charges = charges * rng.lognormal(0, 0.22, N)
charges = np.clip(charges, 1121, 63770).round(2)

df = pd.DataFrame(dict(age=age, sex=sex, bmi=bmi, children=children,
                       smoker=smoker, region=region, charges=charges))
df.to_csv("data/insurance.csv", index=False)
print("Saved data/insurance.csv", df.shape)
