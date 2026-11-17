"""Command-line prediction.
Example:
  python predict.py --age 35 --sex male --bmi 28.5 --children 2 --smoker no --region northeast
"""
import argparse
import joblib
import pandas as pd

def add_features(d):
    d = d.copy()
    d["bmi_obese"] = (d["bmi"] >= 30).astype(int)
    d["smoker_obese"] = ((d["smoker"] == "yes") & (d["bmi"] >= 30)).astype(int)
    d["age_sq"] = d["age"] ** 2
    return d

def predict(age, sex, bmi, children, smoker, region):
    model = joblib.load("models/insurance_model.joblib")
    row = pd.DataFrame([dict(age=age, sex=sex, bmi=bmi, children=children,
                             smoker=smoker, region=region)])
    return float(model.predict(add_features(row))[0])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--age", type=int, required=True)
    p.add_argument("--sex", choices=["male", "female"], required=True)
    p.add_argument("--bmi", type=float, required=True)
    p.add_argument("--children", type=int, required=True)
    p.add_argument("--smoker", choices=["yes", "no"], required=True)
    p.add_argument("--region", choices=["northeast", "northwest", "southeast", "southwest"], required=True)
    a = p.parse_args()
    print(f"Estimated annual medical insurance cost: ${predict(**vars(a)):,.2f}")
