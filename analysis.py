import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

salary_df = pd.read_csv("salary_dataset.csv")
diversity_df = pd.read_csv("diversity_dataset.csv")

print("Salary Dataset Loaded:", salary_df.shape)
print("Diversity Dataset Loaded:", diversity_df.shape)

salary_df.columns = salary_df.columns.str.strip().str.lower().str.replace(" ", "_")
diversity_df.columns = diversity_df.columns.str.strip().str.lower().str.replace(" ", "_")

def find_column(df, keywords):
    for col in df.columns:
        for key in keywords:
            if key.lower() in col.lower():
                return col
    return None

job_col = find_column(salary_df, ["job", "title", "role"])
salary_col = find_column(salary_df, ["salary", "pay"])
gender_col = find_column(salary_df, ["gender", "sex"])

print("\nDetected Job Column:", job_col)
print("Detected Salary Column:", salary_col)
print("Detected Gender Column:", gender_col)

def clean_title(x):
    if pd.isna(x):
        return x
    x = str(x).strip().lower()
    mapping = {
        "sde": "software engineer",
        "swe": "software engineer",
        "developer": "software engineer",
        "software developer": "software engineer"
    }
    return mapping.get(x, x.title())

def clean_salary(x):
    if pd.isna(x):
        return np.nan
    x = str(x).replace(",", "").replace("$", "").strip()
    if "-" in x:
        parts = x.split("-")
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except:
            return np.nan
    try:
        return float(x)
    except:
        return np.nan

if salary_col:
    salary_df["salary_clean"] = salary_df[salary_col].apply(clean_salary)

if job_col:
    salary_df["clean_job"] = salary_df[job_col].apply(clean_title)

salary_df["duplicate_flag"] = salary_df.duplicated()

if "salary_clean" in salary_df.columns:
    salary_df["missing_salary_flag"] = salary_df["salary_clean"].isna()
    salary_df["negative_salary_flag"] = salary_df["salary_clean"] < 0

    q1 = salary_df["salary_clean"].quantile(0.25)
    q3 = salary_df["salary_clean"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    salary_df["outlier_flag"] = (
        (salary_df["salary_clean"] < lower) |
        (salary_df["salary_clean"] > upper)
    )

for col in diversity_df.columns:
    if diversity_df[col].dtype == "object":
        try:
            diversity_df[col] = pd.to_numeric(
                diversity_df[col].astype(str).str.replace("%", "", regex=False)
            )
        except:
            pass

print("\nCleaned Salary Dataset Preview:")
print(salary_df.head())

print("\nCleaned Diversity Dataset Preview:")
print(diversity_df.head())

if "salary_clean" in salary_df.columns:
    print("\nSalary Cleaning Summary:")
    print("Missing cleaned salary values:", salary_df["missing_salary_flag"].sum())
    print("Negative salary values:", salary_df["negative_salary_flag"].sum())
    print("Outliers detected:", salary_df["outlier_flag"].sum())
    print("Duplicate rows:", salary_df["duplicate_flag"].sum())

if job_col:
    print("\nTop Cleaned Job Titles:")
    print(salary_df["clean_job"].value_counts().head(10))

print("\n===== METRIC COMPUTATION =====")

if "salary_clean" in salary_df.columns and gender_col:
    pay_gap_table = salary_df.groupby(gender_col)["salary_clean"].mean()
    print("\nAverage Salary by Gender:")
    print(pay_gap_table)

    if len(pay_gap_table) >= 2:
        apg = pay_gap_table.max() - pay_gap_table.min()
        print("Adjusted Pay Gap (approx.):", round(apg, 2))
    else:
        print("Adjusted Pay Gap could not be computed because fewer than 2 gender groups were found.")
else:
    print("\nAdjusted Pay Gap could not be computed because required columns were not found.")

if all(col in salary_df.columns for col in ["missing_salary_flag", "negative_salary_flag", "outlier_flag"]):
    total_rows = len(salary_df)

    clean_rows = salary_df[
        (~salary_df["missing_salary_flag"]) &
        (~salary_df["negative_salary_flag"]) &
        (~salary_df["outlier_flag"])
    ]

    consistency_score = len(clean_rows) / total_rows if total_rows > 0 else 0
    print("\nData Consistency Score:", round(consistency_score, 4))

    flag_columns = ["missing_salary_flag", "negative_salary_flag", "outlier_flag"]
    total_flags = salary_df[flag_columns].sum().sum()
    possible_flags = len(salary_df) * len(flag_columns)
    flag_density = total_flags / possible_flags if possible_flags > 0 else 0

    print("Audit Flag Density:", round(flag_density, 4))
else:
    print("\nConsistency Score and Audit Flag Density could not be computed because required flag columns were not found.")

plt.figure()
salary_df.isnull().sum().sort_values(ascending=False).head(10).plot(kind="bar")
plt.title("Salary Dataset Missing Values")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("salary_missing.png")

if "salary_clean" in salary_df.columns:
    plt.figure()
    plt.hist(salary_df["salary_clean"].dropna(), bins=30)
    plt.title("Salary Distribution")
    plt.xlabel("Salary")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("salary_distribution.png")

if job_col:
    plt.figure()
    salary_df["clean_job"].value_counts().head(10).plot(kind="bar")
    plt.title("Top Job Titles")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("job_titles.png")

row = diversity_df.iloc[0]
non_numeric_cols = diversity_df.select_dtypes(exclude=["number"]).columns
values = row.drop(labels=non_numeric_cols).astype(float)

plt.figure()
values.plot(kind="bar")
plt.title("Diversity Distribution (Ethnicity)")
plt.xlabel("Ethnicity")
plt.ylabel("Percentage")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("diversity_distribution.png")

salary_df.to_csv("salary_dataset_cleaned.csv", index=False)
diversity_df.to_csv("diversity_dataset_cleaned.csv", index=False)

print("\nCleaned files saved:")
print("- salary_dataset_cleaned.csv")
print("- diversity_dataset_cleaned.csv")
print("- salary_missing.png")
print("- salary_distribution.png")
if job_col:
    print("- job_titles.png")
print("- diversity_distribution.png")
print("\nDONE — Cleaning, metrics, and graphs generated successfully.")