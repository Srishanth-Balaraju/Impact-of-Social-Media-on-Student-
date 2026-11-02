import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv(r"C:\Users\kunut\OneDrive\Desktop\Social_media_impact_on_life.csv")

print("DATASET INFORMATION")
print("Shape:", data.shape)
print("\nFirst five rows:")
print(data.head())
print("\nColumn names:")
print(data.columns.tolist())
print("\nDataset information:")
data.info()
print("\nStatistical summary:")
print(data.describe(include="all").T)

# ============================================================
# 2. DATA CLEANING AND PREPROCESSING
# ============================================================

# Remove spaces from column names
data.columns = data.columns.str.strip()

# Remove duplicate rows
print("\nDuplicate rows:", data.duplicated().sum())
data = data.drop_duplicates().copy()

# Remove unnecessary student identifier
if "Student_ID" in data.columns:
    data = data.drop(columns=["Student_ID"])

# Clean text columns
for col in data.select_dtypes(include=["object", "string"]).columns:
    data[col] = data[col].astype("string").str.strip()
    data[col] = data[col].replace("", pd.NA)

# Convert boolean feature into numeric format
if "Late_Night_Usage" in data.columns:
    data["Late_Night_Usage"] = data["Late_Night_Usage"].map({
        True: 1,
        False: 0,
        "True": 1,
        "False": 0,
        "Yes": 1,
        "No": 0
    })

# Convert numeric columns to numeric data types
numeric_columns = [
    "Age",
    "Daily_Usage_Hours",
    "Weekend_Extra_Hours",
    "Sleep_Duration_Hours",
    "Sleep_Quality_Score",
    "Late_Night_Usage",
    "Perceived_Stress_Score",
    "Mental_Health_Index",
    "Academic_Performance_GPA"
]

for col in numeric_columns:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

# Replace invalid values with missing values
if "Age" in data.columns:
    data.loc[data["Age"] <= 0, "Age"] = np.nan
if "Daily_Usage_Hours" in data.columns:
    data.loc[data["Daily_Usage_Hours"] < 0, "Daily_Usage_Hours"] = np.nan
if "Weekend_Extra_Hours" in data.columns:
    data.loc[data["Weekend_Extra_Hours"] < 0, "Weekend_Extra_Hours"] = np.nan
if "Sleep_Duration_Hours" in data.columns:
    data.loc[data["Sleep_Duration_Hours"] < 0, "Sleep_Duration_Hours"] = np.nan

# Remove rows with missing target values
data = data.dropna(subset=["Overall_Impact"]).copy()

print("\nMissing values after initial cleaning:")
print(data.isnull().sum())
print("\nDataset shape after cleaning:", data.shape)

# ============================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

sns.set_theme(style="whitegrid")

# OBJECTIVE 1: Overall Impact Distribution
print("\nOBJECTIVE 1: Overall Impact Distribution")
print(data["Overall_Impact"].value_counts())
plt.figure(figsize=(8, 5))
data["Overall_Impact"].value_counts().plot(kind="bar", edgecolor="black")
plt.title("Distribution of Social Media Impact")
plt.xlabel("Overall Impact")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("01_impact_distribution.png")
plt.show()

# OBJECTIVE 2: Daily Social Media Usage by Impact
usage_by_impact = data.groupby("Overall_Impact")["Daily_Usage_Hours"].mean()
print("\nOBJECTIVE 2: Average Daily Usage by Impact")
print(usage_by_impact)
plt.figure(figsize=(8, 5))
usage_by_impact.plot(kind="bar", edgecolor="black")
plt.title("Average Daily Social Media Usage")
plt.xlabel("Overall Impact")
plt.ylabel("Daily Usage Hours")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("02_daily_usage.png")
plt.show()

# OBJECTIVE 3: Sleep Duration by Impact
sleep_by_impact = data.groupby("Overall_Impact")["Sleep_Duration_Hours"].mean()
print("\nOBJECTIVE 3: Average Sleep Duration")
print(sleep_by_impact)
plt.figure(figsize=(8, 5))
sleep_by_impact.plot(kind="bar", edgecolor="black")
plt.title("Sleep Duration by Social Media Impact")
plt.xlabel("Overall Impact")
plt.ylabel("Average Sleep Hours")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("03_sleep_analysis.png")
plt.show()

# OBJECTIVE 4: Perceived Stress by Impact
stress_by_impact = data.groupby("Overall_Impact")["Perceived_Stress_Score"].mean()
print("\nOBJECTIVE 4: Average Perceived Stress")
print(stress_by_impact)
plt.figure(figsize=(8, 5))
stress_by_impact.plot(kind="bar", edgecolor="black")
plt.title("Perceived Stress by Social Media Impact")
plt.xlabel("Overall Impact")
plt.ylabel("Average Stress Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("04_stress_analysis.png")
plt.show()

# OBJECTIVE 5: Academic Performance by Impact
gpa_by_impact = data.groupby("Overall_Impact")["Academic_Performance_GPA"].mean()
print("\nOBJECTIVE 5: Average Academic Performance")
print(gpa_by_impact)
plt.figure(figsize=(8, 5))
gpa_by_impact.plot(kind="bar", edgecolor="black")
plt.title("Academic Performance by Social Media Impact")
plt.xlabel("Overall Impact")
plt.ylabel("Average GPA")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("05_academic_performance.png")
plt.show()

# Additional EDA: Correlation Heatmap
numeric_data = data.select_dtypes(include=["number"])
plt.figure(figsize=(11, 7))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("06_correlation_heatmap.png")
plt.show()

# Compare impact across social media platforms
if "Primary_Platform" in data.columns:
    platform_impact = pd.crosstab(
        data["Primary_Platform"],
        data["Overall_Impact"],
        normalize="index"
    )
    print("\nImpact distribution by platform:")
    print(platform_impact.round(3))
    plt.figure(figsize=(10, 6))
    platform_impact.plot(kind="bar", stacked=True, figsize=(10, 6))
    plt.title("Social Media Impact by Primary Platform")
    plt.xlabel("Primary Platform")
    plt.ylabel("Proportion of Students")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("07_platform_impact.png")
    plt.show()

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

data["Estimated_Usage_Hours"] = (
    data["Daily_Usage_Hours"] + data["Weekend_Extra_Hours"].fillna(0)
)

y = data["Overall_Impact"].astype(str)
X = data.drop(columns=["Overall_Impact"]).dropna(axis=1, how="all")

print("\nFeatures used:")
print(X.columns.tolist())
print("\nTarget distribution:")
print(y.value_counts())

# ============================================================
# 5. TRAIN-TEST SPLIT (75:25)
# ============================================================

class_counts = y.value_counts()
if len(class_counts) < 2 or class_counts.min() < 2:
    raise ValueError(
        "Each target category needs at least two records "
        "for a stratified train-test split."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ============================================================
# 6. HANDLE MISSING VALUES AND ENCODE CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# sparse_output=False ensures dense output needed by GaussianNB
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# ============================================================
# 7. BUILD FOUR MACHINE LEARNING MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        class_weight="balanced",
        random_state=42
    )
}

# ============================================================
# 8. TRAIN AND EVALUATE MODELS
# ============================================================

results = []
trained_models = {}
predictions = {}

for name, classifier in models.items():
    print("\nTraining model:", name)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Weighted F1": weighted_f1,
        "Macro F1": macro_f1
    })
    trained_models[name] = model
    predictions[name] = y_pred

    print("Accuracy:", round(accuracy, 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

# ============================================================
# 9. COMPARE MODELS AND SELECT THE BEST MODEL
# ============================================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by=["Macro F1", "Accuracy"],
    ascending=False
).reset_index(drop=True)

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))
results_df.to_csv("model_comparison.csv", index=False)

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
best_prediction = predictions[best_model_name]

print("\nBest-fit model:", best_model_name)
print("Best model accuracy:", round(results_df.iloc[0]["Accuracy"], 4))
print("Best model macro F1:", round(results_df.iloc[0]["Macro F1"], 4))

# Plot model accuracy
plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x="Model", y="Accuracy")
plt.title("Accuracy Comparison of Four Models")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("08_model_accuracy.png")
plt.show()

# Plot macro F1-score
plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x="Model", y="Macro F1")
plt.title("Macro F1-Score Comparison")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("09_model_f1_score.png")
plt.show()

# ============================================================
# 10. CONFUSION MATRIX FOR THE BEST MODEL
# ============================================================

class_labels = best_model.named_steps["classifier"].classes_
cm = confusion_matrix(y_test, best_prediction, labels=class_labels)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

fig, ax = plt.subplots(figsize=(8, 6))
display.plot(ax=ax, cmap="Blues", xticks_rotation=30)
plt.title("Confusion Matrix - " + best_model_name)
plt.tight_layout()
plt.savefig("10_confusion_matrix.png")
plt.show()

# ============================================================
# 11. SAVE ACTUAL AND PREDICTED RESULTS
# ============================================================

prediction_results = X_test.copy()
prediction_results["Actual_Impact"] = y_test
prediction_results["Predicted_Impact"] = best_prediction
prediction_results.to_csv("best_model_predictions.csv", index=False)

# ============================================================
# 12. FINAL FINDINGS AND RECOMMENDATIONS
# ============================================================

print("\n========== PROJECT SUMMARY ==========")
print("Total cleaned records:", len(data))
print("Number of target categories:", y.nunique())
print("Best-fit algorithm:", best_model_name)
print(f"Best model accuracy: {results_df.iloc[0]['Accuracy']:.2%}")
print(f"Best model weighted F1: {results_df.iloc[0]['Weighted F1']:.4f}")
print(f"Best model macro F1: {results_df.iloc[0]['Macro F1']:.4f}")

print("\nRecommendations:")
print("1. Examine daily social media usage patterns.")
print("2. Investigate relationships between usage and sleep.")
print("3. Examine stress and academic performance patterns.")
print("4. Compare performance across all four ML algorithms.")
print("5. Review class-wise recall because the target may be imbalanced.")
print("6. Treat these findings as associations, not proof of causation.")

print("\nProject completed successfully.")
print("Charts and CSV results are saved in the current folder.")
