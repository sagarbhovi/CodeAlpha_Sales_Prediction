
"""
PROJECT: SALES PREDICTION USING MACHINE LEARNING
TASK 4 - CODEALPHA DATA SCIENCE INTERNSHIP

Dataset: Advertising Dataset
Target: Sales
Features: TV, Radio, Newspaper

Models:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    RepeatedKFold,
    RandomizedSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

DATASET_PATH = os.path.join("Dataset", "advertising.csv")
OUTPUT_DIR = "outputs"

VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

RANDOM_STATE = 42

FEATURES = ["TV", "Radio", "Newspaper"]
TARGET = "Sales"

os.makedirs(VISUALIZATION_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

print("=" * 65)
print("             SALES PREDICTION USING MACHINE LEARNING")
print("=" * 65)


def save_plot(filename):
    """Save the current visualization."""
    path = os.path.join(VISUALIZATION_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved visualization:", filename)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n[1] LOADING DATASET")
print("-" * 65)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {DATASET_PATH}. "
        "Place advertising.csv inside the Dataset folder."
    )

df = pd.read_csv(DATASET_PATH)

print("Original dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nDataset columns:")
print(df.columns.tolist())


# ============================================================
# 3. DATA CLEANING
# ============================================================

print("\n[2] DATA CLEANING")
print("-" * 65)

# Remove unnecessary index columns
unnamed_columns = [
    col for col in df.columns
    if col.strip().lower().startswith("unnamed")
]

df.drop(columns=unnamed_columns, inplace=True, errors="ignore")

# Clean column names
df.columns = df.columns.str.strip()

# Validate required columns
required_columns = FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing from dataset: {missing_columns}"
    )

# Keep only the required columns
df = df[required_columns].copy()

# Convert columns to numeric values
for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Handle missing values
missing_before = df.isnull().sum().sum()
df.dropna(inplace=True)

# Remove duplicate records
duplicates_before = df.duplicated().sum()
df.drop_duplicates(inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

if df.empty:
    raise ValueError("Dataset is empty after cleaning.")

print("Missing values removed:", missing_before)
print("Duplicate records removed:", duplicates_before)
print("Cleaned dataset shape:", df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Save cleaned dataset
cleaned_path = os.path.join(OUTPUT_DIR, "cleaned_advertising_data.csv")
df.to_csv(cleaned_path, index=False)

print("\nCleaned dataset saved:", cleaned_path)


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n[3] EXPLORATORY DATA ANALYSIS")
print("-" * 65)

print("\nDescriptive statistics:")
print(df.describe().round(2))

print("\nCorrelation matrix:")
print(df.corr().round(3))

print("\nCorrelation with Sales:")
print(df.corr()[TARGET].sort_values(ascending=False).round(3))

print("\nAverage advertising budgets and sales:")
print(df[required_columns].mean().round(2))


# ============================================================
# 5. DATA VISUALIZATION
# ============================================================

print("\n[4] CREATING VISUALIZATIONS")
print("-" * 65)

# Visualization 1: Sales distribution
plt.figure(figsize=(9, 5))
sns.histplot(df[TARGET], kde=True, bins=20)
plt.title("Distribution of Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")
save_plot("sales_distribution.png")


# Visualization 2: TV advertising vs Sales
plt.figure(figsize=(8, 5))
sns.regplot(
    data=df,
    x="TV",
    y="Sales",
    scatter_kws={"alpha": 0.7}
)
plt.title("TV Advertising vs Sales")
plt.xlabel("TV Advertising Budget")
plt.ylabel("Sales")
save_plot("tv_vs_sales.png")


# Visualization 3: Radio advertising vs Sales
plt.figure(figsize=(8, 5))
sns.regplot(
    data=df,
    x="Radio",
    y="Sales",
    scatter_kws={"alpha": 0.7}
)
plt.title("Radio Advertising vs Sales")
plt.xlabel("Radio Advertising Budget")
plt.ylabel("Sales")
save_plot("radio_vs_sales.png")


# Visualization 4: Newspaper advertising vs Sales
plt.figure(figsize=(8, 5))
sns.regplot(
    data=df,
    x="Newspaper",
    y="Sales",
    scatter_kws={"alpha": 0.7}
)
plt.title("Newspaper Advertising vs Sales")
plt.xlabel("Newspaper Advertising Budget")
plt.ylabel("Sales")
save_plot("newspaper_vs_sales.png")


# Visualization 5: Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation Heatmap")
save_plot("correlation_heatmap.png")


# Visualization 6: Advertising budget comparison
plt.figure(figsize=(9, 5))

budget_means = df[FEATURES].mean()

sns.barplot(
    x=budget_means.index,
    y=budget_means.values
)

plt.title("Average Advertising Budget by Platform")
plt.xlabel("Advertising Platform")
plt.ylabel("Average Budget")
save_plot("average_advertising_budget.png")


# Visualization 7: Pair plot
pair_plot = sns.pairplot(
    df,
    diag_kind="hist",
    corner=True
)
pair_plot.fig.suptitle(
    "Relationships Between Advertising Features and Sales",
    y=1.02
)
pair_path = os.path.join(VISUALIZATION_DIR, "pairplot.png")
pair_plot.savefig(pair_path, dpi=300, bbox_inches="tight")
plt.close("all")
print("Saved visualization: pairplot.png")


# ============================================================
# 6. FEATURE AND TARGET SEPARATION
# ============================================================

print("\n[5] FEATURE AND TARGET SEPARATION")
print("-" * 65)

X = df[FEATURES]
y = df[TARGET]

print("Feature columns:", FEATURES)
print("Target column:", TARGET)

print("Feature shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

print("\n[6] TRAIN-TEST SPLIT")
print("-" * 65)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])


# ============================================================
# 8. MODEL DEFINITIONS AND HYPERPARAMETER SEARCH
# ============================================================

print("\n[7] MODEL TRAINING AND HYPERPARAMETER TUNING")
print("-" * 65)

# Repeated 5-fold cross-validation
cv = RepeatedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=RANDOM_STATE
)

model_configurations = {
    "Linear Regression": {
        "estimator": LinearRegression(),
        "params": {
            "model__fit_intercept": [True, False]
        },
        "iterations": 2
    },

    "Decision Tree": {
        "estimator": DecisionTreeRegressor(
            random_state=RANDOM_STATE
        ),
        "params": {
            "model__max_depth": [None, 2, 3, 5, 7, 10],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4]
        },
        "iterations": 12
    },

    "Random Forest": {
        "estimator": RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "params": {
            "model__n_estimators": [100, 150, 200, 300],
            "model__max_depth": [None, 3, 5, 8, 12],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": [1.0, "sqrt"]
        },
        "iterations": 12
    },

    "Gradient Boosting": {
        "estimator": GradientBoostingRegressor(
            random_state=RANDOM_STATE
        ),
        "params": {
            "model__n_estimators": [50, 100, 150, 200],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [1, 2, 3],
            "model__min_samples_leaf": [1, 2, 4],
            "model__subsample": [0.8, 1.0]
        },
        "iterations": 12
    }
}

trained_models = {}
model_results = []

for model_name, configuration in model_configurations.items():

    print(f"\nTraining {model_name}...")

    # Scaling is useful for Linear Regression.
    # Tree-based models can also work with scaled numeric features.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", configuration["estimator"])
    ])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=configuration["params"],
        n_iter=configuration["iterations"],
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # Evaluate on the separate test dataset
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    cv_rmse = -search.best_score_

    trained_models[model_name] = best_model

    model_results.append({
        "Model": model_name,
        "Repeated_CV_RMSE": cv_rmse,
        "Test_MAE": mae,
        "Test_RMSE": rmse,
        "Test_R2": r2,
        "Best_Parameters": str(search.best_params_)
    })

    print("Best repeated CV RMSE:", round(cv_rmse, 4))
    print("Test MAE:", round(mae, 4))
    print("Test RMSE:", round(rmse, 4))
    print("Test R2:", round(r2, 4))
    print("Best parameters:", search.best_params_)


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

print("\n[8] MODEL COMPARISON")
print("-" * 65)

results_df = pd.DataFrame(model_results)

results_df.sort_values(
    by="Repeated_CV_RMSE",
    ascending=True,
    inplace=True
)

results_df.reset_index(drop=True, inplace=True)

print(results_df.drop(columns=["Best_Parameters"]).round(4).to_string(index=False))

comparison_path = os.path.join(
    REPORT_DIR,
    "model_comparison.csv"
)

results_df.to_csv(comparison_path, index=False)

print("\nModel comparison report saved:", comparison_path)


# ============================================================
# 10. SELECT BEST MODEL
# ============================================================

print("\n[9] BEST MODEL SELECTION")
print("-" * 65)

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("Selected model:", best_model_name)
print(
    "Selection basis: Lowest repeated cross-validation RMSE"
)

# Final predictions from selected model
y_pred = best_model.predict(X_test)

final_mae = mean_absolute_error(y_test, y_pred)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
final_r2 = r2_score(y_test, y_pred)

print("\nFINAL MODEL PERFORMANCE")
print("-" * 40)
print(f"MAE  : {final_mae:.4f}")
print(f"RMSE : {final_rmse:.4f}")
print(f"R2   : {final_r2:.4f}")
print(f"R2 % : {final_r2 * 100:.2f}%")


# ============================================================
# 11. ACTUAL VS PREDICTED VISUALIZATION
# ============================================================

print("\n[10] ACTUAL VS PREDICTED ANALYSIS")
print("-" * 65)

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.75
)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.title(f"Actual vs Predicted Sales - {best_model_name}")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

save_plot("actual_vs_predicted.png")


# ============================================================
# 12. RESIDUAL ANALYSIS
# ============================================================

print("\n[11] RESIDUAL ANALYSIS")
print("-" * 65)

residuals = y_test - y_pred

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=y_pred,
    y=residuals,
    alpha=0.75
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title("Residual Analysis")
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")

save_plot("residual_analysis.png")


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

print("\n[12] FEATURE IMPORTANCE")
print("-" * 65)

final_estimator = best_model.named_steps["model"]

if hasattr(final_estimator, "feature_importances_"):

    importance_values = final_estimator.feature_importances_

    importance_label = "Feature Importance"

elif hasattr(final_estimator, "coef_"):

    importance_values = np.abs(final_estimator.coef_)

    importance_label = "Absolute Coefficient Magnitude"

else:

    importance_values = None
    importance_label = None


if importance_values is not None:

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": importance_values
    })

    importance_df.sort_values(
        by="Importance",
        ascending=False,
        inplace=True
    )

    print(importance_df.to_string(index=False))

    importance_path = os.path.join(
        REPORT_DIR,
        "feature_importance.csv"
    )

    importance_df.to_csv(importance_path, index=False)

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature"
    )

    plt.title(f"{importance_label} - {best_model_name}")
    plt.xlabel(importance_label)
    plt.ylabel("Advertising Platform")

    save_plot("feature_importance.png")

else:
    importance_df = pd.DataFrame(
        columns=["Feature", "Importance"]
    )

    print("Feature importance is not available for this model.")


# ============================================================
# 14. CUSTOM SALES PREDICTION
# ============================================================

print("\n[13] CUSTOM SALES PREDICTION")
print("-" * 65)

# Example advertising budgets.
# Change these values to test different budgets.
custom_input = pd.DataFrame([{
    "TV": 150.0,
    "Radio": 25.0,
    "Newspaper": 30.0
}])

custom_prediction = best_model.predict(custom_input)[0]

print("Custom advertising budget:")
print(custom_input.to_string(index=False))

print(
    f"\nPredicted Sales: {custom_prediction:.2f}"
)


# ============================================================
# 15. SAVE TRAINED MODEL
# ============================================================

print("\n[14] SAVING TRAINED MODEL")
print("-" * 65)

model_path = os.path.join(
    OUTPUT_DIR,
    "best_sales_prediction_model.pkl"
)

joblib.dump(best_model, model_path)

print("Best model saved:", model_path)


# ============================================================
# 16. SAVE PREDICTION RESULTS
# ============================================================

prediction_df = pd.DataFrame({
    "Actual_Sales": y_test.values,
    "Predicted_Sales": y_pred,
    "Residual": residuals.values
})

prediction_path = os.path.join(
    REPORT_DIR,
    "test_predictions.csv"
)

prediction_df.to_csv(prediction_path, index=False)

print("Test predictions saved:", prediction_path)


# ============================================================
# 17. SAVE KEY INSIGHTS
# ============================================================

correlations = df.corr()[TARGET].drop(TARGET).sort_values(
    ascending=False
)

highest_correlation_feature = correlations.index[0]
highest_correlation_value = correlations.iloc[0]

insights_path = os.path.join(
    REPORT_DIR,
    "key_insights.txt"
)

with open(insights_path, "w", encoding="utf-8") as file:

    file.write("SALES PREDICTION - KEY INSIGHTS\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Total cleaned records: {len(df)}\n")
    file.write(f"Average sales: {df[TARGET].mean():.2f}\n")
    file.write(f"Minimum sales: {df[TARGET].min():.2f}\n")
    file.write(f"Maximum sales: {df[TARGET].max():.2f}\n\n")

    file.write("CORRELATION WITH SALES\n")
    file.write("-" * 40 + "\n")

    for feature, value in correlations.items():
        file.write(f"{feature}: {value:.4f}\n")

    file.write("\n")
    file.write(
        f"Strongest correlated feature: "
        f"{highest_correlation_feature}\n"
    )

    file.write(
        f"Correlation value: {highest_correlation_value:.4f}\n\n"
    )

    file.write("BEST MODEL\n")
    file.write("-" * 40 + "\n")
    file.write(f"Model: {best_model_name}\n")
    file.write(f"MAE: {final_mae:.4f}\n")
    file.write(f"RMSE: {final_rmse:.4f}\n")
    file.write(f"R2: {final_r2:.4f}\n")
    file.write(f"R2 percentage: {final_r2 * 100:.2f}%\n\n")

    file.write("CUSTOM PREDICTION\n")
    file.write("-" * 40 + "\n")
    file.write(
        f"TV: {custom_input.iloc[0]['TV']}\n"
    )
    file.write(
        f"Radio: {custom_input.iloc[0]['Radio']}\n"
    )
    file.write(
        f"Newspaper: {custom_input.iloc[0]['Newspaper']}\n"
    )
    file.write(
        f"Predicted Sales: {custom_prediction:.4f}\n"
    )

print("Key insights saved:", insights_path)


# ============================================================
# 18. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("                  PROJECT COMPLETED")
print("=" * 65)

print("\nDATASET SUMMARY")
print("-" * 40)
print("Total cleaned records:", len(df))
print("Features:", FEATURES)
print("Target:", TARGET)

print("\nMODEL SUMMARY")
print("-" * 40)
print("Models compared:", len(model_configurations))
print("Best model:", best_model_name)
print(f"Test MAE: {final_mae:.4f}")
print(f"Test RMSE: {final_rmse:.4f}")
print(f"Test R2: {final_r2:.4f}")

print("\nOUTPUTS")
print("-" * 40)
print("Cleaned dataset:", cleaned_path)
print("Trained model:", model_path)
print("Model comparison:", comparison_path)
print("Reports folder:", REPORT_DIR)
print("Visualizations folder:", VISUALIZATION_DIR)

print("\nAll tasks executed successfully!")
print("=" * 65)
