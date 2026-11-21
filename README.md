# Sales Prediction Using Machine Learning

## Project Overview

This project predicts product sales based on advertising expenditure across three platforms: TV, Radio, and Newspaper.

The project uses supervised machine learning regression algorithms to understand the relationship between advertising budgets and sales and identify the best-performing prediction model.

## Dataset

- Dataset: Advertising Dataset
- Original records: 200
- Features: 3
- Target variable: Sales
- Missing values: 0
- Duplicate records: 0

### Features

| Feature | Description |
|---|---|
| TV | TV advertising budget |
| Radio | Radio advertising budget |
| Newspaper | Newspaper advertising budget |
| Sales | Product sales (target variable) |

The unnecessary `Unnamed: 0` index column was removed during preprocessing.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Project Workflow

1. Load the advertising dataset.
2. Perform data cleaning and preprocessing.
3. Conduct Exploratory Data Analysis (EDA).
4. Analyze relationships between advertising budgets and sales.
5. Create data visualizations and a correlation heatmap.
6. Separate features and target variable.
7. Split the dataset into training and testing sets.
8. Train and compare four regression models.
9. Perform repeated cross-validation and hyperparameter tuning.
10. Evaluate model performance using MAE, RMSE, and R².
11. Select the best-performing model.
12. Analyze feature importance and prediction residuals.
13. Generate custom sales predictions.
14. Save the trained model and reports.

## Machine Learning Models

The following regression algorithms were implemented:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor

Repeated 5-fold cross-validation with 3 repeats and randomized hyperparameter tuning were used for model selection.

## Model Performance

Results from the executed experiment:

| Model | Repeated CV RMSE | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| Gradient Boosting | 0.8190 | 0.5584 | 0.6916 | 98.48% |
| Random Forest | 0.9225 | 0.6754 | 0.8163 | 97.89% |
| Decision Tree | 1.3065 | 1.1088 | 1.5719 | 92.17% |
| Linear Regression | 1.6669 | 1.4608 | 1.7816 | 89.94% |

### Best Model: Gradient Boosting Regressor

- Test MAE: 0.5584
- Test RMSE: 0.6916
- Test R²: 0.9848 (98.48%)

Gradient Boosting achieved the lowest repeated cross-validation RMSE and the best test-set performance in this experiment.

## Feature Importance

Feature importance from the selected model:

| Feature | Importance |
|---|---:|
| TV | 59.06% |
| Radio | 40.68% |
| Newspaper | 0.26% |

TV advertising had the highest feature importance, followed by Radio advertising.

These values describe the fitted model and should not be interpreted as proof of causation.

## Custom Prediction

Example advertising budget:

| Platform | Budget |
|---|---:|
| TV | 150 |
| Radio | 25 |
| Newspaper | 30 |

**Predicted Sales: 14.96**

## Visualizations

The project generates 10 visualizations:

- Sales distribution
- TV advertising vs Sales
- Radio advertising vs Sales
- Newspaper advertising vs Sales
- Correlation heatmap
- Average advertising budget
- Pair plot
- Actual vs Predicted Sales
- Residual analysis
- Feature importance

## Project Structure

```text
Task4_Sales_Prediction/
│
├── Dataset/
│   └── advertising.csv
│
├── outputs/
│   ├── visualizations/
│   ├── reports/
│   ├── cleaned_advertising_data.csv
│   └── best_sales_prediction_model.pkl
│
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/sagarbhovi/CodeAlpha_Sales_Prediction.git
```

### 2. Navigate to the project folder

```bash
cd CodeAlpha_Sales_Prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python main.py
```

## Evaluation Metrics

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R² Score:** Coefficient of Determination

## Conclusion

This project demonstrates how machine learning regression algorithms can be applied to sales prediction using advertising data.

Four regression models were trained and evaluated. Gradient Boosting achieved the best performance, with an R² score of 98.48% on the held-out test set.

The project also demonstrates data preprocessing, exploratory data analysis, visualization, model tuning, evaluation, and prediction.

## Internship

**CodeAlpha Data Science Internship – Task 4: Sales Prediction**
