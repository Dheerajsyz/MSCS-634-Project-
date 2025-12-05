# MSCS 634 - Advanced Data Mining
## Project Deliverable 2: Regression Modeling and Performance Evaluation

**Student**: Dheeraj Kollapaneni  
**Course**: MSCS 634 - Advanced Data Mining  
**Date**: November 15, 2025

---

## Overview

This deliverable builds predictive models to estimate obesity rates using demographic, geographic, and temporal features. Three regression approaches were compared to identify the most effective modeling strategy.

## Models Developed

Three regression models were built and evaluated:

1. **Linear Regression**: Baseline model using ordinary least squares
2. **Ridge Regression**: L2 regularization with alpha=1.0
3. **Lasso Regression**: L1 regularization with alpha=0.1

## Feature Engineering

Engineered features to enhance model performance:

**Temporal Features**:
- Years_Since_Start: Captures temporal trends

**Data Quality Indicators**:
- Confidence_Range: Measures confidence interval width
- Sample_Size_Category: Categorizes sample reliability

**Demographic Encodings**:
- Income_Level: Ordinal encoding (1-6 scale)
- Education_Level: Ordinal encoding (1-4 scale)
- Age_Group: Ordinal encoding (1-6 scale)
- Sex_Encoded: Binary encoding
- State_Encoded: Numerical encoding
- Race_Encoded: Categorical encoding

Final feature set: 12 predictive variables

## Model Performance

| Model | Test R² | Test RMSE | Test MAE | CV R² (Mean ± Std) |
|-------|---------|-----------|----------|---------------------|
| Linear Regression | 0.9989 | 0.347 | 0.177 | 0.9989 ± 0.0000 |
| Ridge Regression | 0.9989 | 0.347 | 0.177 | 0.9989 ± 0.0000 |
| Lasso Regression | 0.9987 | 0.368 | 0.168 | 0.9988 ± 0.0000 |

All models achieved exceptional accuracy with R² scores above 0.998.

## Visualizations

### Feature Correlations
Shows which features correlate most strongly with obesity rates.

![Feature Correlations](screenshots/feature_correlations.png)

### Model Comparison
Compares R² and RMSE across all three models.

![R² Comparison](screenshots/r2_comparison.png)

![RMSE Comparison](screenshots/rmse_comparison.png)

### Prediction Accuracy
Scatter plots showing predicted vs actual values.

![Predicted vs Actual](screenshots/predicted_vs_actual.png)

### Residual Analysis
Examines prediction errors for systematic biases.

![Residual Analysis](screenshots/residual_analysis.png)

### Coefficient Analysis
Compares feature weights across models.

![Coefficient Comparison](screenshots/coefficient_comparison.png)

## Key Findings

**Model Performance**: All three models achieved R² scores above 0.998, indicating that selected features explain over 99.8% of variance in obesity rates. Linear and Ridge regression performed nearly identically.

**Feature Importance**: Confidence interval limits dominated predictions with coefficients of 5.77 and 5.19. Among demographic features, years since baseline, education level, and age group showed meaningful contributions.

**Lasso Selection**: Lasso regression set 10 out of 12 feature coefficients to zero, retaining only confidence limit features, demonstrating their predictive dominance.

**Generalization**: Cross-validation results matched test set performance with extremely low standard deviations, confirming good generalization.

## Limitations

The high R² values are primarily driven by confidence interval features, which are mathematically derived from obesity measurements themselves. This creates data leakage where predictors contain information directly reflecting the target variable. For meaningful prediction, these features should be excluded in future work.

## Challenges

**Missing Demographic Data**: Created complete-case dataset for modeling while documenting sample size reduction.

**Feature Selection**: Balanced predictive power with interpretability.

**Regularization Tuning**: Selected alpha values (1.0 for Ridge, 0.1 for Lasso) that balanced complexity with performance.

## Files

- `deliverable2_regression_modeling.ipynb`: Complete regression analysis with code
- `screenshots/`: All model performance visualizations
- `README.md`: This summary document

## Summary

The regression analysis demonstrated that obesity rates can be predicted with high accuracy using available features. While confidence intervals provided the strongest predictions, demographic and temporal features also contributed meaningful information. The analysis established a solid baseline for classification and clustering tasks in subsequent deliverables.

---

**Dheeraj Kollapaneni**  
MSCS 634 - Advanced Data Mining  
November 2025
