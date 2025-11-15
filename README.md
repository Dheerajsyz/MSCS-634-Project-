# MSCS 634 - Advanced Data Mining Project


## Project Overview

This project analyzes obesity and health data from the CDC to understand patterns in obesity rates across the United States and build predictive models. The dataset comes from the Behavioral Risk Factor Surveillance System (BRFSS), which is a nationwide health survey.

The project has multiple phases:
- **Deliverable 1**: Data collection, cleaning, and exploratory analysis
- **Deliverable 2**: Regression modeling and performance evaluation
- **Future Deliverables**: Classification, clustering, and association rule mining

## About the Dataset

I'm using the "Nutrition, Physical Activity, and Obesity" dataset from the CDC's BRFSS program.

- **Size**: 106,261 records with 33 attributes
- **Time period**: 2011 to 2023 (13 years of data)
- **Coverage**: All 50 states plus DC, Puerto Rico, and US territories

### What's in the Data

The main columns include:
- Years covered (2011-2023)
- State names and locations
- Obesity percentages and other health measurements
- Demographic breakdowns (age, education, income, sex, race/ethnicity)
- Statistical info like confidence intervals and sample sizes

### Why This Dataset Was Selected

This dataset was chosen for several reasons:

First, it meets all the requirements with over 100,000 records and 33 attributes, substantially exceeding the minimum specifications.

Second, it represents real public health data about a critical issue. Obesity affects over 40% of adults in the United States, making this analysis relevant beyond academic purposes.

Third, the data offers variety with numeric columns like obesity rates, categorical data including states and demographics, and temporal data spanning 13 years. This variety supports all project deliverables including regression, classification, clustering, and association rules.

Finally, the dataset presents realistic challenges with missing values and data quality issues, providing opportunities to demonstrate proper data preprocessing techniques.

## Project Structure

```
MSCS-634-Project-/
├── data/
│   ├── Nutrition__Physical_Activity__and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System.csv
│   └── obesity_data_cleaned.csv
├── notebooks/
│   ├── deliverable1_obesity_analysis.ipynb
│   └── deliverable2_regression_modeling.ipynb
├── screenshots/
│   ├── data_value_distribution.png
│   ├── demographic_patterns.png
│   ├── obesity_trends_over_time.png
│   ├── question_types_distribution.png
│   ├── state_obesity_rates.png
│   ├── feature_correlations.png
│   ├── r2_comparison.png
│   ├── rmse_comparison.png
│   ├── predicted_vs_actual.png
│   ├── residual_analysis.png
│   └── coefficient_comparison.png
├── requirements.txt
└── README.md
```

## Deliverable 1: Data Collection, Cleaning, and Exploration

### Data Cleaning Process

The original dataset had 106,260 records. After cleaning:
- Removed 12,755 rows that were missing critical data (location, year, or the actual obesity value)
- No duplicate records found
- All percentage values are valid (between 0 and 100)
- Final cleaned dataset: 93,505 rows

### Key Insights from Exploratory Analysis

**Geographic Patterns**:
- West Virginia shows the highest average obesity rates at 35.49%
- District of Columbia has the lowest at 27.88%
- A 7.6 percentage point difference exists between highest and lowest states

**Trends Over Time**:
- Obesity rates have increased from approximately 31% in 2011 to nearly 34% in 2023
- The trend shows steady annual increases

**Demographic Patterns**:
- Education level demonstrates clear inverse relationships with obesity rates
- Income levels show similar patterns with lower income associated with higher obesity
- Different age groups exhibit varying obesity rates

### Challenges Addressed

The primary challenge involved handling missing data. Many demographic columns contained missing values because survey questions were not administered uniformly across all respondents. The approach taken was to retain rows where optional demographic information was missing while removing rows where essential fields like location or obesity values were absent. This preserved the maximum amount of usable data while maintaining data quality.

Another consideration was validating that all percentage values fell within the valid range of 0 to 100, ensuring data integrity.

---

## Deliverable 2: Regression Modeling and Performance Evaluation

### Modeling Approach

This phase built predictive models to estimate obesity rates using demographic, geographic, and temporal features. The analysis compared three regression approaches to identify the most effective modeling strategy.

### Feature Engineering

Several engineered features were created to enhance model performance:

**Temporal Features**:
- Years_Since_Start: Captures temporal trends from the baseline year

**Data Quality Indicators**:
- Confidence_Range: Measures the width of confidence intervals as a reliability indicator
- Sample_Size_Category: Categorizes sample sizes into reliability tiers

**Demographic Encodings**:
- Income_Level: Ordinal encoding of income brackets (1-6 scale)
- Education_Level: Ordinal encoding of education levels (1-4 scale)
- Age_Group: Ordinal encoding of age categories (1-6 scale)
- Sex_Encoded: Binary encoding for gender
- State_Encoded: Numerical encoding for geographic location
- Race_Encoded: Categorical encoding for race and ethnicity

The final feature set included 12 predictive variables after encoding and engineering, with the target variable being Data_Value (obesity percentage).

### Models Developed

Three regression models were built and compared:

**1. Linear Regression**
- Serves as the baseline model
- Uses ordinary least squares to fit a linear relationship
- No regularization applied

**2. Ridge Regression**
- Applies L2 regularization with alpha=1.0
- Reduces overfitting by penalizing large coefficients
- Keeps all features but shrinks their impact

**3. Lasso Regression**
- Applies L1 regularization with alpha=0.1
- Can perform automatic feature selection by setting some coefficients to zero
- Provides more interpretable models through sparsity

### Model Evaluation Results

All models were evaluated using multiple metrics on both training and test sets:

| Model | Test R² | Test RMSE | Test MAE | CV R² (Mean ± Std) |
|-------|---------|-----------|----------|---------------------|
| Linear Regression | 0.9989 | 0.347 | 0.177 | 0.9989 ± 0.0000 |
| Ridge Regression | 0.9989 | 0.347 | 0.177 | 0.9989 ± 0.0000 |
| Lasso Regression | 0.9987 | 0.368 | 0.168 | 0.9988 ± 0.0000 |

**Evaluation Metrics Used**:
- **R-squared (R²)**: Proportion of variance in obesity rates explained by the model
- **Root Mean Squared Error (RMSE)**: Average prediction error in percentage points
- **Mean Absolute Error (MAE)**: Average absolute deviation from actual values
- **Cross-Validation R²**: 5-fold cross-validation scores with standard deviation

### Cross-Validation Analysis

Five-fold cross-validation was performed to assess model generalization:
- Each model was trained on 80% of data and tested on 20%
- The process repeated 5 times with different splits
- Consistent performance across folds indicates good generalization
- Low standard deviation suggests stable predictions

### Key Findings

**Model Performance**:
All three models achieved exceptionally high predictive accuracy with R² scores above 0.998, indicating that the selected features explain over 99.8% of the variance in obesity rates. Linear and Ridge regression performed nearly identically (Test R² = 0.9989, RMSE = 0.347), while Lasso regression showed slightly lower but still excellent performance (Test R² = 0.9987, RMSE = 0.368). The minimal difference between training and testing scores demonstrates that overfitting is not a concern.

**Feature Importance**:
The confidence interval limits (Low_Confidence_Limit and High_Confidence_Limit) dominated the predictions with coefficients of 5.77 and 5.19 respectively. This strong relationship exists because these intervals are calculated directly from the obesity measurements themselves, making them highly correlated with the target variable. Among the demographic and temporal features, years since baseline, education level, and age group showed meaningful contributions to predictions.

**Lasso Feature Selection**:
Lasso regression performed automatic feature selection by setting 10 out of 12 feature coefficients to zero, retaining only the confidence limit features. This demonstrates that while demographic features add marginal predictive value, the confidence intervals alone capture most of the information needed for accurate predictions.

**Model Generalization**:
Cross-validation results closely matched test set performance across all models. The extremely low standard deviations (< 0.0001 for R²) indicate highly consistent performance across different data subsets. This stability confirms that the models generalize well and are not sensitive to particular data splits.

### Visualizations Created

Six visualizations were generated to analyze model performance (all saved in `screenshots/` folder):

1. **Feature Correlations** (`feature_correlations.png`): Shows which features correlate most strongly with obesity rates. Low and High Confidence Limits show the strongest positive correlations (>0.94), while demographic features show weaker relationships.

   ![Feature Correlations](screenshots/feature_correlations.png)

2. **R² Comparison** (`r2_comparison.png`): Compares R-squared scores across training, testing, and cross-validation. All three models achieve nearly identical performance with R² values above 0.998.

   ![R² Comparison](screenshots/r2_comparison.png)

3. **RMSE Comparison** (`rmse_comparison.png`): Compares prediction errors across all models. Linear and Ridge show identical RMSE (~0.34), while Lasso is slightly higher (~0.36).

   ![RMSE Comparison](screenshots/rmse_comparison.png)

4. **Predicted vs Actual** (`predicted_vs_actual.png`): Scatter plots showing prediction accuracy for each model. Points closely follow the diagonal line, indicating excellent predictions across all models.

   ![Predicted vs Actual](screenshots/predicted_vs_actual.png)

5. **Residual Analysis** (`residual_analysis.png`): Examines prediction errors to identify systematic biases. Residuals are centered around zero with normal distribution, confirming no systematic bias.

   ![Residual Analysis](screenshots/residual_analysis.png)

6. **Coefficient Comparison** (`coefficient_comparison.png`): Compares feature weights across all three models. Lasso sets most coefficients to zero, retaining only confidence limit features.

   ![Coefficient Comparison](screenshots/coefficient_comparison.png)

### Limitations and Considerations

The extremely high R-squared values (>0.998) are primarily driven by the inclusion of confidence interval features, which are mathematically derived from the obesity measurements themselves. This creates a form of data leakage where the predictors contain information that directly reflects the target variable. While this demonstrates technical proficiency in building accurate models, it limits the practical utility for true prediction scenarios.

For meaningful predictive modeling in future work, the confidence interval features should be excluded, forcing the models to rely on genuinely independent predictors such as demographics, geography, and temporal trends. This would provide more realistic performance metrics and reveal which external factors truly drive obesity rates.

The data structure presents another consideration: each record represents a specific demographic stratification (by income, education, age, or sex), with most demographic fields left blank. The approach of filling missing values with zeros (indicating no stratification) preserves all data but may introduce artifacts. Alternative approaches could include:
- Creating separate models for each stratification type
- Using one-hot encoding for stratification categories
- Building hierarchical models that account for the stratified structure

Additional factors not captured in this dataset that likely influence obesity rates include:
- Food environment characteristics (restaurant density, grocery store access, food desert indicators)
- Built environment factors (walkability scores, parks and recreation facilities, urban design)
- Healthcare access and utilization patterns
- Local policy interventions (nutrition programs, physical activity initiatives)
- Economic indicators beyond individual income (unemployment rates, community wealth)
- Cultural and behavioral factors specific to communities

### Challenges Encountered

**Missing Demographic Data**: Many records lacked complete demographic information. The solution involved creating a complete-case dataset for modeling while documenting the reduction in sample size. This approach ensures model reliability while acknowledging potential selection bias.

**Feature Selection**: Determining which features to include required balancing predictive power with interpretability. The approach taken prioritized features with clear theoretical relationships to obesity while testing their empirical contribution.

**Regularization Parameter Tuning**: Selecting appropriate alpha values for Ridge and Lasso required experimentation. The values chosen (alpha=1.0 for Ridge, alpha=0.1 for Lasso) balanced model complexity with performance.

### Recommendations for Future Work

1. **Feature Engineering**: Explore polynomial features and interaction terms to capture non-linear relationships
2. **Advanced Models**: Test ensemble methods like Random Forest or Gradient Boosting
3. **External Data Integration**: Incorporate food environment data, walkability scores, or healthcare access metrics
4. **Hyperparameter Optimization**: Use grid search or cross-validation to optimize regularization parameters
5. **Feature Selection**: Apply recursive feature elimination to identify minimal feature sets
6. **Temporal Analysis**: Build separate models for different time periods to capture changing relationships

---


## How to Run the Analysis

To run this analysis:

1. Ensure Python 3.8 or higher is installed

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run Deliverable 1 notebook:
   ```bash
   jupyter notebook notebooks/deliverable1_obesity_analysis.ipynb
   ```

4. Run Deliverable 2 notebook:
   ```bash
   jupyter notebook notebooks/deliverable2_regression_modeling.ipynb
   ```

5. Execute all cells sequentially from top to bottom

## Required Packages

All necessary packages are listed in `requirements.txt`:

- pandas: Data manipulation and analysis
- numpy: Numerical computations
- matplotlib: Visualization creation
- seaborn: Statistical visualization
- scikit-learn: Machine learning algorithms and metrics

Install all dependencies with:
```bash
pip install -r requirements.txt
```
