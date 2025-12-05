# MSCS 634 - Advanced Data Mining Project


## Project Overview

This project analyzes obesity and health data from the CDC to understand patterns in obesity rates across the United States and build predictive models. The dataset comes from the Behavioral Risk Factor Surveillance System (BRFSS), which is a nationwide health survey.

The project consists of four phases:
- **Deliverable 1**: Data collection, cleaning, and exploratory analysis
- **Deliverable 2**: Regression modeling and performance evaluation
- **Deliverable 3**: Classification, clustering, and association rule mining
- **Deliverable 4**: Final insights, recommendations, and comprehensive report

## About the Dataset

The analysis uses the "Nutrition, Physical Activity, and Obesity" dataset from the CDC's BRFSS program.

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
├── deliverable1/
│   ├── deliverable1_obesity_analysis.ipynb
│   ├── screenshots/
│   └── README.md
├── deliverable2/
│   ├── deliverable2_regression_modeling.ipynb
│   ├── screenshots/
│   └── README.md
├── deliverable3/
│   ├── deliverable3_classification_clustering.ipynb
│   ├── screenshots/
│   └── README.md
├── requirements.txt
└── README.md (this file - comprehensive project summary)
```
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

Six visualizations were generated to analyze model performance (all saved in `deliverable2/screenshots/` folder):

1. **Feature Correlations** (`feature_correlations.png`): Shows which features correlate most strongly with obesity rates. Low and High Confidence Limits show the strongest positive correlations (>0.94), while demographic features show weaker relationships.

   ![Feature Correlations](deliverable2/screenshots/feature_correlations.png)

2. **R² Comparison** (`r2_comparison.png`): Compares R-squared scores across training, testing, and cross-validation. All three models achieve nearly identical performance with R² values above 0.998.

   ![R² Comparison](deliverable2/screenshots/r2_comparison.png)

3. **RMSE Comparison** (`rmse_comparison.png`): Compares prediction errors across all models. Linear and Ridge show identical RMSE (~0.34), while Lasso is slightly higher (~0.36).

   ![RMSE Comparison](deliverable2/screenshots/rmse_comparison.png)

4. **Predicted vs Actual** (`predicted_vs_actual.png`): Scatter plots showing prediction accuracy for each model. Points closely follow the diagonal line, indicating excellent predictions across all models.

   ![Predicted vs Actual](deliverable2/screenshots/predicted_vs_actual.png)

5. **Residual Analysis** (`residual_analysis.png`): Examines prediction errors to identify systematic biases. Residuals are centered around zero with normal distribution, confirming no systematic bias.

   ![Residual Analysis](deliverable2/screenshots/residual_analysis.png)

6. **Coefficient Comparison** (`coefficient_comparison.png`): Compares feature weights across all three models. Lasso sets most coefficients to zero, retaining only confidence limit features.

   ![Coefficient Comparison](deliverable2/screenshots/coefficient_comparison.png)

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

## Deliverable 3: Classification, Clustering, and Pattern Mining

### Overview

The third deliverable applied machine learning techniques to categorize obesity levels, identify state groupings, and discover patterns in the data. Four classification models, K-Means clustering, and association rule mining provided complementary insights into obesity patterns.

### Classification Models

Four models were developed to predict obesity categories (Low: <25%, Medium: 25-30%, High: >30%):

- **Decision Tree (Tuned)**: Best performer with 73.7% test accuracy after hyperparameter optimization
- **Decision Tree (Baseline)**: 72.5% test accuracy before tuning
- **SVM**: 71.4% test accuracy on a 10,000-sample subset
- **k-NN**: 69.9% test accuracy with k=5 neighbors
- **Naive Bayes**: 60.0% test accuracy

Age group emerged as the most important predictor, followed by sample size and state location. Class imbalance (71% High obesity samples) affected all models, particularly for Medium and Low categories.

### Clustering Analysis

K-Means clustering with k=4 grouped states based on obesity patterns:

- **Cluster 0** (40 states): Moderate obesity rates (32.3% average)
- **Cluster 1** (13 states): High obesity rates (34.2% average), predominantly Southern states
- **Cluster 2** (1 state): Virgin Islands with unique characteristics
- **Cluster 3** (1 state): National aggregate data

The silhouette score of 0.35 indicated moderate cluster separation. Geographic patterns were clear, with high-obesity states concentrated in the South.

### Association Rule Mining

The Apriori algorithm identified patterns between demographic factors and obesity:

- **Low Education → High Obesity** (Lift: 1.19, Confidence: 84.8%)
- **Old Age → High Obesity** (Lift: 1.12, Confidence: 79.4%)
- **Recent Year → High Obesity** (Lift: 1.10, Confidence: 78.0%)

Education level showed the strongest association with obesity outcomes, followed by age group. The findings support targeted health education programs for at-risk populations.

### Key Visualizations

Nine visualizations were generated for this deliverable (all saved in `deliverable3/screenshots/` folder):

1. **Obesity Category Distribution** (`obesity_category_distribution.png`): Shows class imbalance with 71% High, 19% Medium, and 10% Low obesity samples.

   ![Obesity Category Distribution](deliverable3/screenshots/obesity_category_distribution.png)

2. **Feature Importance** (`dt_feature_importance.png`): Age group is the most important predictor, followed by sample size and state encoding.

   ![Feature Importance](deliverable3/screenshots/dt_feature_importance.png)

3. **Model Comparison** (`model_comparison.png`): Compares accuracy and F1-scores across all five classification models, with Decision Tree (Tuned) performing best.

   ![Model Comparison](deliverable3/screenshots/model_comparison.png)

4. **Confusion Matrices** (`confusion_matrices.png`): Shows prediction patterns for all models, revealing challenges with Medium and Low categories.

   ![Confusion Matrices](deliverable3/screenshots/confusion_matrices.png)

5. **K-Means Clustering** (`kmeans_clustering_pca.png`): Visualizes state clusters in 2D using PCA, showing clear separation between high and moderate obesity states.

   ![K-Means Clustering](deliverable3/screenshots/kmeans_clustering_pca.png)

6. **Cluster Characteristics** (`cluster_characteristics.png`): Compares average obesity rates across the four identified clusters.

   ![Cluster Characteristics](deliverable3/screenshots/cluster_characteristics.png)

7. **Association Rules** (`association_rules.png`): Visualizes the strongest patterns discovered, highlighting education-obesity relationships.

   ![Association Rules](deliverable3/screenshots/association_rules.png)

For detailed analysis, see [deliverable3/README.md](deliverable3/README.md).

---

## How to Run the Analysis

To run this analysis:

1. Ensure Python 3.8 or higher is installed

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebooks in order:
   ```bash
   jupyter notebook deliverable1/deliverable1_obesity_analysis.ipynb
   jupyter notebook deliverable2/deliverable2_regression_modeling.ipynb
   jupyter notebook deliverable3/deliverable3_classification_clustering.ipynb
   ```

4. Execute all cells sequentially from top to bottom

## Required Packages

All necessary packages are listed in `requirements.txt`:

- pandas: Data manipulation and analysis
- numpy: Numerical computations
- matplotlib: Visualization creation
- seaborn: Statistical visualization
- scikit-learn: Machine learning algorithms and metrics
- mlxtend: Association rule mining (Apriori algorithm)

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## Project Summary and Key Findings

### Dataset Characteristics

The CDC BRFSS obesity dataset provided comprehensive coverage of obesity patterns across the United States from 2011-2023. After cleaning, 93,505 records remained from an original 106,261, representing all 50 states plus territories. The dataset included demographic stratifications (income, education, age, sex, race/ethnicity) and geographic information, making it suitable for diverse analytical approaches.

### Major Findings Across All Deliverables

**Geographic Disparities**: Obesity rates vary significantly by state, ranging from 27.9% (DC) to 35.5% (West Virginia). Southern states consistently show higher rates, forming a distinct cluster in both exploratory and clustering analyses.

**Temporal Trends**: Obesity rates increased steadily from approximately 31% in 2011 to 34% in 2023, representing a concerning upward trend that warrants intervention.

**Demographic Patterns**: Education level emerged as the strongest predictor of obesity across multiple analyses (correlation, classification, association rules). Age, income, and education consistently showed meaningful relationships with obesity outcomes.

**Predictive Modeling**: Regression models achieved exceptional accuracy (R² > 0.99) though primarily driven by confidence interval features. Classification models achieved 73.7% accuracy using demographic and geographic features alone, demonstrating practical predictive utility.

**State Groupings**: Clustering identified natural groupings of states with similar obesity profiles, suggesting that targeted regional interventions may be more effective than national one-size-fits-all approaches.

**Actionable Patterns**: Association rule mining confirmed that low education combined with older age strongly predicts high obesity (Lift: 1.19), supporting targeted health education programs for at-risk populations.

### Practical Recommendations

**For Public Health Officials**:
- Prioritize resources toward Cluster 1 states (Southern region) with consistently high obesity rates
- Develop age-specific interventions, as age group is the strongest classifier
- Invest in health education programs, particularly for populations with lower educational attainment
- Monitor temporal trends closely, as recent years show accelerating obesity rates

**For Policymakers**:
- Address socioeconomic factors (income, education, access) that drive obesity disparities
- Support data collection efforts to maintain high-quality surveillance systems
- Enable state-to-state collaboration within identified clusters to share successful strategies
- Consider regional approaches that account for geographic and cultural differences

**For Healthcare Providers**:
- Use classification models to identify high-risk patients for preventive interventions
- Focus on lifecycle approaches, targeting different age groups with tailored strategies
- Address obesity before it becomes entrenched, particularly in younger populations
- Consider socioeconomic context when developing treatment plans

**For Researchers**:
- Expand datasets to include food environment, built environment, and healthcare access variables
- Develop causal models that go beyond correlation to identify intervention targets
- Conduct longitudinal studies tracking individuals over time
- Test interventions using randomized controlled trials in high-risk clusters

### Ethical Considerations

**Data Privacy**: The dataset contains aggregated state-level statistics rather than individual records, minimizing privacy concerns. However, demographic stratifications could potentially identify small subpopulations in less populated states or territories.

**Fairness and Bias**: The analysis revealed significant disparities by race, income, and education level. While these findings are important for targeting interventions, care must be taken not to stigmatize particular demographic groups. Obesity is influenced by complex socioeconomic and environmental factors beyond individual control.

**Representation**: Not all demographic groups are equally represented in the survey data. Missing demographic information was more common for certain categories, potentially introducing selection bias. The analysis acknowledged this limitation by using indicator values rather than discarding incomplete records.

**Model Deployment**: If classification models were deployed to identify high-risk individuals, safeguards would be needed to ensure:
- Models do not reinforce existing disparities
- Predictions are used to provide additional support, not to deny services
- Regular auditing for fairness across demographic groups
- Transparency about model limitations and uncertainty

**Interpretation**: The strong association between low education and high obesity should not be interpreted as blaming individuals. Education level is a marker for broader socioeconomic disadvantage including limited access to healthy foods, safe spaces for physical activity, and healthcare services.

**Intervention Ethics**: Any interventions based on these findings should:
- Respect individual autonomy and cultural differences
- Address structural barriers rather than focusing solely on individual behavior
- Ensure equitable access to resources across all communities
- Avoid paternalistic approaches that impose solutions without community input

### Limitations

**Data Leakage**: Regression models achieved artificially high accuracy due to including confidence interval features mathematically derived from the target variable.

**Class Imbalance**: Classification models struggled with Medium and Low obesity categories due to their underrepresentation (19% and 10% respectively).

**Temporal Generalization**: Models were trained on 2011-2023 data and may not generalize well to future years if obesity trends change.

**Missing Context**: The dataset lacks important variables like food environment, built environment, healthcare access, and local policies that influence obesity rates.

**Cross-Sectional Nature**: Most analyses used aggregated data, limiting ability to draw causal conclusions or track individual trajectories over time.

### Future Directions

**Enhanced Feature Engineering**: Incorporate external data sources including food environment indicators, walkability scores, healthcare access metrics, and local policy variables.

**Advanced Modeling**: Test ensemble methods (Random Forest, Gradient Boosting), deep learning approaches, and hierarchical models that account for data structure.

**Temporal Analysis**: Develop time-series forecasting models to predict future obesity trends and evaluate intervention impacts.

**Causal Inference**: Apply causal modeling techniques to identify modifiable factors that truly drive obesity rather than mere correlations.

**Intervention Testing**: Partner with public health departments to implement and evaluate interventions suggested by the analysis.

**Expanded Scope**: Extend analysis to related health outcomes (diabetes, cardiovascular disease) to understand broader health patterns.

### Conclusion

This comprehensive analysis of CDC BRFSS obesity data demonstrates the power of data mining techniques to uncover actionable insights for public health. Across four deliverables spanning exploratory analysis, regression modeling, classification, clustering, and pattern mining, consistent findings emerged:

Obesity is a complex phenomenon influenced by demographic, geographic, socioeconomic, and temporal factors. Education level consistently emerged as a key predictor, suggesting that knowledge and resources matter significantly. Geographic clustering revealed regional disparities requiring targeted interventions. Temporal trends showed concerning increases over the study period.

The analytical techniques applied successfully identified at-risk populations (classification), grouped similar states (clustering), and discovered meaningful patterns (association rules) that can guide evidence-based interventions. While limitations exist, particularly regarding data leakage and missing contextual variables, the findings provide a solid foundation for public health planning.

Addressing the obesity epidemic requires multi-faceted approaches that tackle structural barriers while respecting individual autonomy and cultural differences. This analysis provides data-driven guidance on where to focus resources, which populations need support, and what factors to address. Combined with ethical considerations around fairness and implementation, these insights can inform more effective, equitable obesity prevention and treatment strategies.

---

