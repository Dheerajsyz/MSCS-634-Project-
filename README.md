# MSCS 634 - Project Deliverable 1


## What This Project Is About

For this first deliverable, I'm working with obesity and health data from the CDC to understand patterns in obesity rates across the United States. The dataset comes from the Behavioral Risk Factor Surveillance System (BRFSS), which is a nationwide health survey.

The goal is to collect the data, clean it up, and do some exploratory analysis to see what patterns exist before I start building models in the later deliverables.

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

### Why I Chose This Dataset

I picked this dataset for a few reasons:

First, it meets all the requirements - over 100,000 records is way more than the 500 minimum, and 33 attributes is more than the 8-10 needed.

Second, it's real public health data about a serious issue. Obesity affects over 40% of US adults, so understanding the patterns could actually be useful beyond just this assignment.

Third, the data has good variety. There are numeric columns like obesity rates, categorical data like states and demographics, and time data spanning 13 years. This will work for all the future deliverables - regression, classification, clustering, and association rules.

Finally, the dataset has some realistic challenges with missing values and data that needs cleaning, which gives me a chance to show proper data preprocessing techniques

## Project Files

```
proj1/
├── data/
│   ├── Nutrition__Physical_Activity__and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System.csv
│   └── obesity_data_cleaned.csv
├── notebooks/
│   └── deliverable1_obesity_analysis.ipynb
├── visualizations/
│   ├── data_value_distribution.png
│   ├── demographic_patterns.png
│   ├── obesity_trends_over_time.png
│   ├── question_types_distribution.png
│   └── state_obesity_rates.png
├── requirements.txt
└── README.md
```

## What I Found

### Data Cleaning

The original dataset had 106,260 records. After cleaning:
- Removed 12,755 rows that were missing critical data (location, year, or the actual obesity value)
- No duplicate records found
- All percentage values are valid (between 0 and 100)
- Final cleaned dataset: 93,505 rows

### Main Insights

**Geographic Patterns**:
- West Virginia has the highest average obesity rates (35.49%)
- District of Columbia has the lowest (27.88%)
- About 7.6 percentage point difference between highest and lowest

**Trends Over Time**:
- Obesity rates have been increasing - from around 31% in 2011 to nearly 34% in 2023
- The trend is fairly steady, increasing a bit each year

**Demographic Patterns**:
- Education level shows clear patterns - higher education correlates with lower obesity rates
- Income levels also show relationships with obesity
- Age groups have different obesity rates

### Challenges I Ran Into

The biggest challenge was dealing with missing data. A lot of the demographic columns had missing values because not all survey questions were asked to everyone. I had to decide what was "critical" data versus "optional" data. I ended up keeping rows where the optional demographic info was missing since I could still use the other information, but I removed rows where essential fields like the location or the actual data value were missing.

Another thing I had to be careful about was making sure all the percentage values made sense - checking that they were actually between 0 and 100.


## How to Run the Analysis

If you want to run this yourself:

1. Make sure you have Python 3.8 or higher installed

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/deliverable1_obesity_analysis.ipynb
   ```

4. Run all the cells from top to bottom

## Required Packages

- pandas (for data manipulation)
- numpy (for numerical operations)
- matplotlib (for visualizations)
- seaborn (for statistical plots)
- scikit-learn (for data preprocessing)

See `requirements.txt` for specific versions.
