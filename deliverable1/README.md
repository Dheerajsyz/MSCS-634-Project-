# MSCS 634 - Advanced Data Mining
## Project Deliverable 1: Data Collection, Cleaning, and Exploration

**Student**: Dheeraj Kollapaneni  
**Course**: MSCS 634 - Advanced Data Mining  
**Date**: November 1, 2025

---

## Overview

This deliverable focuses on data collection, cleaning, and exploratory analysis of the CDC BRFSS obesity dataset. The analysis examines obesity patterns across the United States, exploring how rates vary by state, demographic groups, and over time.

## Dataset

**Source**: CDC Behavioral Risk Factor Surveillance System (BRFSS)  
**Original Size**: 106,261 records with 33 attributes  
**Time Period**: 2011 to 2023 (13 years)  
**Coverage**: All 50 states plus DC, Puerto Rico, and US territories

## Data Cleaning Process

The original dataset required significant cleaning:

- **Removed**: 12,755 rows with missing critical data (location, year, or obesity values)
- **No duplicates found**
- **Validation**: All percentage values between 0 and 100
- **Final Dataset**: 93,505 rows

## Key Findings

### Geographic Patterns

- **Highest Obesity**: West Virginia (35.49% average)
- **Lowest Obesity**: District of Columbia (27.88% average)
- **Range**: 7.6 percentage point difference between highest and lowest

![State Obesity Rates](screenshots/state_obesity_rates.png)

### Temporal Trends

- Obesity rates increased from approximately 31% in 2011 to nearly 34% in 2023
- Steady annual increases observed
- Concerning upward trajectory

![Obesity Trends Over Time](screenshots/obesity_trends_over_time.png)

### Demographic Patterns

- Education level shows clear inverse relationship with obesity rates
- Income levels demonstrate similar patterns
- Age groups exhibit varying obesity rates

![Demographic Patterns](screenshots/demographic_patterns.png)

### Data Distribution

The dataset contains diverse question types related to obesity, physical activity, and nutrition:

![Question Types Distribution](screenshots/question_types_distribution.png)

![Data Value Distribution](screenshots/data_value_distribution.png)

## Challenges Addressed

**Missing Data**: Many demographic columns contained missing values because survey questions were not administered uniformly. Rows with essential fields (location, obesity values) missing were removed while preserving optional demographic information.

**Data Validation**: Verified all percentage values fell within valid ranges (0-100) to ensure data integrity.

**Data Structure**: Understood that records represent demographic stratifications, with most demographic fields intentionally blank for aggregate statistics.

## Files

- `deliverable1_obesity_analysis.ipynb`: Complete analysis notebook with code and visualizations
- `screenshots/`: All visualizations generated during analysis
- `README.md`: This summary document

## Summary

The cleaned dataset provides a solid foundation for predictive modeling in subsequent deliverables. Clear patterns emerged showing geographic disparities, temporal trends, and demographic relationships with obesity rates. These insights will guide feature engineering and model development in future phases.

---

**Dheeraj Kollapaneni**  
MSCS 634 - Advanced Data Mining  
November 2025
