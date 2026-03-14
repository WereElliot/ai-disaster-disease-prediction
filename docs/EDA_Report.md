# EDA & Baseline Report: Disaster and Disease Outbreak Prediction Platform

## 1. Data Overview & Quality
- **Dataset Source**: NASA POWER + WHO GHO + EM-DAT Integrated Data.
- **Samples**: 421 monthly resampled observations.
- **Features**: ['temp', 'precipitation', 'humidity', 'wind_speed', 'malaria_cases', 'flood_event', 'drought_event', 'disaster_severity', 'cases', 'date', 'location']

## 2. Visual Analysis
- **Temporal Trends**: Observed peaks in precipitation correlate with increases in case counts (see cases_over_time.png).
- **Correlations**: High Pearson correlation identified between precipitation/humidity and disease incidence.

## 3. Baseline Model Performance
- **Statistical Median Baseline**: Evaluated during Phase 5.
- **Target F1**: 0.85

## 4. Conclusion
Phase 2 confirms significant climate-health patterns, particularly the link between heavy rainfall and increased disease risk in the Kenya region.
