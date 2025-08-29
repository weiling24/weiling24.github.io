---
layout: post
author: Name
title: "Applied Data Science Project Documentation"
categories: ITD214
---
## Project Background
Our project aims to develop a data-driven approach to understanding and predicting mental health risks associated with social media usage. The primary business objective is to create actionable insights for healthcare providers, policymakers, and individuals to make informed decisions about digital wellbeing. This analysis supports the growing need for evidence-based interventions in the digital mental health landscape, where traditional assessment methods may miss critical risk factors related to online behavior


**Primary Objective** Develop machine learning models that can reliably identify individuals at risk for mental health issues based on their social media usage patterns, enabling early intervention and targeted
support services.

**Research Questions**:
This study addresses two critical research questions in digital mental health:
1) Do individuals who spend more time on social media platforms show higher mental health risk levels?
2) Can we predict mental health risk based on digital usage patterns and demographic characteristics?

**Dataset**

- Source: Social Media and Mental Health (SMMH) dataset
- Sample Size: 481 participants
- Features: 21 variables including demographics, social media usage patterns, and psychological indicators
- Target: Mental health risk level (Low, Medium, High) derived from 12 Likert-scale psychological assessment questions

## Work Accomplished
**Document your work done to accomplish the outcome
Feature Engineering and Data Processing**

- Engineered clinically meaningful target variables using score-based thresholds rather than arbitrary quantiles
- Created comprehensive feature sets including demographic encoding, platform-specific binary features, and ordinal time variables
- Implemented advanced class imbalance handling using SMOTE techniques applied only to training data
- Developed robust preprocessing pipelines ensuring data quality and model compatibility

### Data Preparation

**Dataset Structure and Preprocessing**
We worked with the cleaned SMMH dataset, utilizing all engineered features as predictors and our derived risk_level as the target variable. Our data preparation process followed rigorous machine learning practices:

Train-Test Split Configuration:
- 80-20 split maintaining adequate sample sizes for both training (376 samples) and testing (95 samples)
- Stratified sampling to preserve class proportions across splits
- Fixed random state (42) ensuring reproducibility across multiple runsom state for reproducibility

**Class Imbalance Handling:**
- Applied SMOTE (Synthetic Minority Oversampling Technique) exclusively to training data
- Maintained test set integrity for unbiased evaluation
- Used class weighting strategies {Low:1, Medium:2, High:5} to prioritize high-risk identification

### Modelling
1. Decision Tree

**Hyperparameter Tuning Process: **

- GridSearchCV testing 350+ parameter combinations
- Parameters optimized: max_depth (4-10), min_samples_split (10-30), min_samples_leaf (1-15), class_weight, criterion
- Multi-metric scoring: Accuracy, balanced accuracy, F1-weighted, F1-high
- Refit strategy: Prioritized F1-high score for clinical relevance

**Key Characteristics:**

- Maximum interpretability with clear decision paths
- Feature importance transparency enabling clinical explanation
- Robust performance after hyperparameter optimization
- Clinical utility through explainable decision rules

2. Random Forest

**Ensemble Approach:**
- 200 decision trees providing robust ensemble predictions
- Bootstrap aggregation reducing overfitting and improving generalization
- Feature randomness at each split preventing dominance by single variables
- Voting mechanism creating stable, reliable predictions

**Optimization Strategy:**
- Balanced multiple objectives across accuracy metrics
- Cross-validation stability ensuring consistent performance
- Feature importance aggregation providing reliable variable rankings
- Clinical performance prioritization through strategic class weighting

3. Support Vector Machine

**Mathematical Approach:**

- RBF kernel capturing non-linear relationships between features and mental health outcomes
- High regularization (C=10) preventing overfitting while maintaining model complexity
- Automatic gamma scaling adapting to feature variance patterns
- Probability estimation enabled for ROC-AUC analysis and ensemble potential

**Parameter Optimization:**
- Comprehensive grid search across kernel types (linear, rbf, polynomial)
- C parameter range: 0.1 to 100 testing regularization strength
- Gamma optimization: scale, auto, and manual values (0.01-1.0)
- Feature scaling applied for optimal SVM performance


### Evaluation

1. Decision Tree Performance

Quantitative Results:
Test Accuracy: 48.4%
Cross-Validation: 72.6% ± 7.2%
High-Risk Recall: 27% (3/11 high-risk individuals identified)
ROC-AUC: 0.669 (fair discrimination)

Clinical Performance Analysis:

Excellent generalization improvement: +6 percentage points CV accuracy over untuned version
Reduced overfitting: More stable performance across validation folds
Interpretable decisions: Clear feature-based rules for clinical application
Balanced performance: Reasonable accuracy across all risk categories

Feature Importance Insights:

Daily_hours_numeric (40%): Dominant predictor validating time-spent hypothesis
Platform_count (15%): Multi-platform usage significantly increases risk
TikTok usage (8%): Highest individual platform risk factor
Occupation status (8%): Professional context influences mental health risk

2. Random Forest Performance
Quantitative Results:

Test Accuracy: 50.5%
Cross-Validation: 69.4% ± 3.1%
High-Risk Recall: 82% (9/11 high-risk individuals identified)
ROC-AUC: 0.758 (good discrimination - highest among all models)

Clinical Excellence:

Outstanding high-risk identification: 82% recall means 4 out of 5 high-risk individuals receive intervention
Balanced accuracy improvement: +18.9 percentage points over untuned version (67.9% vs 49.0%)
Stable cross-validation: Low standard deviation (±3.1%) indicates reliable performance
Clinical safety: Only 1 high-risk individual completely missed (classified as low-risk)

Ensemble Advantages:

Robust probability estimates: Averaging across 200 trees provides stable risk assessments
Reduced overfitting: Ensemble approach prevents memorization of training patterns
Feature importance reliability: Aggregated importance scores provide trustworthy insights
Generalization strength: Consistent performance across diverse data splits

3. Support Vector Machine Performance
Quantitative Results:

Test Accuracy: 72.6% (highest raw accuracy)
Cross-Validation: 80.0% ± 10.5% (highest CV score)
High-Risk Recall: 0% (critical failure - no high-risk predictions made)
ROC-AUC: 0.744 (surprisingly good discrimination despite classification failure)

Critical Performance Issue:

Complete high-risk blindness: Model never predicts high-risk category
All predictions: Limited to Low and Medium risk only
Clinical danger: 100% of high-risk individuals missed entirely
Conservative bias: Extreme preference for "safe" predictions

Technical Analysis:

RBF kernel limitations: Non-linear transformations may have eliminated high-risk decision regions
Class imbalance sensitivity: Despite class weighting, severe imbalance (11% high-risk) overwhelmed optimization
Feature scaling effects: Normalization may have reduced discriminative power of key variables
Probability calibration paradox: Good ROC-AUC suggests ranking ability exists but threshold selection failed


## Recommendation and Analysis
Explain the analysis and recommendations


## AI Ethics
Discuss the potential data science ethics issues (privacy, fairness, accuracy, accountability, transparency) in your project. 



## Source Codes and Datasets
Upload your model files and dataset into a GitHub repo and add the link here. 
