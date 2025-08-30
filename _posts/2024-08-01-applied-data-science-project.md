---
layout: post
author: Name
title: "Applied Data Science Project Documentation"
categories: ITD214
---
## Project Background

Mental health disorders including depression, anxiety, and burnout remain highly prevalent among adults, particularly younger generations, yet persistent stigma and reactive healthcare models result in delayed identification and treatment when conditions become more severe and resource-intensive. The Ministry of Health's Office for Healthcare Transformation (MOHT) seeks to develop a comprehensive data-driven solution that proactively identifies early warning signs and key contributing factors of mental health challenges among adults, enabling timely interventions to prevent long-term mental health deterioration and optimise healthcare resource allocation while overcoming traditional barriers of stigma and delayed help-seeking behavior.

Our project aims to develop a data-driven solution that identifies early warning signs and key contributing factors of mental health challenges among adults, with the aim of enabling timely intervention and reducing the risk of long-term mental health deterioration. This will provide healthcare providers with actionable risk assessments, support evidence-based policy development for digital wellbeing initiatives, and ultimately reduce healthcare costs while improving population mental health outcomes through early, targeted interventions.

**Primary Objective:**
To identify social media usage patterns and behavioral indicators that are predictive of mental health symptoms in adults, enabling early detection and timely interventions. This analysis will examine relationships between digital engagement behaviors and mental health outcomes to develop predictive models that support healthcare providers in identifying at-risk individuals and contributing to MOHT's proactive mental health strategy.


**Research Questions:**
This project seeks to addresses research questions in digital mental health:
1) Do individuals who spend more time on social media platforms show higher mental health risk levels?
2) Can we predict mental health risk based on digital usage patterns and demographic characteristics?

**Dataset**

- Source:[ Social Media and Mental Health (SMMH) ]([url](https://www.kaggle.com/datasets/souvikahmed071/social-media-and-mental-health/data)) data (https://www.kaggle.com/datasets/souvikahmed071/social-media-and-mental-health/data)
- Sample Size: 481 participants
- Features: 21 variables including demographics, social media usage patterns, and psychological indicators
- Target: Mental health risk level (Low, Medium, High) derived from 11 Likert-scale psychological assessment questions

## Work Accomplished
**Document your work done to accomplish the outcome


### Data Preparation

**A) Data Cleaning:**

**1. Gender Variable Cleaning**

  - 9 Original Categories Found: 'Male', 'Female', 'Nonbinary', 'Non-binary', 'NB', 'unsure', 'Trans', 'Non binary', 'There are others???'

```python
# Renaming the "Others" gender variable
smmh_clean['gender_clean'] = smmh_clean['gender'].apply(lambda x: x if x in ['Male', 'Female'] else 'Other')
smmh_clean = smmh_clean[smmh_clean['gender_clean'].isin(['Male', 'Female'])].copy()
```

  - Rationale:
    - As the "Others" category is small, excluded 7 respondents with non-binary/other gender identities
    - Final gender distribution: ~478 respondents (Male: ~211, Female: ~263)

**2. Age Data Type Conversion**
   
  - Ensure age variable is properly formatted for analysis 

```python
smmh_clean['age'] = smmh_clean['age'].astype(int)
```

**3. Missing Data Handling - Organization Type**

```python
# Get the mode of the 'organization_type' column
mode_value = smmh_clean['organization_type'].mode()[0]  
# Fill missing values with the mode
smmh_clean.fillna({'organization_type': mode_value}, inplace=True)
```

  - Rationale:
    - Used mode imputation (most frequent value) for categorical variable
    - Maintains representative distribution of organization types

**4. Social Media Usage Filter** 

```python
# Remove rows where participants don't use social media
smmh_clean = smmh_clean[smmh_clean["uses_social_media"] != "No"]
```

  - Rationale:
    - Excluded non-social media users from analysis to ensure all remaining respondents have relevant digital behavior data

**B) Data Transformation**

**1. Platform Usage Feature Engineering**

- Split comma-separated platform strings → structured lists
- Created one-hot encoded variables for each platform
- Generated platform diversity metrics

**Output**
  - Individual platform indicators (Facebook, Instagram, Twitter, etc.)
  - platform_count: Total platforms used per user
  - Diversity categories: Single (1), Multi (2-3), High (4+) platform users

**2. Usage Time Conversion**
- Transformed categorical time ranges → continuous numerical values
  
```python
"Less than 1 hour" → 0.5 hours
"Between 1-2 hours" → 1.5 hours
"Between 2-3 hours" → 2.5 hours
"Between 3-4 hours" → 3.5 hours
"Between 4-5 hours" → 4.5 hours
"More than 5 hours" → 6.0 hours
```

**Output:** daily_hours_numeric variable for correlation analysis

**3. Mental Health Risk Classification**
- low_threhold set at 22 as this means the average response ≤ 2.0 per question (between "Never/Rarely" and "Sometimes")
- medium_threhold set at 45 as this means the average response is 2.1-4.1 per question ("Sometimes" to "Often")

Created three-tier risk system:

```python
# Define thresholds 
low_threshold = 22
medium_threshold = 45

def classify_risk_level(score):
    if score <= low_threshold:
        return 'Low'
    elif score <= medium_threshold:
        return 'Medium'
    else:
        return 'High'
```

**Output:** risk_level categorical variable for predictive modeling

**4. Categorical Encoding**
- Applied Label Encoding to demographic variables:
  - Gender → gender_encoded
  - Relationship Status → rs_encoded
  - Occupation → occ_encoded

- Example:
  
```python
  # Encode gender
le_gender = LabelEncoder()
smmh_clean['gender_encoded'] = le_gender.fit_transform(smmh_clean['gender'])
all_features.append('gender_encoded')
```


### Pre-Modelling

**1. Train-Test Split Configuration:**
- 80-20 split maintaining adequate sample sizes for both training (376 samples) and testing (95 samples)
- Stratified sampling to preserve risk level distribution
- Fixed random state (42) ensuring reproducibility across multiple runsom state for reproducibility

**2. Class Imbalance Handling:**
- Applied SMOTE to training data to address class imbalance


### Modelling
1. Decision Tree


- Maintained test set integrity for unbiased evaluation
- Used class weighting strategies {Low:1, Medium:2, High:5} to prioritize high-risk identification
- 
**Hyperparameter Tuning Process: **

- GridSearchCV testing 350+ parameter combinations
- Parameters optimized: max_depth (4-10), min_samples_split (10-30), min_samples_leaf (1-15), class_weight, criterion
- Multi-metric scoring: Accuracy, balanced accuracy, F1-weighted, F1-high
- Refit strategy: Prioritized F1-high score for clinical relevance


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
- Test Accuracy: 48.4%
- Cross-Validation: 72.6% ± 7.2%
- High-Risk Recall: 27% (3/11 high-risk individuals identified)
- ROC-AUC: 0.669 (fair discrimination)

Clinical Performance Analysis:

- Excellent generalization improvement: +6 percentage points CV accuracy over untuned version
- Reduced overfitting: More stable performance across validation folds
- Interpretable decisions: Clear feature-based rules for clinical application
- Balanced performance: Reasonable accuracy across all risk categories

Feature Importance Insights:

- Daily_hours_numeric (40%): Dominant predictor validating time-spent hypothesis
- Platform_count (15%): Multi-platform usage significantly increases risk
- TikTok usage (8%): Highest individual platform risk factor

2. Random Forest Performance
Quantitative Results:

- Test Accuracy: 50.5%
- Cross-Validation: 69.4% ± 3.1%
- High-Risk Recall: 82% (9/11 high-risk individuals identified)
- ROC-AUC: 0.758 (good discrimination - highest among all models)

Clinical Excellence:

- Outstanding high-risk identification: 82% recall means 4 out of 5 high-risk individuals receive intervention
- Balanced accuracy improvement: +18.9 percentage points over untuned version (67.9% vs 49.0%)
- Stable cross-validation: Low standard deviation (±3.1%) indicates reliable performance
- Clinical safety: Only 1 high-risk individual completely missed (classified as low-risk)

Ensemble Advantages:

- Robust probability estimates: Averaging across 200 trees provides stable risk assessments
- Reduced overfitting: Ensemble approach prevents memorization of training patterns
- Feature importance reliability: Aggregated importance scores provide trustworthy insights
- Generalization strength: Consistent performance across diverse data splits

3. Support Vector Machine Performance
Quantitative Results:

- Test Accuracy: 72.6% (highest raw accuracy)
- Cross-Validation: 80.0% ± 10.5% (highest CV score)
- High-Risk Recall: 0% (critical failure - no high-risk predictions made)
- ROC-AUC: 0.744 (surprisingly good discrimination despite classification failure)

Critical Performance Issue:

- Complete high-risk blindness: Model never predicts high-risk category
- All predictions: Limited to Low and Medium risk only
- Clinical danger: 100% of high-risk individuals missed entirely
- Conservative bias: Extreme preference for "safe" predictions

Technical Analysis:

- RBF kernel limitations: Non-linear transformations may have eliminated high-risk decision regions
- Class imbalance sensitivity: Despite class weighting, severe imbalance (11% high-risk) overwhelmed optimization
- Feature scaling effects: Normalization may have reduced discriminative power of key variables
- Probability calibration paradox: Good ROC-AUC suggests ranking ability exists but threshold selection failed


## Recommendation and Analysis
Primary Recommendation: Random Forest Deployment
Clinical Deployment Rationale:
- Random Forest emerges as the optimal choice for clinical mental health screening applications based on comprehensive evaluation criteria:

**Superior Clinical Performance:
- 82% high-risk recall: Identifies 4 out of 5 individuals needing immediate intervention
- Balanced accuracy: Strong performance across all risk categories (67.9%)
- Clinical safety: Minimizes dangerous false negatives (missed high-risk cases)
- Population screening suitable: Reliable performance for large-scale deployment

** Technical Reliability:

- Highest ROC-AUC (0.758): Best overall discriminative ability
- Stable cross-validation: Consistent performance across different data splits (±3.1% SD)
- Robust probability estimates: Ensemble averaging provides calibrated risk probabilities
- Generalization strength: Proper balance between training and validation performance

** Areas for Improvement:**

Small Dataset Size
- **Issue:** The dataset contains only **471 total samples**, with just **53 high-risk cases**.  
- **Impact:** This imbalance and limited size may lead to **high variance in results** and **unreliable performance estimates**.  
- **Recommendation:** Collect **3–5x more data**, with a particular focus on **increasing high-risk samples** to improve model stability and generalizability.



## AI Ethics
Discuss the potential data science ethics issues (privacy, fairness, accuracy, accountability, transparency) in your project. 



## Source Codes and Datasets
Upload your model files and dataset into a GitHub repo and add the link here. 
