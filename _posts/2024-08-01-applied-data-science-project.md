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

Gender categories were standardized to 'Male' and 'Female', removing a small number of non-binary/other respondents. Age values were converted to integers, missing organization types were filled with the mode, and participants who did not use social media were excluded to ensure all remaining data reflected relevant digital behavior. The specific cleaning steps are as follows: 

**1. Gender Variable Cleaning**

- Original dataset contained multiple non-binary/other categories.
- Recoded all non-'Male'/'Female' entries as 'Other' and removed them due to small sample size (~7 respondents).
- Final dataset: ~478 respondents (Male: 211, Female: 263).

```python
# Renaming the "Others" gender variable
smmh_clean['gender_clean'] = smmh_clean['gender'].apply(lambda x: x if x in ['Male', 'Female'] else 'Other')
smmh_clean = smmh_clean[smmh_clean['gender_clean'].isin(['Male', 'Female'])].copy()
```

**2. Age Data Type Conversion**
   
- Ensure age variable is properly formatted for analysis 

```python
smmh_clean['age'] = smmh_clean['age'].astype(int)
```

**3. Missing Data Handling - Organization Type**
- Imputed missing values with the mode to maintain representative distribution.

```python
# Get the mode of the 'organization_type' column
mode_value = smmh_clean['organization_type'].mode()[0]  
# Fill missing values with the mode
smmh_clean.fillna({'organization_type': mode_value}, inplace=True)
```

**4. Social Media Usage Filter** 
- Removed participants who do not use social media to focus on relevant digital behavior.

```python
# Remove rows where participants don't use social media
smmh_clean = smmh_clean[smmh_clean["uses_social_media"] != "No"]
```

---

**B) Data Transformation**

Platform usage data was split and one-hot encoded, with platform diversity metrics generated for each user. Categorical time ranges were converted into numeric hours, and survey scores were classified into Low, Medium, and High mental health risk based on defined thresholds. Demographic variables like gender, relationship status, and occupation were label-encoded for modeling. The detailed transformation steps are as follows:

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

**Output:** `daily_hours_numeric` variable for correlation analysis

**3. Mental Health Risk Classification**
- Defined low_threhold at 22 as this means the average response ≤ 2.0 per question (between "Never/Rarely" and "Sometimes")
- Defined medium_threhold at 45 as this means the average response is 2.1-4.1 per question ("Sometimes" to "Often")

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

**Output:** `risk_level` categorical variable for predictive modeling

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
Before modeling, the dataset was split 80-20 with stratified sampling to preserve risk level distribution and ensure reproducibility. To address the class imbalance, SMOTE was applied to the training set to generate synthetic samples for the underrepresented Low- and High-Risk classes. The detailed pre-modelling steps are as follows:

**1. Train-Test Split Configuration:**
- 80-20 split with 376 training samples and 95 testing samples
- Stratified sampling to maintain risk level distribution
- Fixed random state (42) for reproducibility

```python
- # Prepare data for modeling
X = smmh_clean[all_features].copy()
y = smmh_clean['risk_level']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

**2. Class Imbalance Handling:**
- Dataset is highly imbalanced: 53 high-risk cases, 51 low-risk and 367 medium-risk.  
- Applied **SMOTE** on the training set to generate synthetic samples for minority classes (i.e low and high risk).  

 ```python 
# Apply SMOTE only on training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
```


### Modelling

Model Selection: Decision Tree, Random Forest, and Support Vector Machine (SVM) were chosen for the classification task to balance interpretability, robustness, and accuracy. Decision Trees provide simple, understandable rules, Random Forest improve generalisation and handle complex feature interactions, and SVM find optimal boundaries in high-dimensional spaces while detecting minority classes effectively. Together, they offer a comprehensive approach for predicting categorical risk levels. Below are the three model configurations used for evaluation and tuning:

**1. Decision Tree (DT)**

**1A. Baseline Model**
- Trained with SMOTE and used class weighting strategies to prioritise high-risk identification
  - `class_weight={'Low': 1, 'Medium': 2, 'High': 5}`
- Initialised with basic hyperparameters:
  - `max_depth = 8`  , `min_samples_split = 15`  , `min_samples_leaf = 5`  


**1B. Tuned Model**

- Used `GridSearchCV` to automatically find the best combination of hyperparameters, testing over 350+ combinations, reducing manual trial-and-error
  - Parameters optimised: max_depth (4-10), min_samples_split (10-20), min_samples_leaf (5-10), criterion
  - Best DT parameters:  `criterion': 'gini`, `max_depth': 8`, `min_samples_leaf: 5`, `min_samples_split': 10`
- Custom scoring metric: Weighted F1 score to prioritise High-Risk class detection.
- Cross validation: StratifiedKFold (5 folds) to maintain class distribution in each fold.

---

**2. Random Forest (RF)**

**2A. Baseline Model**
- Trained with SMOTE and used class weighting strategies to prioritise high-risk identification
  - `class_weight={'Low': 1, 'Medium': 2, 'High': 5}`
- Initialised with basic hyperparameter:
  - `n_estimators = 200`, `max_depth = None`, `min_samples_split = 2`, `min_samples_leaf = 1`  


**2B. Tuned Model**
- Used `GridSearchCV` for hyperparameter optimisation (same rationale as Decision Tree).
  - Parameter optimised: n_estimators: [100, 200], max_depth: [6, 8, 10], min_samples_split: [10, 20], min_samples_leaf: [5, 10]  
  - Best RF paramter: `max_depth: 10`, `min_samples_leaf: 5`, `min_samples_split: 10`, `n_estimators': 200`
- Custom scoring metric: Weighted F1 score to prioritise High-Risk class (same as Decision Tree).
- Cross validation: StratifiedKFold (5 folds) to preserve class distribution.
  
---

**3. Support Vector Machine (SVM)** 

**3A. Baseline Model**
- Trained with SMOTE and used class weighting strategies to prioritise high-risk identification
  - `class_weight={'Low': 1, 'Medium': 2, 'High': 5}`
- Initialised with basic hyperparameter:
  - `probability=True`, `random_state=42`

**3B. Tuned Model**
- Used `GridSearchCV` for hyperparameter optimisation (same rationale as Decision Tree).
  - Parameter optimised: svm__C': [0.1, 1, 10], svm__kernel: ['linear', 'rbf', 'poly'] , svm__gamma: ['scale', 'auto'] (for rbf/poly kernels)   
  - Best SVM paramter: `svm__C: 10` , `svm__gamma: scale`, `svm__kernel': 'rbf` 
- Custom scoring metric: Weighted F1 score to prioritise High-Risk class (same as Decision Tree).
- Cross validation: StratifiedKFold (5 folds) to preserve class distribution.

### Evaluation
The following evaluates the three machine learning models for predicting **mental health risk levels** (Low, Medium, High).  

**1. Decision Tree (DT)**

**Test Set Performance**
- Accuracy: Untuned = 0.611 | Tuned = 0.621  
- Weighted F1: Untuned = 0.653 | Tuned = 0.659  
- Balanced Accuracy: Untuned = 0.537 | Tuned = 0.516  

**Observations**
- Medium- and Low-Risk classes were predicted reasonably well, while High-Risk detection remained limited (F1 0.17–0.21).  
- 5-fold cross-validation showed consistent performance: accuracy ~0.72, weighted F1 ~0.72.  

**Summary**
- Decision Tree is interpretable and provides a simple baseline, but struggles to identify minority High-Risk cases.

---

**2. Random Forest (RF)**

**Test Set Performance**
- Accuracy: Untuned = 0.747 | Tuned = 0.505  
- Weighted F1: Untuned = 0.728 | Tuned = 0.546  
- Balanced Accuracy: Untuned = 0.490 | Tuned = 0.679  

**Observations**
- Untuned model had high overall accuracy but poor minority class detection.  
- Tuned model improved recall for High-Risk (0.09 → 0.82) and Low-Risk, but overall accuracy dropped due to trade-off.  
- Cross-validation confirmed better balanced performance after tuning.  

**Summary**
- Random Forest is effective at detecting High-Risk individuals after tuning, though overall accuracy may decrease due to trade-offs.

---

**3. Support Vector Machine (SVM)**

**Test Set Performance**
- Accuracy: Untuned = 0.600 | Tuned = 0.726  
- Weighted F1: Untuned = 0.636 | Tuned = 0.729  
- Balanced Accuracy: Untuned = 0.507 | Tuned = 0.532  

**Observations**
- Tuned SVM improved detection of Medium-Risk and overall predictive performance.  
- High-Risk detection remained modest (precision 0.25, recall 0.27).  
- Cross-validation showed robust generalization: accuracy ~0.80, weighted F1 ~0.80.  

**Summary**
- Tuned SVM provides the best overall performance across metrics, balancing accuracy, weighted F1, and robustness, making it the preferred model for predicting mental health risk levels.

---

**Overall Model Recommendation**

- **Random Forest:** Strong at detecting High-Risk cases, useful when prioritizing recall for minority classes.  
- **Decision Tree:** Highly interpretable, suitable for baseline modeling and explanation of decision rules, but limited in High-Risk detection.








1. Decision Tree Performance
- The untuned model had stronger overall accuracy but weak High-Risk detection.
- The tuned model performed better during cross-validation with SMOTE data, showing improved fairness across classes, but struggled when tested on the original imbalanced dataset.
- This highlights the challenge of severe class imbalance. While resampling + weighting improves fairness, reliable generalization requires more High-Risk samples.


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



## Recommendation and Analysis


** Areas for Improvement:**

Small Dataset Size
- **Issue:** The dataset contains only **471 total samples**, with just **53 high-risk cases**.  
- **Impact:** This imbalance and limited size may lead to **high variance in results** and **unreliable performance estimates**.  
- **Recommendation:** Collect **3–5x more data**, with a particular focus on **increasing high-risk samples** to improve model stability and generalizability.


Additional Data Fields for Feature Engineering  

To enhance model performance and capture deeper behavioral patterns, future data collection could include **temporal, content & behavior, social interaction, and physiological indicators** (e.g., late-night usage, message response time, sleep duration). These would enable the creation of interaction fields, offering richer insights into digital behavior and its association with mental health risk.  


## AI Ethics
Discuss the potential data science ethics issues (privacy, fairness, accuracy, accountability, transparency) in your project. 

Developing models for mental health risk prediction might involve sensitive data from the respondents. The following ethical issues were carefully considered:

## 🔒 Privacy
- Mental health and social media data are highly sensitive.  
- Mitigation: Encryption, anonymization, pseudonymization, and restricted access controls.  

## ⚖️ Fairness
- Risk of bias if certain demographic groups are underrepresented.  
- Mitigation: Stratified sampling, SMOTE for class balance, subgroup performance monitoring.  

## 📈 Accuracy
- Misclassification may cause stigma (false positives) or missed support (false negatives).  
- Mitigation: Prioritize recall for High-Risk detection, continuous validation, use as decision-support not replacement.  

## 🧾 Accountability
- Responsibility must be defined if predictions cause harm.  
- Mitigation: Human-in-the-loop validation, clear responsibility framework, logging of all modeling decisions.  

## 🔍 Transparency
- Black-box models reduce trust.  
- Mitigation: Interpretable models (Decision Tree), explainability techniques (SHAP/LIME), thorough documentation.  

---

✅ **Conclusion:**  
By embedding **privacy protection, fairness checks, accuracy validation, accountability frameworks, and transparency tools**, this project ensures **responsible and trustworthy use of AI** in mental health risk prediction.  

## Source Codes and Datasets
Upload your model files and dataset into a GitHub repo and add the link here. 
