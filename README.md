# DSandAIP2026_PhyoMaMaTheint
Predicting, Analysing, and Explaining Customer Churn in Telecom Services
Clara | Student Number: 24044014 | UFCEKP-30-3 — Data Science and AI Individual Project | UWE Bristol | April 2026


📋 Project Overview
This project develops a complete, end-to-end machine learning pipeline for predicting, segmenting, and explaining customer churn in the telecommunications industry. Applied to the IBM Watson Telco Customer Churn dataset (7,043 customers, 21 features), the system combines:

Five supervised ML classifiers compared on a consistent evaluation framework
K-Means customer segmentation identifying four distinct customer archetypes
SMOTE oversampling to address class imbalance
Explainable AI (LIME + SHAP) for both individual and population-level interpretability
Tiered risk scoring with thresholds grounded in an expected-value cost framework

Best model: Logistic Regression — ROC-AUC 0.846, Recall 0.789


📁 Repository Structure
Dissertation/

│

├── telecom_churn_analysis.ipynb         # Main Jupyter Notebook (36 annotated cells)

├── WA_Fn-UseC_-Telco-Customer-Churn.csv # dataset file

├── customers_with_churn_scores.csv      # Scored output: all 7,043 customers with churn probability and risk tier

├── Output Files                         # Output Files

└── README.md                        # This file

🗂️ Dataset
Name: IBM Watson Telco Customer Churn
Source: Kaggle — blastchar/telco-customer-churn
Records: 7,043 customers | Features: 21 | Churn rate: 26.54%
Download Instructions
Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Click Download
Extract the ZIP file
Place WA_Fn-UseC_-Telco-Customer-Churn.csv in the same folder as the notebook
Dataset Features
Category
Columns
Demographics
gender, SeniorCitizen, Partner, Dependents
Services
PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
Account Info
tenure, Contract, PaymentMethod, PaperlessBilling, MonthlyCharges, TotalCharges
Target
Churn (Yes / No)



⚙️ Installation
Prerequisites
Python 3.10 or higher
pip (Python package installer)
Jupyter Notebook or JupyterLab
Step 1 — Clone or download the repository

Step 2 — Install dependencies
pip install -r requirements.txt

Step 3 — Place the dataset

Step 4 — Launch Jupyter
jupyter notebook telecom_churn_analysis.ipynb

Or if using JupyterLab:

jupyter lab telecom_churn_analysis.ipynb


📦 Dependencies
All packages are listed in requirements.txt. Key libraries:

Package
Version
Purpose
pandas
≥ 1.5.0
Data loading and manipulation
numpy
≥ 1.23.0
Numerical computation
matplotlib
≥ 3.6.0
Visualisation
seaborn
≥ 0.12.0
Statistical charts
scikit-learn
≥ 1.2.0
ML models, preprocessing, evaluation
imbalanced-learn
≥ 0.10.0
SMOTE oversampling
lime
≥ 0.2.0
Local Interpretable Model-agnostic Explanations
shap
≥ 0.41.0
SHapley Additive exPlanations
plotly
≥ 5.11.0
Interactive visualisations


Install all at once:

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn lime shap plotly


🚀 How to Run
Open telecom_churn_analysis.ipynb and run all cells top-to-bottom (Kernel → Restart & Run All).

The notebook is organised into 36 cells across 6 phases:

Phase
Cells
What happens
Setup
1–2
Install packages, import libraries
Data Loading & Cleaning
3–8
Load CSV, fix BOM encoding, handle missing values, visualise missingness
Feature Engineering
9
Create TenureGroup, ChargeGroup, NumServices, AvgMonthlySpend
Exploratory Data Analysis
10–16
14 visualisations across churn distribution, demographics, contracts, services, correlations
Customer Segmentation
17–19
K-Means elbow method, k=4 clustering, PCA visualisation, cluster profiles
Preprocessing for ML
20–21
Label encoding, 80/20 stratified split, StandardScaler, SMOTE
Model Training & Evaluation
22–27
Train 5 models, evaluate 5 metrics, cross-validation, ROC curves, confusion matrices
Feature Importance
28
Permutation importance for Logistic Regression
Explainable AI
29–31
LIME (individual), SHAP bar chart (global), SHAP beeswarm
Risk Scoring & Output
32–36
Score all customers, risk tier classification, export CSV, recommendations



📊 Key Results
Model Comparison
Model
Accuracy
Precision
Recall
F1
ROC-AUC
Logistic Regression
0.748
0.517
0.789
0.624
0.846
AdaBoost
0.764
0.541
0.730
0.621
0.839
Gradient Boosting
0.781
0.577
0.652
0.612
0.837
Random Forest
0.769
0.562
0.586
0.573
0.824
K-Nearest Neighbours
0.683
0.441
0.735
0.552
0.763


Logistic Regression was selected as the best model. Its high recall (0.789) prioritises catching genuine churners — the most important outcome for a retention strategy — consistent with Shwartz-Ziv and Armon (2022), who demonstrated that well-tuned linear models remain competitive on tabular datasets at moderate sample sizes.
Customer Segments (K-Means, k=4)
Cluster
Avg Tenure
Avg Monthly Charges
Avg Services
Churn Rate
0 — Loyal Low-Spenders
53.6 months
£30.96
1.48
4.6%
1 — Mid-Tenure High-Payers
18.4 months
£80.41
3.28
43.0%
2 — Long-Tenure Heavy Users
59.8 months
£92.09
5.06
14.5%
3 — New Low-Spenders
9.0 months
£37.71
1.20
32.0%


Cluster 1 is the highest-priority retention target: mid-tenure, high-paying customers with a 43% churn rate.
Risk Tier Distribution
Risk Tier
Threshold
Customers
% of Base
🔴 High Risk
> 0.60
2,271
32.2%
🟡 Medium Risk
0.30 – 0.60
1,671
23.7%
🟢 Low Risk
< 0.30
3,101
44.0%

Top Churn Drivers (SHAP + Permutation Importance)
MonthlyCharges — highest impact by a substantial margin
AvgMonthlySpend — engineered feature, validates preprocessing
tenure — short tenure strongly predicts churn
Contract — month-to-month contracts: 42.7% churn vs 2.8% for two-year
TotalCharges — correlated with tenure and spend


⚠️ Known Issues and Fixes
BOM Encoding Error (KeyError: 'Churn')
The CSV file contains a hidden Unicode byte-order mark (BOM) character that corrupts column names when loaded with default settings. Fixed in Cell 3:

# WRONG — causes KeyError: 'Churn'

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# CORRECT — strips BOM and whitespace from column names

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv', encoding='utf-8-sig')

df.columns = df.columns.str.strip()
SHAP Version Compatibility
Newer versions of SHAP (≥ 0.41) changed TreeExplainer output from a list of 2D arrays to a single 3D array (samples × features × classes). The notebook handles this with a conditional check:

if isinstance(shap_values_raw, list):

    sv = shap_values_raw[1]           # Old SHAP: list per class

elif shap_values_raw.ndim == 3:

    sv = shap_values_raw[:, :, 1]    # New SHAP: 3D array

else:

    sv = shap_values_raw              # Single output model
Delimiter Detection
If your CSV fails to load with the correct number of columns, the file may be using a non-comma separator. Use auto-detection:

df = pd.read_csv(

    'WA_Fn-UseC_-Telco-Customer-Churn.csv',

    encoding='utf-8-sig',

    sep=None,

    engine='python'

)

df.columns = df.columns.str.strip()


📤 Output Files
After running all cells, the following files are saved in the working directory:

File
Description
customers_with_churn_scores.csv
All 7,043 customers with ChurnProbability, RiskCategory, and Actual_Churn
churn_distribution.png
Pie and bar chart of overall churn balance
churn_demographics.png
Churn rates by gender, senior status, partner, dependents, tenure group, charge group
churn_contract_payment.png
Churn rates by contract type and payment method
numeric_distributions.png
Overlapping density histograms for tenure, charges, and total charges
correlation_heatmap.png
Feature correlation matrix
services_heatmap.png
Churn rate by internet service add-on
services_churn.png
Churn rate and count by number of services
kmeans_elbow.png
Elbow method for optimal k
customer_segments.png
PCA projection of customer clusters
cluster_churn_rates.png
Churn rate per segment
model_comparison.png
Side-by-side bar chart of all 5 metrics across all 5 models
roc_curves.png
ROC curves with AUC scores for all models
confusion_matrices.png
Confusion matrices for all 5 models
feature_importance.png
Permutation importance for Logistic Regression
lime_explanation.png
LIME local explanation for high-confidence churner
shap_summary_bar.png
SHAP global feature importance bar chart
shap_beeswarm.png
SHAP beeswarm plot showing direction and magnitude
churn_risk_distribution.png
Probability histogram and risk tier pie chart



🔬 Methodology Summary
Raw CSV (7,043 rows × 21 cols)

        │

        ▼

Data Cleaning

  • Fix BOM encoding         • Impute TotalCharges (11 missing)

  • Drop customerID          • Encode binary columns

        │

        ▼

Feature Engineering

  • TenureGroup  • ChargeGroup  • NumServices  • AvgMonthlySpend

        │

        ├──────────────────────────────────────┐

        ▼                                      ▼

Exploratory Data Analysis              K-Means Segmentation (k=4)

  14 visualisations                    4 customer archetypes

  Churn patterns identified            Cluster 1: 43% churn rate

        │                                      │

        └──────────────────────────────────────┘

                           │

                           ▼

                    Preprocessing for ML

                    • 80/20 stratified split

                    • StandardScaler (fit on train only)

                    • SMOTE (train set only → 1:1 class ratio)

                           │

                           ▼

                  5 Classifiers Trained & Evaluated

                  • Logistic Regression  ← Best (ROC-AUC 0.846)

                  • Random Forest

                  • Gradient Boosting

                  • AdaBoost

                  • K-Nearest Neighbours

                           │

                           ▼

                    Explainable AI

                    • Permutation Importance

                    • LIME (individual — high-confidence example, p=0.872)

                    • SHAP (population — bar chart + beeswarm)

                           │

                           ▼

                  Risk Scoring & Retention Strategy

                  • All 7,043 customers scored

                  • 3 risk tiers (thresholds: 0.30, 0.60)

                  • Tiered retention recommendations

                  • Output: customers_with_churn_scores.csv


📚 Key References
Chawla, N.V. et al. (2002) SMOTE. Journal of Artificial Intelligence Research, 16, pp. 321–357.
Fernandez, A. et al. (2018) SMOTE for learning from imbalanced data. Journal of Artificial Intelligence Research, 61, pp. 863–905.
Lundberg, S.M. and Lee, S.I. (2017) A unified approach to interpreting model predictions. NeurIPS, 30.
Ribeiro, M.T., Singh, S. and Guestrin, C. (2016) Why should I trust you? KDD 2016, pp. 1135–1144.
Shwartz-Ziv, R. and Armon, A. (2022) Tabular data: Deep learning is not all you need. Information Fusion, 81, pp. 84–90.
Verbeke, W. et al. (2012) New insights into churn prediction. European Journal of Operational Research, 218(1), pp. 211–229.

Full reference list available in the project report: Churn_Dissertation_Clara_24044014.docx


📄 Licence
This project was produced for academic assessment purposes at the University of the West of England (UWE Bristol) as part of module UFCEKP-30-3. The dataset is owned by IBM and distributed via Kaggle under their terms of use.


👤 Author
Phyo Ma Ma Theint
Student Number: 24044014
Module: UFCEKP-30-3 — Data Science and AI Individual Project
University of the West of England, Bristol
April 2026

