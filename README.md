# DSandAIP2026_PhyoMaMaTheint
Predicting, Analysing, and Explaining Customer Churn in Telecom Services  

Phyo Ma Ma Theint  

Student Number: 24044014  

UFCEKP-30-3 — Data Science and AI Individual Project  

UWE Bristol  

April 2026


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

| Package           | Version   | Purpose                                      |
|------------------|----------|----------------------------------------------|
| pandas           | ≥ 1.5.0  | Data loading and manipulation                |
| numpy            | ≥ 1.23.0 | Numerical computation                        |
| matplotlib       | ≥ 3.6.0  | Visualisation                                |
| seaborn          | ≥ 0.12.0 | Statistical charts                           |
| scikit-learn     | ≥ 1.2.0  | ML models, preprocessing, evaluation         |
| imbalanced-learn | ≥ 0.10.0 | SMOTE oversampling                           |
| lime             | ≥ 0.2.0  | Local Interpretable Model-agnostic Explanations |
| shap             | ≥ 0.41.0 | SHapley Additive exPlanations                |
| plotly           | ≥ 5.11.0 | Interactive visualisations     


Install all at once:

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn lime shap plotly


🚀 How to Run
Open telecom_churn_analysis.ipynb and run all cells top-to-bottom (Kernel → Restart & Run All).



📤 Output Files
After running all cells, the following files are saved in the working directory:

File
| File Name                         | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| customers_with_churn_scores.csv  | Dataset of 7,043 customers with ChurnProbability, RiskCategory, and Actual_Churn |
| churn_distribution.png           | Pie and bar chart showing overall churn balance                            |
| churn_demographics.png           | Churn rates by gender, senior status, partner, dependents, tenure group, and charge group |
| churn_contract_payment.png       | Churn rates by contract type and payment method                            |
| numeric_distributions.png        | Density histograms for tenure, monthly charges, and total charges          |
| correlation_heatmap.png          | Feature correlation matrix                                                 |
| services_heatmap.png             | Churn rate by internet service add-ons                                     |
| services_churn.png               | Churn rate and count by number of services                                 |
| kmeans_elbow.png                 | Elbow method for determining optimal number of clusters (k)                |
| customer_segments.png            | PCA projection of customer clusters                                        |
| cluster_churn_rates.png          | Churn rate per customer segment                                            |
| model_comparison.png             | Comparison of 5 evaluation metrics across models                           |
| roc_curves.png                   | ROC curves with AUC scores for all models                                  |
| confusion_matrices.png           | Confusion matrices for all models                                          |
| feature_importance.png           | Permutation feature importance for Logistic Regression                     |
| lime_explanation.png             | LIME explanation for a high-confidence churn prediction                    |
| shap_summary_bar.png             | SHAP global feature importance (bar chart)                                 |
| shap_beeswarm.png                | SHAP beeswarm plot showing feature impact                                  |
| churn_risk_distribution.png      | Probability distribution and risk tier breakdown                           |





📚 Key References
Chawla, N.V. et al. (2002) SMOTE. Journal of Artificial Intelligence Research, 16, pp. 321–357.

Fernandez, A. et al. (2018) SMOTE for learning from imbalanced data. Journal of Artificial Intelligence Research, 61, pp. 863–905.

Lundberg, S.M. and Lee, S.I. (2017) A unified approach to interpreting model predictions. NeurIPS, 30.

Ribeiro, M.T., Singh, S. and Guestrin, C. (2016) Why should I trust you? KDD 2016, pp. 1135–1144.

Shwartz-Ziv, R. and Armon, A. (2022) Tabular data: Deep learning is not all you need. Information Fusion, 81, pp. 84–90.

Verbeke, W. et al. (2012) New insights into churn prediction. European Journal of Operational Research, 218(1), pp. 211–229.

Full reference list available in the project report: DSandAIP2026_PhyoMaMaTheint.pdf


📄 Licence
This project was produced for academic assessment purposes at the University of the West of England (UWE Bristol) as part of module UFCEKP-30-3. The dataset is owned by IBM and distributed via Kaggle under their terms of use.


👤 Author
Phyo Ma Ma Theint  
Student Number: 24044014  
Module: UFCEKP-30-3 — Data Science and AI Individual Project  
University of the West of England, Bristol  
April 2026

