Customer Churn Prediction Model
Business Analytics Capstone Project | Spring 2024 | Grade: A
🎯 Project Overview
This project developed a machine learning model to predict customer churn for a telecommunications company, identifying customers at risk of leaving and enabling proactive retention strategies. The model achieved 85% accuracy and identified $2.3M in potential revenue recovery through targeted interventions.
📊 Business Problem
Customer churn represents a critical challenge for subscription-based businesses. Acquiring new customers costs 5-7x more than retaining existing ones. The goal was to:

Identify customers likely to churn before they leave
Understand key factors driving customer attrition
Develop actionable retention strategies
Quantify potential revenue impact of interventions

🔍 Dataset

Size: 7,043 customer records
Features: 21 variables including demographics, service usage, billing information
Target: Binary churn indicator (Yes/No)
Time Period: 12 months of customer data
Industry: Telecommunications services

Key Features Analyzed:

Demographics: Age, gender, senior citizen status, dependents
Services: Phone, internet, streaming, security services
Account: Contract length, payment method, tenure
Billing: Monthly charges, total charges, paperless billing

🛠️ Methodology
1. Data Preprocessing

Missing Values: Handled 11 records with missing TotalCharges
Feature Engineering: Created tenure groups, charge ratios, service counts
Encoding: One-hot encoding for categorical variables
Scaling: StandardScaler for numerical features
Class Balance: Addressed 73%/27% imbalance using SMOTE

2. Exploratory Data Analysis
Key Findings:

Month-to-month contracts show 43% churn rate vs 3% for long-term contracts
Customers with fiber optic internet have higher churn (30% vs 20%)
Electronic check payments correlate with increased churn risk
New customers (tenure < 12 months) represent 35% of churn cases

3. Model Development
Models Evaluated:

Logistic Regression (baseline)
Random Forest
Gradient Boosting (XGBoost)
Support Vector Machine
Neural Network

Best Performing Model: Random Forest Classifier

Accuracy: 85.2%
Precision: 82.1%
Recall: 78.9%
F1-Score: 80.5%
AUC-ROC: 0.891

4. Feature Importance Analysis
Top Predictive Features:

Total Charges (18.2% importance)
Monthly Charges (15.7% importance)
Tenure (14.3% importance)
Contract Type (12.1% importance)
Internet Service Type (9.8% importance)

📈 Results & Business Impact
Model Performance

Successfully identified 78.9% of actual churners
False positive rate: Only 12.3% of predictions were incorrect
Risk Scoring: Customers ranked by churn probability (0-100%)

Business Value Delivered

$2.3M Revenue Recovery Potential: Based on intervention success rates
Customer Retention: Target 1,200 high-risk customers quarterly
Cost Efficiency: Focus retention spend on highest-value at-risk customers
Strategic Insights: Contract and payment method optimization recommendations

Key Business Insights

Contract Strategy: Incentivize long-term contracts to reduce churn by 40%
Payment Methods: Migrate customers from electronic check to autopay
New Customer Focus: Implement 90-day onboarding program
Service Bundling: Promote multi-service packages for stability

🎯 Implementation Strategy
Retention Campaign Framework
High-Risk Customers (Probability > 80%):

Personal outreach from retention specialists
Customized discount offers (10-20% for 6 months)
Service upgrade incentives

Medium-Risk Customers (Probability 50-80%):

Automated email campaigns
Contract upgrade offers
Payment method optimization

Low-Risk Customers (Probability < 50%):

Satisfaction surveys
Loyalty program enrollment
Service expansion opportunities

Expected ROI

Investment: $500K in retention campaigns
Revenue Protected: $2.3M annually
Net ROI: 360% return on investment
Customer Lifetime Value: Increased by avg. $1,200 per retained customer

💻 Technical Implementation
Tools & Technologies Used

Python: Pandas, NumPy, Scikit-learn
Visualization: Matplotlib, Seaborn, Plotly
Model Development: RandomForestClassifier, XGBoost, GridSearchCV
Deployment: Tableau dashboard for business stakeholders
Version Control: Git for code management

Model Deployment

Scoring Pipeline: Automated monthly batch scoring of customer base
Dashboard Integration: Real-time churn risk monitoring in Tableau
Alert System: Automated notifications for customers exceeding risk thresholds
Performance Monitoring: Monthly model performance validation

📊 Visualizations
The project included comprehensive data visualizations:

Customer churn distribution by key segments
Feature importance rankings
ROC curves and confusion matrices
Business impact projections
Interactive Tableau dashboard for stakeholders

🔮 Future Enhancements

Real-time Scoring: Implement streaming data pipeline
Advanced Features: Customer behavior patterns, support ticket sentiment
Deep Learning: Neural network models for improved accuracy
A/B Testing: Validate intervention effectiveness
Segment-Specific Models: Tailored models for customer segments

🏆 Project Outcomes

Academic Achievement: Grade A on capstone project
Business Impact: $2.3M revenue recovery identification
Technical Skills: Advanced machine learning and business analytics
Stakeholder Communication: Executive presentation and dashboard delivery

📝 Key Learnings

Feature Engineering: Domain knowledge critical for creating predictive features
Class Imbalance: SMOTE significantly improved model recall
Business Context: Understanding cost of acquisition vs. retention essential
Model Interpretability: Random Forest provided best balance of accuracy and explainability
Deployment Considerations: Model monitoring crucial for sustained performance


This project demonstrates end-to-end analytics capabilities from problem definition through business impact quantification, showcasing both technical machine learning skills and strategic business thinking.strategic business thinking.*
