        
# Predicting Cerebral Stroke Risk Using Machine Learning

# Predicting Cerebral Stroke Risk Using Machine Learning

## 📌 Project Overview
This project aims to develop a machine learning model to predict the risk of cerebral stroke using health and lifestyle data. By leveraging structured datasets and advanced classification algorithms, the goal is to enhance early detection, improve patient outcomes, and reduce healthcare costs.

## 📂 Dataset
- **Source**: Kaggle Stroke Prediction Dataset ([Link](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset))
- **Rows**: 9,722
- **Columns**: 15
- **Features**:
  - `id`, `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`, `Physical Activity Level`, `Chronic Stress Level`, `Family History of Stroke`
- **Target Variable**: `stroke` (0: No Stroke, 1: Stroke)

## 🔍 Problem Statement
Stroke is a leading cause of disability and mortality worldwide. Traditional risk assessment methods often lack accuracy and timeliness. This project focuses on developing a predictive model that can assess stroke likelihood based on patient data.

## 🎯 Objectives
- Build a classification model to predict the risk of stroke.
- Use structured health datasets to improve accuracy.
- Reduce false negatives to ensure early detection.
- Deploy the model for real-world usability.

## 🏗️ Methodology
1. **Data Collection**
   - Used publicly available datasets from Kaggle.
   - Ensured data quality and reliability.
2. **Exploratory Data Analysis (EDA)**
   - Identified missing values and outliers using bar plots and box plots.
   - Analyzed categorical data (e.g., smoking status, gender) using bar and pie charts.
   - Examined numerical data (e.g., age, BMI, glucose levels) using histograms and violin plots.
3. **Data Preprocessing**
   - **Handling Missing Values**: Mean for numerical features, mode for categorical features.
   - **Outlier Removal**: Used IQR method to detect and remove outliers.
   - **Label Encoding**: Converted categorical variables to numerical values.
   - **Feature Scaling**: Standardization (mean=0, standard deviation=1).
4. **Model Training & Evaluation**
   - Applied multiple classification algorithms:
     - Logistic Regression
     - Decision Trees
     - Random Forest
     - Support Vector Machine (SVM)
     - AdaBoost
     - XGBoost
     - CatBoost
   - Evaluated models using accuracy, precision, recall, F1-score, and AUROC curve.
   - Focused on maximizing recall to reduce false negatives.
5. **Hyperparameter Tuning**
   - Implemented Grid Search and Random Search for optimal performance.
   - Achieved best results with Decision Tree + AdaBoost.
6. **Deployment**
   - Deployed on AWS using EC2.
   - Built a web application using Flask with HTML and CSS frontend.

## 🚀 Results
| Model | Accuracy | F1 Score |
|--------|----------|---------|
| Decision Tree + AdaBoost | **99.02%** | **0.99** |
| CatBoost | 98.81% | 0.98 |
| XGBoost | 98.01% | 0.97 |

## 🔥 Key Challenges
- **Data Quality**: Ensuring accuracy and completeness.
- **High Recall**: Maximizing true positive rate to correctly identify stroke cases.
- **Minimizing False Negatives**: Preventing misclassification to reduce potential risks.

## 📊 Business Insights
- **Early Detection & Prevention**: Identifies high-risk individuals before symptoms appear.
- **Improved Healthcare Decision-Making**: Helps prioritize high-risk patients for early intervention.
- **Cost Reduction in Healthcare**: Reduces hospitalizations and long-term rehabilitation costs.
- **Integration with Wearable Tech**: Potential to link with smartwatches for real-time monitoring.

## 🔮 Future Scope
- Expanding model capabilities to predict other cardiovascular diseases.
- Enhancing real-time monitoring with IoT devices and wearable technology.
- Integrating deep learning for improved feature extraction and classification.

## 🤝 Contributors
- **Prateek Alur**

## 📜 License
This project is licensed under the MIT License.

## 🎯 Contact
For any queries or collaborations, feel free to reach out to **Prateek Alur** email at **[alurprateek2@gmail.com]**.

