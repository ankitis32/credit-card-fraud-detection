# 💳 Credit Card Fraud Detection - Machine Learning Project

## 📚 Project Description
This project focuses on detecting fraudulent credit card transactions using supervised learning techniques. Since fraud cases are rare, the dataset is highly imbalanced — making it an excellent case for precision-recall tradeoffs, anomaly detection, and robust evaluation metrics.

## 📌 Problem Statement
Given anonymized transaction data, build a machine learning model to accurately identify fraudulent transactions (Class = 1) while minimizing false positives and maximizing recall.

## 📊 Dataset
- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Records:** 284,807 transactions
- **Fraudulent:** 492 cases (~0.17%)
- **Features:**
  - `Time`, `Amount`
  - `V1` to `V28` (PCA-anonymized)
  - `Class` (0 = Non-Fraud, 1 = Fraud)

## 📂 Project Structure
telco-churn-prediction/

├── fraud_model.ipynb

├── creditcard.csv

├── README.md

└── requirements.txt

## 🔍 Exploratory Data Analysis (EDA)
- Highly imbalanced dataset (~0.17% fraud)
- Features are already standardized due to PCA transformation
- Analyzed distribution of amount, time, and class imbalance
- Identified correlation trends and visualized fraudulent vs normal behavior

## 🧹 Preprocessing
- Normalized `Amount` and `Time` columns
- Split data into training and test sets using stratification
- Applied SMOTE for oversampling minority class (optional)

## 🧠 Models Used
### 1. Logistic Regression
- Simple baseline model
- Evaluated using confusion matrix, precision, recall, F1-score, ROC-AUC

### 2. Random Forest
- Ensemble model with improved performance
- Feature importance used to interpret predictive power

### 3. [Optional] XGBoost or Isolation Forest
- Tried for potential improvement in recall and AUC

## 📈 Results
| Model              | Precision | Recall | F1-Score | ROC-AUC |
|-------------------|-----------|--------|----------|---------|
| Logistic Regression | xx%     | xx%    | xx%      | xx%     |
| Random Forest       | xx%     | xx%    | xx%      | xx%     |

- Focused on maximizing **recall** to catch fraudulent transactions
- Tradeoff with precision carefully analyzed

## 📊 Key Insights
- Fraudulent transactions tend to have lower amounts and irregular patterns
- SMOTE helped improve recall but slightly reduced precision
- Ensemble models like Random Forest handled imbalanced data better

## 🛠️ Technologies Used
- Python
- Pandas, NumPy, Seaborn, Matplotlib
- Scikit-learn, imbalanced-learn
- Jupyter Notebooks

## 🚀 How to Run
1. Clone this repository.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open fraud_model.ipynb in Jupyter Notebook.
4. Run all cells sequentially to reproduce results.

## 🙌 Author
Ankit Tiwari (https://github.com/ankitis32)