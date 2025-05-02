# 💳 Credit Card Fraud Detection - Machine Learning Project

## 📚 Project Description
This project aims to accurately detect fraudulent credit card transactions using supervised machine learning. Due to the extreme class imbalance (~0.17% fraud), the project emphasizes recall, robust model evaluation, and handling imbalanced data.

## 🧩 Problem Statement
Build a predictive model that can distinguish between legitimate and fraudulent credit card transactions based on anonymized features, minimizing false negatives while maintaining an acceptable false positive rate.

## 📊 Dataset
- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Records:** 284,807 transactions
- **Fraudulent:** 492 cases (~0.17%)
- **Features:**
  - `Time`, `Amount`
  - `V1` to `V28` (PCA-anonymized)
  - `Class` (0 = Non-Fraud, 1 = Fraud)

## 📂 Project Structure
credit-card-fraud-detection/

├── fraud_model.ipynb         # Jupyter Notebook with EDA, training, evaluation

├── xgboost_model.pkl         # Trained XGBoost model (exported)

├── amount_scaler.pkl         # Fitted StandardScaler for input amount transformation

├── time_scaler.pkl           # Fitted StandardScaler for input time transformation

├── creditcard.csv            # Dataset file

├── app.py                    # Streamlit web app for interactive predictions

└── README.md                 # Project documentation

## 🔍 Exploratory Data Analysis (EDA)
- Identified and visualized severe class imbalance
- Analyzed distribution of transaction amounts and time
- Found clear behavioral differences between fraudulent and legitimate transactions
- Correlation heatmaps and boxplots for feature understanding

## 🧹 Data Preprocessing
- Scaled Amount and Time using StandardScaler
- Dropped original Amount and Time columns
- Split data using stratified train-test split
- Applied SMOTE for minority class oversampling (optional during tuning)

## 🤖 Models Trained
| Model              | Key Characteristics                     |
|--------------------|-----------------------------------------|
| Logistic Regression | Baseline performance                    |
| Random Forest       | Strong ensemble performance             |
| XGBoost             | Final tuned model with highest recall & AUC |

## Evaluation Metrics
- Confusion Matrix, Precision, Recall, F1-Score, ROC AUC
- Focused on maximizing recall to ensure most frauds are detected

## 📈 Results (Sample Metrics)
| Model              | Precision | Recall | F1-Score | ROC-AUC |
|--------------------|-----------|--------|----------|---------|
| Logistic Regression | 84%       | 72%    | 77%      | 0.93    |
| Random Forest       | 90%       | 81%    | 85%      | 0.96    |
| XGBoost (Tuned)     | 89%       | 83%    | 86%      | 0.98    |

✅ Focused on maximizing **recall** to catch fraudulent transactions
✅ Tradeoff with precision carefully analyzed
✅ **XGBoost** was selected as the final model based on strong recall and ROC AUC.

## 🌐 Streamlit Web App
An interactive fraud detection tool was built using Streamlit:
- Real-time prediction using sliders for V1–V28 and Amount
- Batch CSV upload for fraud scoring multiple transactions
- Displays prediction, fraud probability, and downloadable results

### Run the App Locally:
```bash
streamlit run app.py
```

## 🛠️ Technologies Used
- Languages: Python
- Libraries: Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, XGBoost, imbalanced-learn, Streamlit
- Tools: Jupyter Notebook, Streamlit, joblib

## 🚀 Getting Started
1. Clone this repository.
2. Install the dependencies.
3. Open fraud_model.ipynb in Jupyter Notebook.
4. Run all cells sequentially to reproduce results.
5. (Optional) Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 💡 Key Takeaways
Extreme class imbalance requires careful metric selection and model tuning

SMOTE helped increase fraud recall without major performance loss

XGBoost + feature scaling + Streamlit = scalable and interactive solution

## 🙋‍♂️ Author
Ankit Tiwari (https://github.com/ankitis32)