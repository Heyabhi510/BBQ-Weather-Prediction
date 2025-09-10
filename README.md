**Project Overview:**
This project predicts whether the weather conditions in Oslo are suitable for a BBQ day. It is designed as an industry-style end-to-end ML pipeline using scikit-learn Pipelines + Deep Learning, with modularized code for maintainability and reproducibility.

Kaggle Dataset : https://www.kaggle.com/datasets/thedevastator/weather-prediction/data


**The pipeline covers:**
- Data loading & cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering (Month, Season)
- Preprocessing via Pipelines (scaling + encoding)
- Multi-model training (LogReg, RF, XGBoost, LightGBM, ANN)
- Model evaluation (accuracy, confusion matrices, SHAP explainability)



**Project Structure:**

.
├── data_preprocessing.py    # Data loading, cleaning, feature engineering, train/test split
├── pipeline_train_models.py # ML Pipelines, multiple models training & evaluation
├── evaluate.py              # Feature importance, SHAP explainability, confusion matrices
├── utils.py                 # Helper functions (confusion matrix plots)
├── data/
│   ├── weather_prediction_dataset.csv
│   └── weather_prediction_bbq_labels.csv
├── requirements.txt
└── README.md



**Workflow:**

**1. Data Preprocessing**
Loaded weather dataset + BBQ outcome labels (OSLO_BBQ_weather).
Converted DATE column to datetime.

Created additional features:
- Day
- Season (Winter, Spring, Summer, Fall)
Merged with BBQ labels.


**2. Exploratory Data Analysis (EDA)**
KDE plots for cloud cover, wind, humidity, sunshine, temperature, pressure, radiation, precipitation.
Correlation heatmap to study relationships.


**3. Feature Engineering**
Added derived categorical feature Season.
Defined numeric vs. categorical features for preprocessing.


**4. Preprocessing Pipeline**
StandardScaler → numeric features
OneHotEncoder → categorical features (Season)
Wrapped in a ColumnTransformer.


**5. Models Trained**
- Logistic Regression
- Random Forest
 -XGBoost
- LightGBM
- Neural Net (Keras Sequential wrapped with KerasClassifier)
Each model runs inside the same pipeline, ensuring consistent preprocessing.


**6. Model Evaluation**
Accuracy, Precision, Recall, F1
Confusion Matrix (plotted via utils.plot_confusion)
Feature Importance:
- RandomForest → .feature_importances_
- XGBoost → plot_importance

Explainability:
- SHAP TreeExplainer on RandomForest



**🚀 How to Run**
1. Clone the repo:
git clone https://github.com/Heyabhi510/BBQ-Weather-Prediction.git
cd weather-prediction

2. Install requirements:
pip install -r requirements.txt

3. Run preprocessing:
python data_preprocessing.py

4. Train models:
python pipeline_train_models.py

5. Evaluate results:
python evaluate.py



**📦 Requirements**
- Python 3.9+
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- xgboost
- lightgbm
- shap
- tensorflow / keras



**🔮 Future Improvements**
- Hyperparameter tuning (GridSearchCV, Optuna).
- Cross-validation for robust performance.
- Deployment as REST API (FastAPI / Flask).
- Interactive Streamlit dashboard.
