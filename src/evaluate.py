import shap
import xgboost as xgb
from data_preprocessing import plt, y_test, np
from pipeline_train_models import models, y_pred, feature_names, X_test_transformed
from utils import plot_confusion


# Prediction & Visualization:
# - Plot predicted vs actual temperature
# - Future prediction example

# 1. Feature Importance (RandomForest)
plt.figure(figsize=(10,6))
importances = models["Random Forest"].feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
plt.title("Random Forest Feature Importance")
plt.show()


# 2. Feature Importance (XGBoost)
xgb.plot_importance(models["XGBoost"], max_num_features=10, importance_type="weight")
plt.title("XGBoost Feature Importance")
plt.show()


# 3. SHAP Values (Explainability)
explainer = shap.TreeExplainer(models["Random Forest"])
shap_values = explainer.shap_values(X_test_transformed)

# Summary Plot (which features matter most overall)
shap.summary_plot(shap_values, X_test_transformed)


# 4. Confusion Matrix + Classification Report (All Models)
for name, model in models.items():
    plot_confusion(y_test, y_pred, name)
    
    plt.show()