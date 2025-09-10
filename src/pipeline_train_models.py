from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from data_preprocessing import X_train, X_test, y_train, y_test
import warnings
warnings.filterwarnings('ignore')


# Storing accuracies of all the model in a dictonary
results = {}


# Define features
numeric_features = ['OSLO_cloud_cover', 'OSLO_wind_speed', 'OSLO_wind_gust', 'OSLO_humidity', 'OSLO_pressure', 'OSLO_global_radiation', 'OSLO_precipitation', 'OSLO_sunshine', 'OSLO_temp_mean', 'OSLO_temp_min', 'OSLO_temp_max']
categorical_features = ["Season"]


# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# Building wrapper for Neural Net Models
def ann_model_wrapper(input_dim):
    ann_model = Sequential([
        Dense(units = 128, activation = 'relu', kernel_regularizer = 'l2', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(units = 64, activation = 'relu', kernel_regularizer = 'l2'),
        Dropout(0.2),
        Dense(units = 32, activation = 'relu', kernel_regularizer = 'l2'),
        Dropout(0.2),
        Dense(units = 1, activation = 'sigmoid')
    ])

    ann_model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])

    return ann_model


# First get transformed training and testing data shape
X_train_transformed = preprocessor.fit_transform(X_train)
input_dim = X_train_transformed.shape[1]
X_test_transformed = preprocessor.fit_transform(X_test)

# Transformed feature names
feature_names = (
    preprocessor.named_transformers_['num'].get_feature_names_out(numeric_features).tolist() +
    preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
)

ann_clf = KerasClassifier(model=ann_model_wrapper(input_dim), epochs=100, batch_size=32, verbose=1)


# Model Dictionary
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=150, random_state=42, eval_metric="mlogloss", use_label_encoder=False),
    "LightBGM": LGBMClassifier(random_state=42),
    "Neural Net": ann_clf
}


# Train and Evaluate
for name, model in models.items():
    print(f"\n--- {name} ---")
    
    # Pipeline = Preprocessing + Model
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    
    # Fit
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)

    # Accuracy
    results[name] = accuracy_score(y_test, y_pred)
    
    # Metrics
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Final Results
print("\n✅ Model Accuracies:")
for model_name, acc in results.items():
    print(f"{model_name}: {acc:.4f}")