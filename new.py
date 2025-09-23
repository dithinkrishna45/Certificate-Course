import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve

# ------------------------------
# Load dataset
# ------------------------------
df = pd.read_csv("Divorce Prediction/marriage_data.csv")

# Separate features and target
X = df.drop("divorced", axis=1)
y = df["divorced"]

# Encode categorical features
label_encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# Random Forest Model
# ------------------------------
rf = RandomForestClassifier(
    n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print("\n=== Random Forest Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_rf))
print(classification_report(y_test, y_pred_rf))

# ------------------------------
# ROC Curve
# ------------------------------
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={roc_auc_score(y_test, y_proba_rf):.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.show()

# ------------------------------
# Feature Importance
# ------------------------------
plt.figure(figsize=(10, 5))
importances_rf = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
importances_rf.plot(kind="bar", title="Top 10 Random Forest Feature Importances")
plt.show()

# ------------------------------
# User Input Prediction
# ------------------------------
print("\n=== Divorce Prediction from User Input ===")
user_data = {}
for col in X.columns:
    if col in label_encoders:  # categorical
        categories = list(label_encoders[col].classes_)
        val = input(f"Enter {col} ({categories}): ")
        user_data[col] = label_encoders[col].transform([val])[0]
    else:  # numerical
        val = float(input(f"Enter {col} (numeric): "))
        user_data[col] = val

user_df = pd.DataFrame([user_data])

# Prediction
pred_rf = rf.predict_proba(user_df)[0][1]

print("\nPredicted Probability of Divorce (Random Forest): {:.2f}".format(pred_rf))
