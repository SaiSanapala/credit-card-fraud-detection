# model/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("../data/creditcard.csv")

# Separate features and labels
X = df.drop(['Class', 'Time'], axis=1)
y = df['Class']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

# Apply SMOTE to training set
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# Train model (try RandomForest or XGBoost)
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_res, y_res)

# Evaluate
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
plt.savefig("precision_recall_curve.png")
plt.close()

# Include original Amount column (not scaled)
original_test_df = df.iloc[y_test.index]
flagged = original_test_df.copy()
flagged['Prediction'] = y_pred
flagged = flagged[flagged['Prediction'] == 1]
flagged.to_csv("../data/flagged_frauds.csv", index=False)
