import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, confusion_matrix

# Load data
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn_df = pd.read_csv(url)

# Select relevant columns
cols = ['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']
churn_df = churn_df[cols].copy()
churn_df['churn'] = churn_df['churn'].astype(int)

# Features/target
X = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']].to_numpy()
y = churn_df['churn'].to_numpy()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4, stratify=y
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train
LR = LogisticRegression(max_iter=1000).fit(X_train, y_train)
yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)

print("Accuracy:", LR.score(X_test, y_test))
print("Log loss:", log_loss(y_test, yhat_prob))
print("Confusion matrix:\n", confusion_matrix(y_test, yhat))
