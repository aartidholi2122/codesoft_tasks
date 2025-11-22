import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("creditcard.csv")

print("Columns loaded:", df.columns)

# Features and target
X = df.drop('Class', axis=1)    # Class = 1 (Fraud), 0 (Normal)
y = df['Class']

# Scale the amount and time (optional but recommended)
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Print performance report
print("\nClassification Report:")
print(classification_report(y_test, pred))