import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset (example local CSV)
df = pd.read_csv("house_data.csv")

# EDA
print(df.head())
print(df.info())
print(df.describe())

# Feature & Target
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("R2 Score:", r2_score(y_test, y_pred))
