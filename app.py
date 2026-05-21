import pandas as pd
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [20, 30, 40, 50, 60, 70, 80, 90]
}

df = pd.DataFrame(data)

# Input and Output
X = df[["Hours"]]
y = df["Marks"]

# Model Training
model = LinearRegression()
model.fit(X, y)

# Prediction
hours = float(input("Enter study hours: "))

prediction = model.predict([[hours]])

print("Predicted Marks:", prediction[0])