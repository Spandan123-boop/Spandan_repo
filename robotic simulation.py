# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Step 1: Load the dataset
df = pd.read_csv(r"C:\Users\spand\Documents\sensor_data.csv")  # Ensure the CSV is in the same directory

# Step 2: Prepare data for training
X = df[['Voltage']]  # Input feature: Voltage
y = df['Temperature']  # Output label: Temperature

# print(np.max(df['Temperature']))
# print(np.max(df['Voltage']))

# print(x_modified)

# Step 3: Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Step 4: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict on test data
print(model.predict(X_test))
print(model.score(X_test,y_test))
# Step 6: Calculate Efficiency Score (R² Score)
# efficiency_score = r2_score(X_test,y_test)
# print(f"Model Efficiency (R² Score): {efficiency_score:.4f}")

# Step 7: Display Model Parameters (Equation: y = mx + c)
# slope = model.coef_[0]
# intercept = model.intercept_
# print(f"Equation: Temperature = {slope:.2f} * Voltage + {intercept:.2f}")

#Step 8: Visualize Results (Actual vs Predicted)
plt.scatter(X_test, y_test, color='blue', label="Actual Data")
plt.plot(X_test, y_test, color='red', linewidth=2, label="Predicted Line")
plt.xlabel("Sensor Voltage (V)")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Prediction using Linear Regression")
plt.legend()
plt.grid(True)
plt.show()

# Step 9: Predict Temperature for a New Voltage Input
# new_voltage = np.array([[1.5]])  # Example input: 1.5V
# predicted_temp = model.predict(new_voltage)
# print(f"Predicted Temperature for 1.5V: {predicted_temp[0]:.2f}°C")
