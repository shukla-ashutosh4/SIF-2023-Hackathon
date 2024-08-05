# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the merged dataset
data = pd.DataFrame({
    'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10'],
    'Precipitation': [12.5, 8.2, 3.0, 5.8, 10.0, 15.2, 7.5, 4.2, 6.0, 8.9],
    'Max_Temperature': [25.2, 22.5, 20.0, 23.8, 18.7, 15.3, 17.5, 21.0, 19.6, 24.2],
    'Min_Temperature': [15.8, 13.0, 10.2, 12.5, 8.9, 7.2, 9.0, 11.8, 10.5, 14.7],
    'River_Discharge': [150.2, 180.5, 120.0, 200.3, 160.8, 230.1, 190.5, 210.0, 180.2, 195.7],
    'Water_Level': [2.5, 3.0, 2.2, 3.5, 2.8, 4.2, 3.9, 3.0, 2.7, 3.8],
    'Impervious_Percentage': [20.0, 40.5, 15.0, 25.0, 15.8, 22.3, 18.6, 20.2, 12.5, 30.0],
    'NDVI': [0.8, 0.6, 0.7, 0.65, 0.72, 0.75, 0.68, 0.78, 0.81, 0.73]
})

# Feature selection (X) and target variable (y)
X = data[['Precipitation', 'Max_Temperature', 'Min_Temperature', 'Impervious_Percentage', 'NDVI']]
y = data['River_Discharge']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model and train it
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize actual vs predicted river discharge
plt.scatter(X_test['Precipitation'], y_test, color='black', label='Actual')
plt.scatter(X_test['Precipitation'], y_pred, color='blue', label='Predicted')
plt.xlabel('Precipitation (mm)')
plt.ylabel('River Discharge (m³/s)')
plt.legend()
plt.show()
