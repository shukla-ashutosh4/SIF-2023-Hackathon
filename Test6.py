import requests
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Function to fetch real-time weather data from Open-Meteo API
def fetch_real_time_data(latitude, longitude):
    api_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an error for bad responses
        
        data = response.json()
        
        # Extract relevant real-time weather features
        real_time_data = {
            'Precipitation': data['current_weather']['precipitation']['1h'],
            'Temperature': data['current_weather']['temperature_2m'],
            'Wind_Speed': data['current_weather']['wind']['speed_10m'],
            'Humidity': data['current_weather']['humidity_2m'],
            'Pressure': data['current_weather']['pressure_sea_level'],
            'Cloud_Cover': data['current_weather']['clouds']['cloudiness']
            # Add more features as needed
        }

        return real_time_data

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# Function to append real-time data to a CSV file
def append_real_time_data_to_csv(data, csv_path):
    # Check if the CSV file exists, if not, create it with headers
    if not os.path.exists(csv_path):
        headers = list(data.keys())
        headers.append('Timestamp')
        pd.DataFrame(columns=headers).to_csv(csv_path, index=False)

    # Append the real-time data along with a timestamp
    data['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df = pd.DataFrame([data])
    df.to_csv(csv_path, mode='a', header=False, index=False)

# Example coordinates (replace with actual coordinates)
latitude_location = 40.7128  # Fictional latitude for demonstration
longitude_location = -74.0060  # Fictional longitude for demonstration

# Fetch real-time data
real_time_data = fetch_real_time_data(latitude_location, longitude_location)

# Append real-time data to CSV file
csv_file_path = 'real_time_data.csv'
if real_time_data is not None:
    append_real_time_data_to_csv(real_time_data, csv_file_path)
    print("Real-time data appended to CSV successfully.")
else:
    print("Failed to fetch real-time data.")

# Load historical data (replace 'your_data.csv' with actual data)
try:
    data = pd.read_csv(csv_file_path)
except pd.errors.EmptyDataError:
    print("Error: The CSV file is empty or has no columns.")

# Define features and labels
features = data[['Precipitation', 'River_Discharge', 'Soil_Moisture', 'Elevation', 'Land_Use']]
labels = data['Flood_Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')

# Fetch real-time data again for prediction
real_time_data = fetch_real_time_data(latitude_location, longitude_location)

# Make a real-time prediction
if real_time_data is not None:
    new_observation = [
        [real_time_data['Precipitation'], real_time_data['Temperature'],
         real_time_data['Wind_Speed'], real_time_data['Humidity'],
         real_time_data['Pressure'], real_time_data['Cloud_Cover'],
         200.0, 1]  # Assuming fixed elevation and land use
    ]
    
    real_time_prediction = clf.predict(new_observation)

    # Display the result
    if real_time_prediction == 1:
        print("Flood Alert: High risk of flooding!")
    else:
        print("No Flood Alert: Conditions are within normal range.")
else:
    print("Failed to fetch real-time data.")
