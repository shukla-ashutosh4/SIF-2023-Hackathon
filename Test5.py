import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to fetch real-time data from a free weather API
def get_real_time_data(latitude, longitude):
    api_url = f"https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an error for bad responses (e.g., 4xx or 5xx status codes)

        data = response.json()

        # Check if the response is a boolean indicating an error
        if isinstance(data, bool):
            error_message = "Unknown error"
            print(f"Error: {error_message}")
            return None
        else:
            precipitation = data.get('current', {}).get('precipitation_sum_1h', 0)
            return precipitation

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# Example coordinates (you should replace these with the coordinates of your specific location)
latitude_location = '52.52'
longitude_location = '13.41'

# Fetch real-time data
current_precipitation = get_real_time_data(latitude_location, longitude_location)

# Example: Use precipitation as a simple feature for flood prediction
# In a real-world scenario, you'd need more features and a more sophisticated model.

# Assuming a threshold value for precipitation to predict flood
threshold_precipitation = 10.0

# Simple flood prediction model
def predict_flood(precipitation, threshold):
    if precipitation is not None:
        if precipitation > threshold:
            return True
        else:
            return False
    else:
        return None

# Make a prediction
flood_prediction = predict_flood(current_precipitation, threshold_precipitation)

# Display the result
if flood_prediction is not None:
    if flood_prediction:
        print("Flood Alert: High precipitation detected!")
    else:
        print("No Flood Alert: Precipitation is within normal range.")
