import subprocess
import json

def get_weather(city, latitude, longitude):
    # Construct the curl command
    curl_command = f'curl "https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"'

    try:
        # Execute the curl command and capture the output
        process = subprocess.Popen(curl_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        output, error = process.communicate()

        # Decode the output to a string
        output_str = output.decode('utf-8')

        # Check for errors
        if process.returncode == 0:
            # Parse the JSON response
            data = json.loads(output_str)

            # Access the desired information from the response
            current_temperature = data.get('current', {}).get('temperature_2m', 'N/A')
            wind_speed = data.get('current', {}).get('wind_speed_10m', 'N/A')

            print(f"Current weather in {city}:")
            print(f"Temperature: {current_temperature}°C")
            print(f"Wind Speed (10m): {wind_speed} m/s")
        else:
            print(f"Error: {error.decode('utf-8')}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Coordinates for Berlin
latitude_berlin = '52.52'
longitude_berlin = '13.41'
city_berlin = 'Berlin'

get_weather(city_berlin, latitude_berlin, longitude_berlin)
