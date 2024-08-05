from flask import Flask, render_template, request
import requests
import time
import threading

app = Flask(__name__)

risk_level = 50
current_data = 0
flood_predicted = False

@app.route('/')
def home():
    return render_template('index.html', risk_level=risk_level, current_data=current_data, flood_predicted=flood_predicted)

def update_data():
    global current_data, flood_predicted
    while True:
        time.sleep(5)
        current_data = get_real_time_data()
        if current_data > risk_level:
            flood_predicted = True
        else:
            flood_predicted = False

def get_real_time_data():
    # In a real-world scenario, you would replace this with an API call to fetch the real-time data.
    return int(risk_level * 0.8)

if __name__ == '__main__':
    update_data_thread = threading.Thread(target=update_data)
    update_data_thread.start()
    app.run(debug=True)