from flask import Flask, request, jsonify
from geopy.geocoders import Nominatim
from flask_cors import CORS
import requests
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import os


app = Flask(__name__)
CORS(app) 

MODEL_PATH = os.getenv("MODEL_PATH", "fire_size_classifier.h5")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "preprocessor_fireclass.pkl")
LABEL_ENCODER_PATH = os.getenv("LABEL_ENCODER_PATH", "label_encoder.pkl")

model = tf.keras.models.load_model(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")

geolocator = Nominatim(user_agent=os.getenv("GEOLOCATOR_USER_AGENT", "count_locator_app"))

CORS(app, resources={r"/*": {"origins": "https://main.d3kerizsq6l6h9.amplifyapp.com"}})


cause_mapping = {
    "Lightning": 0,
    "Debris Burning": 1,
    "Campfire": 2,
    "Other": 3
}

def reverse_geocode(lat, lon):
    """Use geopy to determine the county from latitude/longitude."""
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True, addressdetails=True)
        if location and "county" in location.raw["address"]:
            return location.raw["address"]["county"]
        return None
    except Exception as e:
        print(f"Reverse geocoding error: {e}")
        return None

def predict_fire_size(county, month, year, cause, latitude, longitude):
    cause_mapping = {
        "Lightning": 1.0,
        "Equipment Use": 2.0,
        "Smoking": 3.0,
        "Campfire": 4.0,
        "Debris Burning": 5.0,
        "Railroad": 6.0,
        "Arson": 7.0,
        "Children": 8.0,
        "Miscellaneous": 9.0
    }

    input_data = pd.DataFrame({
        "FIRE_YEAR": [year],
        "LATITUDE": [latitude],
        "LONGITUDE": [longitude],
        "MONTH_sin": [np.sin(2 * np.pi * month / 12)],
        "MONTH_cos": [np.cos(2 * np.pi * month / 12)],
        "STAT_CAUSE_CODE": [cause_mapping[cause]]
    })

    input_transformed = preprocessor.transform(input_data)
    predicted_class_index = np.argmax(model.predict(input_transformed))
    fire_size_class = label_encoder.inverse_transform([predicted_class_index])[0]
    return fire_size_class


#Groq API that is called once we predict the fire_size_class we get
def get_fire_mitigation_recommendation(county, month, year, cause, fire_size_class):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    fire_severity = {
        "A": "Minimal fire, local containment is sufficient.",
        "B": "Small fire, may require regional response.",
        "C": "Moderate fire, requires coordinated suppression efforts.",
        "D": "Large fire, state-level intervention may be needed.",
        "E": "Very large fire, multiple agencies required.",
        "F": "Extreme fire, national-level response advised.",
        "G": "Catastrophic fire, mass evacuations and air support required."
    }

    prompt = (
        f"Explicitly state that a fire was discovered in {county} county in {month} {year}. Mention the circumstances behind the report indicate it will be a class {fire_size_class} fire which has the description {fire_severity[fire_size_class]}."
        f"Explicitly state that the reported cause is {cause}. Estimate the cost to put out based on historical data and the {fire_size_class} class of fire."
        f"Numerically list 3 fire departments in that county and tell me what steps the fire departments should take to minimize damage. "
        f"Use the context of the cause and the county/area. (ex: how urban the area is, demographic of residents) "
        f"Give 3 tips to reduce the likelihood and damage of the fires in the future (ex: trim certain kinds of vegetation, limit certain behaviors)."
        f"Do not give extra information. Give you answer in a minimalist format with bullet points."
    )

    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are an expert in wildfire prevention and emergency response."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code == 200:
        mitigation_plan = response.json()["choices"][0]["message"]["content"]
        return mitigation_plan
    else:
        return f"Error: {response.text}"

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://main.d3kerizsq6l6h9.amplifyapp.com"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response
    
@app.route('/predict', methods=['OPTIONS'])
def handle_options():
    return add_cors_headers(jsonify({"message": "CORS preflight handled"}))


@app.route('/predict', methods=['POST'])
def predict_fire():
    data = request.json
    county = data.get("county")
    month = data.get("month")
    year = data.get("year")
    cause = data.get("cause")
    latitude = data.get("latitude")
    longitude = data.get("longitude")

    if not latitude and not longitude:
        coordinates = geocode(county)
        latitude = coordinates[0]
        longitude = coordinates[1]
    
    if latitude and longitude and not county:
        county = reverse_geocode(latitude, longitude)
        if not county:
            return jsonify({"error": "Could not determine county from coordinates"}), 400

    fire_size_class = predict_fire_size(county, month, year, cause, latitude, longitude)
    print("PREDICTION:", fire_size_class)
    mitigation_plan = get_fire_mitigation_recommendation(county, month, year, cause, fire_size_class)

    response = jsonify({
        "predicted_fire_size_class": fire_size_class,
        "mitigation_plan": mitigation_plan
    })

    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")

    return response

def geocode(county):
    if not county:
        return {'error': 'No county provided'}

    location = geolocator.geocode(county)
    if location:
        return [location.latitude, location.longitude]
    else:
        return {'error': 'County not found'}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
