from flask import Flask, render_template, request
import numpy as np
import joblib
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict/', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        age = request.form.get('age')
        gender = request.form.get('gender')
        customer_type = request.form.get('customer_type')
        customer_class = request.form.get('customer_class')
        type_of_travel = request.form.get('type_of_travel')
        checkin_service = request.form.get('checkin_service')
        flight_distance = request.form.get('flight_distance')
        departure_delay_in_minutes = request.form.get('departure_delay_in_minutes')

        try:
            prediction = preprocessDataAndPredict(age,gender,customer_type,customer_class,type_of_travel,checkin_service,flight_distance,departure_delay_in_minutes)
            return render_template('predict.html', prediction = prediction)

        except ValueError:
            return "Please Enter valid values"

def preprocessDataAndPredict(age,gender,customer_type,customer_class,type_of_travel,checkin_service,flight_distance,departure_delay_in_minutes):
    input_data = [[age,gender,customer_type,customer_class,type_of_travel,checkin_service,flight_distance,departure_delay_in_minutes]]

    scaler = joblib.load("output/scaler.pkl")
    scaled_input_data = scaler.transform(input_data)

    classifier = joblib.load("output/xgbclassifier.pkl")
    prediction = classifier.predict(scaled_input_data)
   
    return prediction

if __name__ == '__main__':
    app.run(debug=True)