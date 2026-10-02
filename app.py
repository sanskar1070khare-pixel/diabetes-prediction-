from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = "models/diabetes_pipeline.joblib"

model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    try:
        pregnancies = float(request.form["Pregnancies"])
        glucose = float(request.form["Glucose"])
        blood_pressure = float(request.form["BloodPressure"])
        skin_thickness = float(request.form["SkinThickness"])
        insulin = float(request.form["Insulin"])
        bmi = float(request.form["BMI"])
        diabetes_pedigree = float(
            request.form["DiabetesPedigreeFunction"]
        )
        age = float(request.form["Age"])

        input_data = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": diabetes_pedigree,
            "Age": age
        }])

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0][1]

        probability_percent = probability * 100

        if prediction == 1:
            result = "Higher Risk"
        else:
            result = "Lower Risk"

        return render_template(
            "index.html",
            prediction=result,
            probability=f"{probability_percent:.2f}"
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
