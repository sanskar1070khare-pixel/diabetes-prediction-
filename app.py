from flask import Flask, render_template, request, jsonify
import joblib, pandas as pd, math
app=Flask(__name__)
model=joblib.load("models/diabetes_pipeline.joblib")
FEATURES=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
def validate(values):
    data={}
    for f in FEATURES:
        if f not in values: raise ValueError(f"Missing field: {f}")
        v=float(values[f])
        if not math.isfinite(v): raise ValueError(f"Invalid number for {f}")
        data[f]=v
    return data
@app.route("/")
def home(): return render_template("index.html")
@app.route("/predict",methods=["POST"])
def predict():
    try:
        data=validate(request.form.to_dict()); X=pd.DataFrame([data])
        pred=int(model.predict(X)[0])
        prob=float(model.predict_proba(X)[0][1]) if hasattr(model,"predict_proba") else None
        label="Higher-risk prediction" if pred else "Lower-risk prediction"
        return render_template("index.html",prediction=label,probability=prob)
    except Exception as e: return render_template("index.html",error=str(e)),400
@app.route("/api/predict",methods=["POST"])
def api_predict():
    data=validate(request.get_json(force=True)); X=pd.DataFrame([data]); pred=int(model.predict(X)[0])
    r={"prediction":pred,"label":"Higher-risk prediction" if pred else "Lower-risk prediction"}
    if hasattr(model,"predict_proba"): r["probability"]=float(model.predict_proba(X)[0][1])
    return jsonify(r)
if __name__=="__main__": app.run(debug=True)
