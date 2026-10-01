# Diabetes Prediction — Complete B.Tech AI/ML Project

## Important dataset note
`data/diabetes_demo_dataset.csv` is a **synthetic demonstration dataset generated for this ZIP**, not the real Pima Indians Diabetes dataset. For your final academic project, replace it with a properly sourced dataset such as the Pima dataset and rename it to `diabetes.csv`, then update `DATA_PATH` in `train_model.py`.

## Run
1. Open this folder in VS Code.
2. Create a virtual environment (optional but recommended).
3. Install packages:
   `pip install -r requirements.txt`
4. Train:
   `python train_model.py`
5. Start backend:
   `python app.py`
6. Open `http://127.0.0.1:5000` in your browser.

## Project structure
data/ — dataset
models/ — trained model is created here
templates/ — HTML
static/ — CSS
train_model.py — training
app.py — Flask backend
requirements.txt — dependencies
