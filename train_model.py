import os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

DATA_PATH="data/diabetes_demo_dataset.csv"
MODEL_PATH="models/diabetes_pipeline.joblib"
df=pd.read_csv(DATA_PATH)
target="Outcome"
features=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
zero_as_missing=["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
for col in zero_as_missing: df[col]=df[col].replace(0,np.nan)
df = pd.read_csv(DATA_PATH)

# Remove rows where the target value is missing
df = df.dropna(subset=["Outcome"])

X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X,y=df[features],df[target]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
numeric_pipe=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
preprocessor=ColumnTransformer([("numeric",numeric_pipe,features)])
models={
"Logistic Regression":LogisticRegression(max_iter=2000),
"Decision Tree":DecisionTreeClassifier(max_depth=5,random_state=42),
"Random Forest":RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced"),
"SVM":SVC(probability=True,random_state=42)}
results={}
for name,clf in models.items():
    pipe=Pipeline([("preprocessor",preprocessor),("classifier",clf)])
    pipe.fit(X_train,y_train); pred=pipe.predict(X_test)
    results[name]={"accuracy":accuracy_score(y_test,pred),"precision":precision_score(y_test,pred,zero_division=0),
                   "recall":recall_score(y_test,pred,zero_division=0),"f1":f1_score(y_test,pred,zero_division=0),"pipeline":pipe}
    print("\n",name); print(classification_report(y_test,pred,zero_division=0))
best=max(results,key=lambda n:results[n]["f1"])
os.makedirs("models",exist_ok=True)
joblib.dump(results[best]["pipeline"],MODEL_PATH)
print("Selected model:",best,"\nSaved:",MODEL_PATH)
