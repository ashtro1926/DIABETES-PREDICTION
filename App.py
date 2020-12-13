from flask import Flask,render_template,request

import pandas as pd
import numpy as np
import pickle

File_Open=open('Diabetes.pkl','rb')
Model=pickle.load(File_Open)

app=Flask(__name__)

@app.route("/")
def Index():
    return render_template('Index.html')

@app.route("/Predict",methods = ["GET", "POST"])
def Predict():
    if request.method == "POST":

        Pregnancies=int(request.form["pregnant"])
        Glucose=int(request.form["glucose"])
        BloodPressure=int(request.form["blood"])
        SkinThickness=int(request.form["skin"])
        Insulin=int(request.form["insulin"])
        BMI=float(request.form["bmi"])
        DiabetesPedigreeFunction=float(request.form["dia"])
        Age=int(request.form["age"])

        Prediction=Model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        if Prediction == 0:
            return render_template("Result.html",prediction_text="You don't have Diabetes")
        else:
            return render_template("Result.html",prediction_text="You have Diabetes")
    return render_template("Index.html")






if __name__=="__main__":
    app.run(debug=True)
