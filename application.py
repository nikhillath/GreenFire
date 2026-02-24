from flask import Flask,request,jsonify,render_template
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application


ridge_model=pickle.load(open('models/ridge_model.pkl','rb'))
scaler=pickle.load(open('models/scaler.pkl','rb'))

print("Scaler expects:", scaler.n_features_in_)
print("Model expects:", ridge_model.n_features_in_)


@app.route("/")
def home():
    return render_template('home.html')


@app.route('/predictData',methods=['GET','POST'])
def predictFWI():
    if request.method=="POST":
        
        Temperature=float(request.form.get("Temperature"))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI= float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        
        new_scaled_data=scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_scaled_data)

        return render_template('predict.html',results=result[0])
    else:
        return render_template('predict.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")