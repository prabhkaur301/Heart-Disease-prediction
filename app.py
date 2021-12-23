
from flask.templating import render_template_string
import numpy as np
import pandas as pd
import pickle
import joblib
from flask import Flask,render_template,request
# start flask
app = Flask(__name__)
model=joblib.load(open('model_pkl','rb'))


# render default webpage
@app.route('/')
def home():
    return render_template('./home.html')
@app.route('/form')
def form():
       return render_template('./form.html')
@app.route('/predict',methods=['POST'])
def predict():
   
     values=request.form.values()
     input_features = [float(str(i).replace(",", "")) for i in values]
     features_value = [np.array(input_features)]
     features_name=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
    
     df=pd.DataFrame(features_value,columns=features_name)
     output=model.predict(df)
    
     if output==1:
       output=True
     else:
       output=0
       
     return render_template('predict.html',prediction_text=output)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port="33")
    



