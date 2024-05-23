from flask import Flask, request
from flask import jsonify
import joblib
import pandas as pd
from flask_cors import CORS
model = None

app = Flask(__name__)

CORS(app)
def load_model():
    global model
    model  =  joblib.load('./ecommerce-predicter-model.pkl')



@app.route('/predict',methods=['POST'])
def predict():
    load_model()
    data  = pd.DataFrame(request.json)
    prediction = model.predict(data)
    print("this prediction ############## ",prediction)
    return jsonify({'prediction':list(prediction)})

@app.route('/',methods=['GET'])
def index():
    
    return "Welcome to Ecommerce Prediction API"

if __name__ == '__main__':
    
    app.run(port=4000,debug=True)

