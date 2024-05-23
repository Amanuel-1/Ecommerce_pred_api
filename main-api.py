from flask import Flask, request
from flask import jsonify
import joblib
import pandas as pd
model = None

app = Flask(__name__)
def load_model():
    global model
    model  =  joblib.load('./ecommerce-predicter-model.pkl')



@app.route('/predict',methods=['POST'])
def predict():
    data  = pd.DataFrame(request.json)
    prediction = model.predict(data)
    print("this prediction ############## ",prediction)
    return jsonify({'prediction':list(prediction)})

@app.route('/',methods=['GET'])
def index():
    return "Welcome to Ecommerce Prediction API"

if __name__ == '__main__':
    load_model()
    app.run(debug=True)

