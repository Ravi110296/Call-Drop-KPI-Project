import numpy as np
import pandas as pd
import pickle as pkl
from flask import Flask,render_template,json,jsonify,request

scalar = pkl.load(open('scalar.pkl','rb'))
model = pkl.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    data = np.array(list(data.values())).reshape(1,-1)
    data = scalar.transform(data)
    output = model.predict(data)
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    data = np.array(data).reshape(1,-1)
    data = scalar.transform(data)
    output = model.predict(data)
    return render_template('index.html',prediction_text = f'''The estimated call drop percentage
                            is {np.round(output[0],2)}''')

if __name__ == '__main__':
    app.run(debug = True, port=5000)