from flask import Flask, render_template, request
import numpy as np
import torch

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

modelo = torch.jit.load('models/iris_script.pt')
modelo.eval()

@app.route('/predict', methods=["POST"])
def predict():
    sepal_l = float(request.form['sepal_l'])
    sepal_w = float(request.form['sepal_w'])
    petal_l = float(request.form['petal_l'])
    petal_w = float(request.form['petal_w'])
    data = np.array([sepal_l, sepal_w, petal_l, petal_w])
    data_tensor = torch.FloatTensor(data)
    y_pred = modelo(data_tensor)
    result = torch.round(y_pred, decimals=2).tolist()
    return render_template('resultado.html', resultado = result, entrada=data)