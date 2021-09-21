import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

#prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,8)
    loaded_model = pickle.load(open("pizza.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

#Output page and Logic
@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        
        return render_template('result.html', prediction = result)
    
          
# Main Function
if __name__ == '__main__':
    app.run(debug=True)
    app.config['TEMPLATES_AUTO_RELOAD']=True
