from flask import Flask,request, url_for, redirect, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)
pickle_in=open('classifier.pkl','rb')
clf=pickle.load(pickle_in)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features=[x for x in request.form.values()]
    final=np.array(int_features).reshape(1,-1)
    prediction=clf.predict(final)
    if (prediction==0):
        return render_template('home.html',pred='Species Type will be Setosa')
    elif (prediction==1):
        return render_template('home.html',pred='Species Type will be Versicolor')
    else:
        return render_template('home.html',pred='Species Type will be Virginica')          

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = np.array([data]).reshape(1,-1)
    prediction = clf.predict(data_unseen)
    return prediction

if __name__ == '__main__':
    app.run(debug=True)
