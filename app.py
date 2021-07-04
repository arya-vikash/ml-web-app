from flask import Flask,request,render_template
import pickle
import numpy as np
from auto_mpg_model import predict_mpg
from wine_quality_model import predict_wine_quality
from fake_news_model import tfidf_predict
from spam_email_model import predict_spam

############## model pickle files##################
model_iris = pickle.load(open('iris.pkl', 'rb'))
model_fuel = pickle.load(open('auto_mpg.pkl', 'rb'))
model_wine = pickle.load(open('wine_quality.pkl','rb'))
model_fake_news = pickle.load(open('fake_news.pkl', 'rb'))

app=Flask(__name__)


@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')


################## fuel ###################################
@app.route('/fuel',methods=['GET'])
def fuel():
    return render_template('fuel.html')
@app.route('/fuel',methods=['POST'])
def predict_fuel():
    vehicle= {
    'Cylinders': [int(request.form['Cylinders'])],
    'Displacement': [int(request.form['Displacement'])],
    'Horsepower': [int(request.form['Horsepower'])],
    'Weight': [int(request.form['Weight'])],
    'Acceleration': [int(request.form['Acceleration'])],
    'Model Year': [int(request.form['Model Year'])],
    'Origin': [int(request.form['Origin'])]
    }

    predictions = predict_mpg(vehicle, model_fuel)
    return render_template('fuel.html',data=predictions)


############### iris #######################################
@app.route('/iris',methods=['GET'])
def iris():
    return render_template('iris.html')
@app.route('/iris',methods=['POST'])
def predict_iris():
    data1 = request.form['sepal_length']
    data2 = request.form['sepal_width']
    data3 = request.form['petal_length']
    data4 = request.form['petal_width']
    arr = np.array([[data1, data2, data3, data4]])
    pred = model_iris.predict(arr)
    return render_template('iris.html', data=pred)


############ wine ############################################
@app.route('/wine',methods=['GET'])
def wine():
    return render_template('wine.html')
@app.route('/wine',methods=['POST'])
def predict_wine():
    wine_values={'fixed acidity':[float(request.form['fixed acidity'])],
        'volatile acidity':[float(request.form['volatile acidity'])],
        'citric acid':[float(request.form['citric acid'])],
        'residual sugar':[float(request.form['residual sugar'])],
        'chlorides':[float(request.form['chlorides'])],
        'free sulfur dioxide':[float(request.form['free sulfur dioxide'])],
        'density':[float(request.form['density'])],
        'ph':[float(request.form['ph'])],
        'sulphates':[float(request.form['sulphates'])],
        'alcohol':[float(request.form['alcohol'])],
        'type':[request.form['type']]
    }

    predictions = predict_wine_quality(wine_values, model_wine)[0]
    return render_template('wine.html',data=predictions)

################ Fake News #################
@app.route('/fake_news')
def fake_news():
    return render_template('fake_news.html')


@app.route('/fake_news', methods=['POST'])
def predict_fake_news():
    data1 = request.form['news']
    y=tfidf_predict(data1)[0]
    
    return render_template('fake_news.html', data=y)


############## Spam email ####################################
@app.route('/spam_email')
def spam_email():
    return render_template('spam_email.html')

@app.route('/spam_email',methods=['POST'])
def predict_spam_email():
    data=request.form['email_text']
    y=predict_spam(data)
    return render_template('spam_email.html',data=y)
    


if __name__=='__main__':
    app.run()