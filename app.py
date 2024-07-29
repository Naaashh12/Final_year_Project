import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for, request
import sqlite3
import cv2
import shutil

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('home.html')

    return render_template('index.html')

@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"

        shutil.copy("testing\\"+fileName, dst)
        
        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'poverty-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            for img in os.listdir(verify_dir):
                path = os.path.join(verify_dir, img)
                img_num = img.split('.')[0]
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                verifying_data.append([np.array(img), img_num])
                np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 3, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('Model Loaded!')


        accuracy=" "
        str_label=" "
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            #y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            if np.argmax(model_out) == 0:
                str_label = 'Poverty Level is Low'
                accuracy = "The predicted poverty of the satellite image is low with a accuracy of {} %".format(model_out[0]*100)
           
            elif np.argmax(model_out) == 1:
                str_label = 'Poverty Level is Medium'
                accuracy = "The predicted povery of the satellite image is high with a accuracy of {} %".format(model_out[1]*100)
                
            elif np.argmax(model_out) == 2:
                str_label = 'Poverty Level High'
                
                accuracy = "The predicted poverty of the satellite image is low with a accuracy of {} %".format(model_out[2]*100)
           

        return render_template('result.html', status=str_label,accuracy=accuracy, ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName)
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.form
        values = []
        for key in data:
            try:
                values.append(float(data[key]))
            except:
                values.append(data[key])
        print(len(values))
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import mean_squared_error

        # Load dataset
        df = pd.read_csv("city_data.csv")

        # Define features and target variable
        X = df.drop(columns=["City"])  # Features
        y = df["Wealth Index"]  # Target variable (poverty level)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define preprocessing steps
        numeric_features = X.select_dtypes(include=['int64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        numeric_transformer = Pipeline(steps=[
            ('num', 'passthrough')
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Append classifier to preprocessing pipeline
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LinearRegression())])

        # Fit the model
        clf.fit(X_train, y_train)

        # Get user input for a new city
        new_city_data = pd.DataFrame([values[1:]],columns=X.columns)

        # Predict poverty level for the new city
        predicted_poverty_level = clf.predict(new_city_data)
        str_label = f"Predicted Poverty Level:{predicted_poverty_level[0]}"
        print(str_label)
        return render_template('result.html', status2=str_label)
    return render_template('home.html')

@app.route('/predict2', methods=['GET', 'POST'])
def predict2():
    if request.method == 'POST':
        data = request.form
        values = []
        for key in data:
            try:
                values.append(float(data[key]))
            except:
                values.append(data[key])
        print(values)

        if values[1] < 5:
            str_label = 'Poverty due to Wealth Index less than 5'
        elif values[2] < 100:
            str_label = 'Poverty due to no. of Hospital less than 100'
        elif values[3] < 100:
            str_label = 'Poverty due to no. of Schools less than 100'
        elif values[6] == 'Scarce':
            str_label = 'Poverty due to Scarce Water Resource'
        elif values[8] > 1000:
            str_label = 'Poverty due to no. of Unemployment greater than 1000'
        else:
            str_label = 'No Poverty'
        return render_template('result.html', status2=str_label)
    return render_template('home.html')

if __name__ == "__main__":

    app.run(debug=True, use_reloader=False)