from flask import Flask, render_template, request, redirect, url_for, jsonify
from pymongo import MongoClient, errors
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTENC
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, fbeta_score
import time
import os
import pickle

app = Flask(__name__)

def cleaned_data_collection():
    connection = MongoClient('mongodb://localhost:27017')
    db = connection['FinalProject']
    cleaned_col = db['CleanedData']
    return cleaned_col

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collecting form inputs
    type = request.form.get('type')
    air_temp = float(request.form.get('air_temp'))
    process_temp = float(request.form.get('process_temp'))
    speed = float(request.form.get('speed'))
    torque = float(request.form.get('torque'))
    tool_wear = float(request.form.get('tool_wear'))

    # Validation
    if air_temp < 295 or air_temp > 305:
        return render_template('index.html', message='Air Temperature is not in the range of 295 to 305.')
    if process_temp < 310 or process_temp > 315:
        return render_template('index.html', message='Process temperature is not in the range of 310 to 315.')
    if speed < 1168 or speed > 2886:
        return render_template('index.html', message='Rotational speed is not in the range of 1168 to 2886.')
    if torque < 3 or torque > 77:
        return render_template('index.html', message='Torque is not in the range of 3 to 77.')
    if tool_wear < 0 or tool_wear > 253:
        return render_template('index.html', message='Tool wear is not in the range of 0 to 253.')

    # Prepare data for prediction
    input_data = [[type, air_temp, process_temp, speed, torque, tool_wear]]
    df = pd.DataFrame(input_data, columns=['Type', 'Air temperature', 'Process temperature', 'Rotational speed', 'Torque','Tool wear'])

    df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})

    sensor_readings = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque','Tool wear']
    with open(os.path.join('pickle', 'scaler.pkl'), 'rb') as f:
        sc = pickle.load(f) 
    df[sensor_readings] = sc.transform(df[sensor_readings])

    with open(os.path.join('pickle', 'binary_xgb.pkl'), 'rb') as f1:
        model1 = pickle.load(f1)    
    with open(os.path.join('pickle', 'multi_xgb.pkl'), 'rb') as f2:
        model2 = pickle.load(f2)

    target = model1.predict(df)
    failure_type = model2.predict(df)

    failure_type_name = 'No Failure' if failure_type == 0 else 'Power Failure' if failure_type == 1 else 'Tool Wear Failure' if failure_type == 2 else 'Overstrain Failure' if failure_type == 3 else 'Random Failure' if failure_type == 4 else 'Heat Dissipation Failure'
    
    if target[0] == 0:
        result_message = f'✅ No Failure: Machine is predicted to be functioning normally and got Failure Type as {failure_type_name}'
    else:
        result_message = f'⚠️ Failure: Machine is predicted to have failed and got Failure Type as {failure_type_name}'

    # Store in MongoDB
    cleaned_data = {
        'Type': int(df['Type'].values[0]),
        'Air temperature': df['Air temperature'].values[0],
        'Process temperature': df['Process temperature'].values[0],
        'Rotational speed': df['Rotational speed'].values[0],
        'Torque': df['Torque'].values[0],
        'Tool wear': df['Tool wear'].values[0],
        'Target': int(target[0]),
        'Failure Type': int(failure_type[0])
    }

    cleaned_col = cleaned_data_collection()
    try:
        result = cleaned_col.insert_one(cleaned_data)
        if result:
            return render_template('index.html', message=result_message, data=cleaned_data)
    except errors.DuplicateKeyError:
        return render_template('index.html', message='Data already exists in database')

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        retrain_msg = "Retraining in progress..."
        collection = cleaned_data_collection()
        result = list(collection.find())
        df = pd.DataFrame(result)

        with open(os.path.join('pickle', 'resample.pkl'), 'rb') as f:
            smote = pickle.load(f)
        X_res, y_res = smote.fit_resample(df, df['Failure Type'])

        X = X_res.drop(['Target', 'Failure Type'], axis=1)
        y = X_res[['Target', 'Failure Type']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y['Failure Type'], random_state=42)

        xgb = XGBClassifier(random_state=42)
        params = {
            'n_estimators': [300, 500, 700],
            'max_depth': [5, 7],
            'learning_rate': [0.01, 0.1],
            'objective': ['binary:logistic']
        }
        f2_scorer_binary = make_scorer(fbeta_score, pos_label=1, beta=2)
        binary_search = GridSearchCV(xgb, param_grid=params, cv=5, scoring=f2_scorer_binary)
        binary_search.fit(X_train, y_train['Target'])

        with open(os.path.join('pickle', 'binary_xgb.pkl'), 'wb') as f:
            pickle.dump(binary_search.best_estimator_, f)

        f2_scorer_mulit = make_scorer(fbeta_score, beta=2, average='weighted')
        multi_search = GridSearchCV(xgb, param_grid=params, cv=5, scoring=f2_scorer_mulit)
        multi_search.fit(X_train, y_train['Failure Type'])

        with open(os.path.join('pickle', 'multi_xgb.pkl'), 'wb') as f:
            pickle.dump(multi_search.best_estimator_, f)

        return render_template('index.html', message="Model retrained successfully")
    except Exception as e:
        return render_template('index.html', message=f"Error during retraining: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
