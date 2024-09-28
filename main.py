import streamlit as st
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

def cleaned_data_collection():
    connection = MongoClient('mongodb://localhost:27017')
    db = connection['FinalProject']
    cleaned_col = db['CleanedData']
    return cleaned_col

def retrain_model():
    try:
        retrain_msg = st.empty()
        retrain_msg.success("Retraining inprogress...")

        collection = cleaned_data_collection()
        result = list(collection.find())
        df = pd.DataFrame(result)

        with open(os.path.join('pickle', 'resample.pkl'), 'rb') as f:
            smote = pickle.load(f)
        X_res, y_res = smote.fit_resample(df, df['Failure Type'])

        X = X_res.drop(['Target', 'Failure Type'], axis=1)
        y = X_res[['Target', 'Failure Type']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y['Failure Type'], random_state=42)

        binary_retrain_msg = st.empty()
        binary_retrain_msg.success("Retraining Binary Model with XGBClassifier...")

        xgb = XGBClassifier(random_state=42),
        params = {
                'n_estimators':[300,500,700],
                'max_depth':[5,7],
                'learning_rate':[0.01,0.1],
                'objective':['binary:logistic']
            }
        f2_scorer_binary = make_scorer(fbeta_score, pos_label=1, beta=2)
        binary_search = GridSearchCV(xgb, param_grid=params, cv=5, scoring=f2_scorer_binary)
        binary_search.fit(X_train, y_train['Target'])

        with open(os.path.join('pickle', 'binary_xgb.pkl'),'wb') as f:
            pickle.dump(binary_search.best_estimator_, f)

        f2_scorer_mulit= make_scorer(fbeta_score, beta=2, average='weighted')
        multi_search = GridSearchCV(xgb, param_grid=params, cv=5, scoring=f2_scorer_mulit)
        multi_search.fit(X_train, y_train['Failure Type'])

        with open(os.path.join('pickle', 'multi_xgb.pkl'),'wb') as f:
            pickle.dump(multi_search.best_estimator_, f)
        
        return "Model retrained successfully"
    except Exception as e:
        return f"Error during retraining {str(e)}"

st.set_page_config(
    page_title="Predictive Maintenance for Manufacturing Equipment",
    page_icon="🔩",
    layout="wide",
)

st.markdown("<h1 style = 'text-align: center'>Predictive Maintenance for Manufacturing Equipment</h1>", unsafe_allow_html=True)

st.markdown("""
    <style>
    .prediction-success {
        background-color: #d4edda;
        color: #155724;
        padding: 20px;
        border-radius: 10px;
    }
    .prediction-failure {
        background-color: #f8d7da;
        color: #721c24;
        padding: 20px;
        border-radius: 10px;
    }
    .failure-type {
        background-color: #fff3cd;
        color: #856404;
        padding: 15px;
        margin-top: 10px;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

success = False
with st.form("my_form"):
    type = st.selectbox("Select Product Quality Type", ["H","M", "L"])
    air_temp = st.number_input("Enter the Air temperature value [K] (should be between 295 and 305)")
    process_temp = st.number_input("Enter Process temperature [K] (should be between 310 and 315)")
    speed = st.number_input("Enter Rotational speed [rpm] (should be between 1168 and 2886)")
    torque = st.number_input("Enter Torque [Nm] (should be between 3 and 77)")
    tool_wear = st.number_input("Enter Tool wear [min] (should be between 0 and 253)")

    submitted = st.form_submit_button("Predict")
if submitted:
    if  (air_temp < 295 or air_temp > 305):
        st.warning('Air Temperature is not in the range of 295 to 305. Retraining will not proceed')
    elif  (process_temp < 305 or process_temp > 315):
        st.warning('Process temperature is not in the range of 305 to 315. Retraining will not proceed')
    elif  (speed < 1168 or speed > 2886):
        st.warning('Rotational speed is not in the range of 1168 to 2886. Retraining will not proceed')
    elif  (torque < 3 or torque > 77):
        st.warning('Torque is not in the range of 3 to 77. Retraining will not proceed')
    elif  (tool_wear < 0 or tool_wear > 253):
        st.warning('Tool wear is not in the range of 0 to 253. Retraining will not proceed')
    else:
        message_placeholder = st.empty()
        message_placeholder.success("Form submitted successfully")
        time.sleep(3)
        message_placeholder.empty()
        success = True

if success:

    input = [[type, air_temp, process_temp, speed, torque, tool_wear]]
    df = pd.DataFrame(input, columns=['Type', 'Air temperature', 'Process temperature', 'Rotational speed', 'Torque','Tool wear'])

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
        st.markdown(f'<div class="prediction-success">✅ <strong>No Failure</strong>: Machine is predicted to be functioning normally and got Failure Type as {failure_type_name}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="failure-type">⚠️ <strong>Failure</strong>: Machine is predicted to have failed and got Failure Type as {failure_type_name}</div>', unsafe_allow_html=True)

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
            st.success('Data transformed and inserted into DB successfully and ready for retraining')

            st.write('Inserted Data:')
            st.write(cleaned_data)

            st.write('Retraining inprogress...')
            try:
                collection = cleaned_data_collection()
                result = list(collection.find())
                final_df = pd.DataFrame(result)
                final_df = final_df.drop('_id',axis=1)

                no_failure = int((final_df['Failure Type'] == 0).sum())
                minority_class = final_df['Failure Type'].nunique() - 1
                minority_percent = 0.20
                final_count = no_failure / (1 - minority_percent)
                sample_per_class = round((final_count - no_failure) / minority_class)
                sample_strategy = {0 : no_failure}
                for i in range(1, minority_class+1):
                    sample_strategy[i] = sample_per_class

                smote = SMOTENC(categorical_features=[0,6], sampling_strategy=sample_strategy, random_state=42)
                X_res, y_res = smote.fit_resample(final_df, final_df['Failure Type'])

                st.write('Oversampling Completed')
                
                
                X = X_res.drop(['Target', 'Failure Type'], axis=1)
                y = X_res[['Target', 'Failure Type']]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y['Failure Type'], random_state=42)

                st.write("Retraining Binary Model with XGBClassifier...")

                xgb = XGBClassifier(random_state=42)
                params = {
                        'n_estimators':[300,500,700],
                        'max_depth':[5,7],
                        'learning_rate':[0.01,0.1],
                        'objective':['binary:logistic']
                    }
                f2_scorer_binary = make_scorer(fbeta_score, pos_label=1, beta=2)
                binary_search = GridSearchCV(xgb, param_grid=params, cv=5, scoring=f2_scorer_binary)
                binary_search.fit(X_train, y_train['Target'])

                with open(os.path.join('pickle', 'binary_xgb.pkl'),'wb') as f:
                    pickle.dump(binary_search.best_estimator_, f)

                st.write("Retraining Multiclass Model with XGBClassifier...")

                f2_scorer_mulit= make_scorer(fbeta_score, beta=2, average='weighted')
                multi_search = GridSearchCV(xgb, param_grid=params, cv=5, scoring=f2_scorer_mulit)
                multi_search.fit(X_train, y_train['Failure Type'])

                with open(os.path.join('pickle', 'multi_xgb.pkl'),'wb') as f:
                    pickle.dump(multi_search.best_estimator_, f)
                
                st.success("Model retrained successfully")
            except Exception as e:
                st.error(f"Error during retraining {str(e)}")

            # retrain = st.button('Retrain Model')
            
            # if retrain:
            #     message = retrain_model()
            #     st.info(message)

    except errors.DuplicateKeyError:
        st.error('Data already exists in database')


    


