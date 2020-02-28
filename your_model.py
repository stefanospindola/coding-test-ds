# This is where the interfaces to your model should be. Please edit the functions below and add whatever is needed
# for your model to work
import joblib
import pandas as pd

def get_model():
    return joblib.load('model/lgbm.pkl')

def get_state_machine():
    return {'assessment_end_time': 0}

def get_features(state_machine, patient_id):
    features = ['arrival_time','assessment_start_time','assessment_end_time','temperature','priority','pain']
    df = pd.DataFrame(state_machine, index=[patient_id])
    df.temperature = df.temperature.astype('float64').round(1)
    df.priority = df.priority.astype('category').cat.codes
    df.pain = df.pain.astype('category').cat.codes
    return df

def get_estimate(model, features):
    return features.assessment_end_time + model.predict(features)

def update_state(state_machine, event):
    state_machine['patient'] = event.patient
    state_machine['day'] = event.day
    if event.event == 'arrived':
        state_machine['arrival_time'] = event.time     
    if event.event == 'assessment initiated':
        state_machine['assessment_start_time'] = event.time 
    if event.event == 'assessment concluded':
        state_machine['assessment_end_time'] = event.time
        state_machine['temperature'] = event.assessment.split("|")[1]
        state_machine['priority'] = event.assessment.split("|")[0]
        state_machine['pain'] = event.assessment.split("|")[2]