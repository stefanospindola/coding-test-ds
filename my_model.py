import pandas as pd
import joblib

def get_model():
    return joblib.load('model/lgbm.pkl')

def get_state_machine():
    return {'assessment_concluded': 0}

def get_features(state_machine, patient_id):
    features = ['assessment_concluded',
                'temperature',
                'priority',
                'pain']
    df = pd.DataFrame(state_machine, index=[patient_id])
    df['temperature'] = df.assessment.str.split("|", n = 2, expand = True)[1].astype('float64').round(1)
    df['priority'] = df.assessment.str.split("|", n = 2, expand = True)[0]
    df['pain'] = df.assessment.str.split("|", n = 2, expand = True)[2]
    df.priority = df.priority.astype('category').cat.codes
    df.pain = df.pain.astype('category').cat.codes
    return df[features]

def get_estimate(model, features):
    return features.assessment_concluded + model.predict(features)

def update_state(state_machine, event):
    state_machine['assessment_concluded'] = event.time
    state_machine['assessment'] = event.assessment
    state_machine['patient'] = event.patient
    state_machine['day'] = event.day