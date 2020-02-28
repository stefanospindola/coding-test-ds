# This is where the interfaces to your model should be. Please edit the functions below and add whatever is needed
# for your model to work
import joblib
import pandas as pd
from train import number_out, 

def get_model():
    return joblib.load('model/lgbm.pkl')

def get_state_machine():
    return {'assessment_end_time': 0}

def get_features(state_machine, patient_id):
    features = ['assessment_end_time',
                'temperature',
                'priority',
                'pain']
    data = pd.DataFrame(state_machine, index=[patient_id])
    df = pd.concat(data)
    df['patient_number'] = df.groupby('day').cumcount() + 1
    
    df['priority_cumcount'] = df.groupby(['day','priority']).cumcount()+1
    df['pain_cumcount'] = df.groupby(['day','pain']).cumcount()+1
    df['priority_pain_cumcount'] = df.groupby(['day','priority','pain']).cumcount()+1
 
    df.temperature = df.temperature.astype('float64').round(1)
    df.priority = df.priority.astype('category').cat.codes
    df.pain = df.pain.astype('category').cat.codes
    patient_out = number_out(df, 'assessment_end_time', 'consultation_end_time')
    df['patient_number_in'] = [i-j for i,j in zip(df.patient_number.to_list(),patient_out)]
    number_patient_enter_consultation_since_assessment_end = number_out(df, 'assessment_end_time', 'consultation_start_time')
    df['patient_number_waiting_consulting'] = [i-j for i,j in zip(df.patient_number.to_list(),number_patient_enter_consultation_since_assessment_end)]
    df['doctor_free'] = df.apply(lambda x: (6 - (x.patient_number_in - x.patient_number_waiting_consulting)), axis = 1)
  
    return df[features]

def get_estimate(model, features):
    return features.assessment_end_time + model.predict(features)

def update_state(state_machine, event):
    state_machine['assessment_end_time'] = event.time
    state_machine['temperature'] = event.assessment.split("|")[1]
    state_machine['priority'] = event.assessment.split("|")[0]
    state_machine['pain'] = event.assessment.split("|")[2]
    state_machine['patient'] = event.patient
    state_machine['day'] = event.day