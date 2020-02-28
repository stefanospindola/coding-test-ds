import warnings
warnings.filterwarnings("ignore")

import argparse
import logging
from os import listdir, path

import pandas as pd
import numpy as np
import datetime as dt

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
import joblib

from file_evaluation import evaluate_file


logger = logging.getLogger()
features = ['arrival_time','assessment_start_time','assessment_end_time','temperature','priority','pain']
# features = ['arrival_time','assessment_start_time','assessment_end_time','temperature','priority','pain','patient_number_in','priority_cumcount','pain_cumcount','priority_pain_cumcount','patient_number_waiting_consulting','doctor_free']
label = 'target'


def main():
    args = get_arguments()
    set_logger(args.debug)
    df = pd.concat([pd.read_csv(file_path,index_col=0) for  file_path in get_files_in_folder(args.folder_name)])
    df = summary(df)
    train, test = train_test_for_model(df, 5)
    lgbm, lgb = train_lgbm(train[features], train[label])
    save_model(lgbm)
    predictions = predict_lgbm(lgbm, test[features])
    # print rmsle
    print("RMSlE of the validation set:",rmsle(test[label], predictions))
    print("RMSlE of the validation set:",dt.timedelta(seconds=rmsle(test[label], predictions)))

    # errors = np.array([])
    # for file_path in get_files_in_folder(args.folder_name):
    #     file_errors = evaluate_file(file_path)
    #     errors = np.append(errors, file_errors)
    # score = get_rms(errors)
    # logger.info('Your score was %f', score)

def number_out(df, time_now, time_out):
    patient_out = []
    for k in range(1,51):
        for (i, row) in df[df.day==k].iterrows():
            n=0
            for (j, row) in df[df.day==k].iterrows():
                if df[df.day==k].loc[j,time_out] <= df[df.day==k].loc[i,time_now]:
                    n+=1
            patient_out.append(n)
    return patient_out

def summary(df):
    
    df['event_number'] = (df.groupby(['day','patient']).cumcount() + 1).astype(str)
    df = df.set_index(['patient','event_number']).unstack()
    df.sort_index(axis = 1,level = 1, inplace=True)
    df.columns = ['_'.join(col) for col in df.columns]
    df.reset_index(inplace=True)
    df.rename(columns={"time_1": "arrival_time", 
                       "time_2": "assessment_start_time",
                       "time_3": "assessment_end_time",
                       "time_4": "consultation_start_time",
                       "time_5": "consultation_end_time",
                       "day_1": "day"}, inplace = True)  
    df['temperature'] = df["assessment_3"].str.split("|", n = 2, expand = True)[1].astype('float64').round(1)
    df['priority'] = df["assessment_3"].str.split("|", n = 2, expand = True)[0]
    df['pain'] = df["assessment_3"].str.split("|", n = 2, expand = True)[2]
    # df['patient_number'] = df.groupby('day').cumcount() + 1
    # df['priority_cumcount'] = df.groupby(['day','priority']).cumcount()+1
    # df['pain_cumcount'] = df.groupby(['day','pain']).cumcount()+1
    # df['priority_pain_cumcount'] = df.groupby(['day','priority','pain']).cumcount()+1
    df.priority = df.priority.astype('category').cat.codes
    df.pain = df.pain.astype('category').cat.codes
    # patient_out = number_out(df, 'assessment_end_time', 'consultation_end_time')
    # df['patient_number_in'] = [i-j for i,j in zip(df.patient_number.to_list(),patient_out)]
    # number_patient_enter_consultation_since_assessment_end = number_out(df, 'assessment_end_time', 'consultation_start_time')
    # df['patient_number_waiting_consulting'] = [i-j for i,j in zip(df.patient_number.to_list(),number_patient_enter_consultation_since_assessment_end)]
    # df['doctor_free'] = df.apply(lambda x: (6 - (x.patient_number_in - x.patient_number_waiting_consulting)), axis = 1)
    df['target'] = df.apply(lambda x: (x.consultation_end_time - x.assessment_end_time), axis = 1)
    df.drop(columns = ['assessment_1', 'assessment_2', 'assessment_3','assessment_4', 'assessment_5', 'day_2', 'day_3', 'day_4', 'day_5', 'event_1', 'event_2', 'event_3', 'event_4', 'event_5'], inplace=True)
    return df

def train_test_for_model(df, last_ndays):
    train = df[df.day<=df.day.nunique()-last_ndays]
    test  = df[df.day>df.day.nunique()-last_ndays]
    train.to_csv('train/train.csv',index=False)
    test.to_csv('test/test.csv',index=False)
    return train, test

def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean(np.power(np.log1p(y_true + 1) - np.log1p(y_pred + 1), 2)))

def train_lgbm(train, train_labels):
    train_data = lgb.Dataset(train, label = train_labels)
    # Selecionando os Hyperparameters
    params = {'boosting_type': 'gbdt',
              'max_depth' : -1,
              'objective': 'regression',
              'nthread': 5,
              'num_leaves': 64,
              'learning_rate': 0.07,
              'metric' : 'rmsle'
            }
    # criando os parametros para busca
    gridParams = {'max_depth' : [-1,6],
                  'learning_rate': [0.09,0.1],
                  'n_estimators': [100,1000],
                  'num_leaves': [64,100],
                  'boosting_type' : ['gbdt'],
                  'objective' : ['regression'],
                  'random_state' : [0], 
                  'colsample_bytree' : [0.63],
                  'subsample' : [0.7]
                }
    # Criando o classificador
    mdl = lgb.LGBMRegressor(boosting_type= params['boosting_type'],
                            objective = params['objective'],
                            n_jobs = -2,
                            max_depth = params['max_depth']
                            )
    # View the default model params:
    mdl.get_params().keys()
    # Create the grid
    grid = GridSearchCV(mdl, gridParams, verbose=0, cv=3, n_jobs=-2)
    # Run the grid
    grid.fit(train, train_labels)
    # Print the best parameters found
    print('best params:',grid.best_params_)
    print('best score:',grid.best_score_)
    # Using parameters already set above, replace in the best from the grid search
    params['colsample_bytree'] = grid.best_params_['colsample_bytree']
    params['learning_rate'] = grid.best_params_['learning_rate']
    params['num_leaves'] = grid.best_params_['num_leaves']
    params['subsample'] = grid.best_params_['subsample']
    params['n_estimators'] = grid.best_params_['n_estimators']
    print('Fitting with params: ')
    print(params)
    #Train model on selected parameters and number of iterations
    lgbm = lgb.train(params,
                     train_data,
                     verbose_eval= 2
                    )
    return lgbm, lgb

def predict_lgbm(lgbm, test):
    predictions = lgbm.predict(test)
    return predictions

def save_model(model):
    joblib.dump(model, 'model/lgbm.pkl')

def load_model(model_path):
    lgbm_pickle = joblib.load(model_path)

def get_rms(values):
    """
    Returns the root-mean-square of a numpy array
    >>> get_rms(np.array([0.0]))
    0.0
    >>> get_rms(np.array([3.0, -4.0, -2.0, 14.0]))
    7.5
    """
    return np.sqrt(np.mean(values**2))

def get_files_in_folder(folder_name):
    for file_name in listdir(folder_name):
        file_path = path.join(folder_name, file_name)
        if path.isfile(file_path):
            yield file_path

def set_logger(debug_level):
    if debug_level:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

def get_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('folder_name', type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
