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
from evaluate import get_arguments, set_logger, get_files_in_folder

logger = logging.getLogger()
features = ['assessment_concluded','temperature','priority','pain']
label = 'target'

def main():
    args = get_arguments()
    set_logger(args.debug)
    df = pd.concat([pd.read_csv(file_path,index_col=0) for  file_path in get_files_in_folder(args.folder_name)])
    df = summary(df)
    train, test = split_train_test(df, 5)
    lgbm, lgb = train_lgbm(train[features], train[label])
    save_model(lgbm)
  
def summary(df):
    '''
    summarize data for training

    Args:
    df (dataframe): data to summarize

    Return:
    df (dataframe): data suummarized
    '''
    df['event_number'] = (df.groupby(['day','patient']).cumcount() + 1).astype(str)
    df = df.set_index(['patient','event_number']).unstack()
    df.sort_index(axis = 1,level = 1, inplace=True)
    df.columns = ['_'.join(col) for col in df.columns]
    df.reset_index(inplace=True)
    df.rename(columns={"time_1": "arrived", 
                       "time_2": "assessment_initiated",
                       "time_3": "assessment_concluded",
                       "time_4": "consultation_initiated",
                       "time_5": "consultation_finished",
                       "day_1": "day"}, inplace = True)  
    df['temperature'] = df["assessment_3"].str.split("|", n = 2, expand = True)[1].astype('float64').round(1)
    df['priority'] = df["assessment_3"].str.split("|", n = 2, expand = True)[0]
    df['pain'] = df["assessment_3"].str.split("|", n = 2, expand = True)[2]
    df.priority = df.priority.astype('category').cat.codes
    df.pain = df.pain.astype('category').cat.codes
    df['target'] = df.apply(lambda x: (x.consultation_finished - x.assessment_concluded), axis = 1)
    df.drop(columns = ['assessment_1', 'assessment_2', 'assessment_3','assessment_4', 'assessment_5', 'day_2', 'day_3', 'day_4', 'day_5', 'event_1', 'event_2', 'event_3', 'event_4', 'event_5'], inplace=True)
    return df

def split_train_test(df, last_ndays):
    '''
    split data in train and test

    Args:
    df (dataframe): data to split
    last_ndays (int): number of last days to split

    Return:
    train (dataframe): train data
    test (dataframe): test data
    '''
    train = df[df.day<=df.day.nunique()-last_ndays]
    test  = df[df.day>df.day.nunique()-last_ndays]
    train.to_csv('train/train.csv',index=False)
    test.to_csv('test/test.csv',index=False)
    return train, test

def rmsle(y_true, y_pred):
    '''
    rmsle metric

    Args:
    y_true(): true target
    y_pred(): predicted target

    Return:
    rmsle (float): rmsle metric
    '''
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean(np.power(np.log1p(y_true + 1) - np.log1p(y_pred + 1), 2)))

def train_lgbm(train, train_labels):
    '''
    train lightgbm model

    Args:
    train(): train data
    train_label(): train target

    Return:
    lgbm(): model
    lgb(): lightgbm method
    '''
    train_data = lgb.Dataset(train, label = train_labels)
    # Selecting hyperparameters
    params = {'boosting_type': 'gbdt',
              'max_depth' : -1,
              'objective': 'regression',
              'nthread': 5,
              'num_leaves': 64,
              'learning_rate': 0.07,
              'metric' : 'rmsle'
            }
    # Creating search parameters
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
    # Creating the classifier
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
    # Using parameters already set above, replace in the best from the grid search
    params['colsample_bytree'] = grid.best_params_['colsample_bytree']
    params['learning_rate'] = grid.best_params_['learning_rate']
    params['num_leaves'] = grid.best_params_['num_leaves']
    params['subsample'] = grid.best_params_['subsample']
    params['n_estimators'] = grid.best_params_['n_estimators']
    # Train model on selected parameters and number of iterations
    lgbm = lgb.train(params,
                     train_data,
                     verbose_eval= 0
                    )
    print('Done')
    return lgbm, lgb

def predict_lgbm(model, test):
    predictions = model.predict(test)
    return predictions

def save_model(model):
    joblib.dump(model, 'model/lgbm.pkl')

def load_model(model_path):
    lgbm_pickle = joblib.load(model_path)

if __name__ == '__main__':
    main()
