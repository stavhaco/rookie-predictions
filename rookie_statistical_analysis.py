import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import RFE
import seaborn as sns
sns.set()
from sklearn.ensemble import RandomForestRegressor


def prepare_data(data):
    data["star"]=np.where(data["BEST EFF"]<=100,1,0)
    #dropping corrolated features and unnecceserry ones
    #dropping rank features
    features_drop_ranks = ['GP_RANK', 'W_RANK', 'L_RANK', 'W_PCT_RANK', 'MIN_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK', 'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK', 'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK', 'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK', 'PFD_RANK', 'PTS_RANK', 'PLUS_MINUS_RANK', 'NBA_FANTASY_PTS_RANK', 'DD2_RANK', 'TD3_RANK']
    #dropping corolated features according to heat map
    features_drop_correlated = ["NBA_FANTASY_PTS","FGM","FGA","FG3M","FG3A","FTM","FTA","REB","W","L","PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "W_PCT","BEST EFF","CFPARAMS","CFID","PLUS_MINUS"] #instead of made and attampted we have % of FG,3s,FT,instead of tot REB looking at DREB and OREB
    data = data.drop(features_drop_ranks,axis=1)
    data = data.drop(features_drop_correlated,axis=1)
    return data

def check_corr(data_train):
    plt.figure(figsize=(15, 6))
    sns.heatmap(data_train.corr())
    plt.show()

def preform_regression(data_train,data_test):
    X_train = data_train.drop(["star"],axis=1)
    #cols = ['TEAM_ID', 'AGE', 'GP', 'MIN', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'OREB', 'DREB', 'AST', 'TOV', 'STL', 'BLK',
            #'BLKA', 'PF', 'PFD', 'PTS', 'DD2', 'TD3']
    cols = ['TEAM_ID','AGE','GP','FG_PCT','AST','OREB','PTS']
    X_train = data_train[cols]
    y_train = data_train["star"]
    X_test = data_test[cols]
    y_test = data_test["star"]
    model = sm.Logit(y_train, X_train)
    result = model.fit()
    print result.summary2()
    y_pred = result.predict(X_test)
    y_pred_binary = y_pred
    y_pred_binary[y_pred_binary >= 0.5] = 1
    y_pred_binary[y_pred_binary < 0.5] = 0
    conf_matrix = confusion_matrix(y_pred_binary,y_test)
    print conf_matrix
    true_neg,false_pos,false_neg,true_pos = conf_matrix[1][1],conf_matrix[1][0],conf_matrix[0][1],conf_matrix[0][0]
    print "prediction of not-great players "+str(float(true_neg)/(true_neg+false_neg))
    print "prediction of great players "+str(round(float(true_pos)/(true_pos+false_pos),2))

def preform_rand_forest(data_train,data_test):
    X_train = data_train.drop(["star"],axis=1)
    y_train = data_train["star"]
    X_test = data_test.drop(["star"],axis=1)
    y_test = data_test["star"]
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    rf.fit(X_train, y_train)
    #print rf.feature_importances_
    predictions = rf.predict(X_test)
    y_pred_binary =predictions
    y_pred_binary[y_pred_binary >= 0.5] = 1
    y_pred_binary[y_pred_binary < 0.5] = 0
    conf_matrix = confusion_matrix(y_pred_binary,y_test)
    print conf_matrix
    true_neg,false_pos,false_neg,true_pos = conf_matrix[1][1],conf_matrix[1][0],conf_matrix[0][1],conf_matrix[0][0]
    print "prediction of not-great players "+str(float(true_neg)/(true_neg+false_neg))
    print "prediction of great players "+str(round(float(true_pos)/(true_pos+false_pos),2))
#print rookie_data_train.groupby('TEAM_ID').star.value_counts()
#print rookie_data_train[['TEAM_ID', 'star']].groupby(['TEAM_ID'], as_index=False).mean()
#print rookie_data_train[['PTS', 'star']].groupby(['PTS'], as_index=False).mean()
#print rookie_data_train[['AGE', 'star']].groupby(['AGE'], as_index=False).mean()
#print rookie_data_train[['DD2', 'star']].groupby(['DD2'], as_index=False).mean()

rookie_data_train_features = pd.DataFrame.from_csv('C:/Users/stav/Desktop/study/nba/rookies_train_EFF_rank.csv')
rookie_data_test_features = pd.DataFrame.from_csv('C:/Users/stav/Desktop/study/nba/rookies_test_EFF_rank.csv')
rookie_data_train = prepare_data(rookie_data_train_features)
rookie_data_test = prepare_data(rookie_data_test_features)
preform_regression(rookie_data_train,rookie_data_test)
preform_rand_forest(rookie_data_train,rookie_data_test)











