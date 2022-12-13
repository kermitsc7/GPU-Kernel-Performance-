import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
import numpy as np 
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import joblib


train_df = pd.read_csv('sgemm_product.csv', header=0, delimiter=',')

train_df['Runtime'] = train_df[['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)']].mean(axis=1)
train_df = train_df.drop(['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)'], axis = 1)


def detect_outliers_zscore(data):
    outliers = []
    thres = 3
    index = 0
    mean = np.mean(data)
    std = np.std(data)
    # print(mean, std)
    for i in data:
        index = index + 1
        z_score = (i-mean)/std
        if (np.abs(z_score) > thres):
            outliers.append(index)
    return outliers# Driver code

sample_outliers_Runtime = detect_outliers_zscore(train_df['Runtime'])
train_df.drop(sample_outliers_Runtime , inplace = True)
train_df['Target']=np.log(train_df.Runtime)
train_df.drop(columns = "Runtime", inplace = True)

df = train_df.copy()
one_hot_encoded_data = pd.get_dummies(df, columns = ['SA', 'SB','STRM', 'STRN'])
df = one_hot_encoded_data.copy()

prework = df['NWG']*df['MWG']
df['prework'] = prework
df["MWI"] = df["MWG"]*df["MDIMC"]
df['NWI'] = df["NWG"]*df["NDIMC"]

df = df.drop(columns = ['SA_1', 'SB_1', 'STRM_1', 'STRN_1'])
df.head()
X = df.drop(columns=['Target'])
y = df['Target'].values
Min_Max = MinMaxScaler()
y = y.reshape(-1,1)
X = Min_Max.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.30)

regr = RandomForestRegressor(criterion = 'squared_error', max_features= 1.0, n_estimators =100)
regr.fit(X_train, np.ravel(Y_train,order='C'))


joblib.dump(regr, "regr.pkl")
