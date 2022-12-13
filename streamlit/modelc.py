import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

mean = df['Target'].mean()
df.loc[df['Target'] <= mean, 'Target'] = 0
df.loc[df['Target'] > mean, 'Target'] = 1
df['Target'] = df['Target'].astype('int')

df_target=df[['Target']].values
df_features=df.drop(columns=['Target'],axis=1).values
x1_train, x1_test, y1_train, y1_test = train_test_split(df_features, df_target, test_size = 0.3, random_state = 0)


sc = StandardScaler()
x1_train = sc.fit_transform(x1_train)
x1_test = sc.transform(x1_test)

gbc1 = GradientBoostingClassifier()
gbc1.fit(x1_train,y1_train)



joblib.dump(gbc1, "class.pkl")
