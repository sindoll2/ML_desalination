import torch
from sklearn.linear_model import LinearRegression
from matplotlib import rcParams, pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from warnings import simplefilter

rcParams['figure.figsize'] = (12, 8)
plt.style.use('seaborn-ticks')
simplefilter('ignore')

df = pd.read_csv('membranedt.csv',encoding='cp949')

target_col = 'Membrane Flux \n(LMH)'
seed = 10
print(df.shape)
df.head()

df['FS Velocity (cm/s)'].fillna(df['FS Velocity (cm/s)'].mean(), inplace=True)
df['DS Velocity (cm/s)'].fillna(df['DS Velocity (cm/s)'].mean(), inplace=True)
df = df.drop(df.columns[[18,20]], axis=1)

df.info()

num_cols = [x for x in df.columns if df[x].dtype in [np.int64, np.float64]and x != target_col] 
cat_cols = ['Author', 'Membrane AL Type', 'Manufacture', 'Membrane Direction','Flow Orientation\n(counter / cocurrent)','Feed Solution (FS)','Draw Solution (DS)']
print(f'    numeric ({len(num_cols)}):\t{num_cols}')
print(f'categorical ({len(cat_cols)}):\t{cat_cols}')

df_ohe = pd.get_dummies(df, columns=cat_cols)

X = df_ohe.drop(target_col, axis=1)
X.shape

X_trn, X_tst, y_trn, y_tst = train_test_split(X, df[target_col], test_size=0.2, random_state=seed)
clf = LinearRegression()
clf.fit(X_trn, np.log1p(y_trn))
p = np.expm1(clf.predict(X_tst))

p_trn = np.expm1(clf.predict(X_trn))

if('Membrane Flux (LMH)' in df.keys()):
    print(df['Membrane Flux (LMH)'])
else:
    print('Membrane Flux (LMH) not found')

print('r2_score :', r2_score(y_tst, p))
print('r2_score :', r2_score(y_trn, p_trn))
print('MAE :', mean_absolute_error(y_tst, p))
print('MSE :', mean_squared_error(y_tst, p))
print('RMSE : ', mean_squared_error(y_tst, p, squared = False))
print('MAPE :', mean_absolute_percentage_error(y_tst, p))

rcParams['figure.figsize'] = (12, 8)
plt.scatter(p, y_tst, s=100, c='g')
plt.xlabel("Predicted permeate flux (LMH)", size=20)
plt.ylabel("Actual permeate flux (LMH)", size=25)
plt.grid(True)
plt.axis([0,50,0,50])

x = np.arange(0,50, 1)
y = x
plt.plot(x,y, 'k', linestyle='--')
plt.figure(figsize=(8,4))
plt.show()

rcParams['figure.figsize'] = (12, 8)
plt.scatter(p_trn, y_trn, s=70, c='g')
plt.xlabel("Predicted permeate flux (LMH)", size=20)
plt.ylabel("Actual permeate flux (LMH)",size=25)
plt.grid(True)
plt.axis([0,50,0,50])

x = np.arange(0, 50,1)
y = x 
plt.plot(x,y, 'k', linestyle='--')
plt.figure(figsize=(8,2))
plt.show()

