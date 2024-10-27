import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/drive/MyDrive/Movie_classification.csv')
df.head()
df.shape
df.columns
df = df.drop(['Start_Tech_Oscar'], axis=1)
df
df.isnull().sum()
df1 = df.fillna(method = 'bfill')
df1.isnull().sum()
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

encoder.fit(df['3D_available'])
df['3D_available'] = encoder.transform(df['3D_available'])
encoder = LabelEncoder()

encoder.fit(df['Genre'])
df['Genre'] = encoder.transform(df['Genre'])
sns.boxplot(x='Marketing expense', data = df1)
sns.boxplot(x='Production expense', data = df1)
sns.boxplot(x='Multiplex coverage', data = df1)
sns.boxplot(x='Budget', data = df1)
sns.boxplot(x='Movie_length', data = df1)
sns.boxplot(x='Lead_ Actor_Rating', data = df1)
sns.boxplot(x='Lead_Actress_rating', data = df1)
sns.boxplot(x='Director_rating', data = df1)
sns.boxplot(x='Producer_rating', data = df1)
sns.boxplot(x='Critic_rating', data = df1)
sns.boxplot(x='Trailer_views', data = df1)
sns.boxplot(x='Time_taken', data = df1)
sns.boxplot(x='Twitter_hastags', data = df1)
sns.boxplot(x='Avg_age_actors', data = df1)
sns.boxplot(x='Num_multiplex', data = df1)
q1,q3 = np.percentile(df1['Marketing expense'],[25,75])
q1,q3
iqr = q3 - q1
iqr
lb = q1 - (1.5)*iqr
ub = q3 + (1.5)*iqr
lb, ub
no_outliers = []

for i in df1['Marketing expense']:
  if i<=lb or i>=ub:
    pass
  else:
    no_outliers.append(i)

sns.boxplot(no_outliers)
q1,q3 = np.percentile(df1['Time_taken'],[25,75])
q1,q3
iqr = q3 - q1
iqr
lb = q1 - (1.5)*iqr
ub = q3 + (1.5)*iqr
lb, ub
no_outliers = []
for i in df1['Time_taken']:
  if i<=lb or i>=ub:
    pass
  else:
    no_outliers.append(i)
sns.boxplot(no_outliers)
q1,q3 = np.percentile(df1['Trailer_views'],[25,75])
q1,q3
iqr = q3 - q1
iqr
lb = q1 - (1.5)*iqr
ub = q3 + (1.5)*iqr
lb, ub
no_outliers = []

for i in df1['Trailer_views']:
  if i<=lb or i>=ub:
    pass
  else:
    no_outliers.append(i)
sns.boxplot(no_outliers)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
X = df1.drop(['Collection'], axis=1)
y = df1['Collection']
encoder = LabelEncoder()

encoder.fit(df1['3D_available'])
df1['3D_available'] = encoder.transform(df1['3D_available'])
encoder = LabelEncoder()

encoder.fit(df1['Genre'])
df1['Genre'] = encoder.transform(df1['Genre'])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred_lr = lr.predict(X_test)
r2_score(y_pred_lr,y_test)
dtr= DecisionTreeRegressor()
dtr.fit(X_train,y_train)
y_pred_dtr = dtr.predict(X_test)
r2_score(y_pred_dtr,y_test)
rfr = RandomForestRegressor()
rfr.fit(X_train,y_train)
y_pred_rfr = rfr.predict(X_test)
r2_score(y_pred_rfr,y_test)
