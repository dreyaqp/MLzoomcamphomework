import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor




df= pd.read_csv(r"C:\Users\HP\Downloads\titanic.csv")

df.columns = df.columns.str.lower().str.replace(' ', '_')
df[df.select_dtypes(include='number').columns] = df.select_dtypes(include='number').fillna(0.0)
df[df.select_dtypes(include='object').columns] = df.select_dtypes(include='object').fillna("N/A")
df = df.drop('ticket', axis=1)
from sklearn.preprocessing import LabelEncoder

for col in ['sex']:
    df[col] = LabelEncoder().fit_transform(df[col])
    le = LabelEncoder()
df['cabin_encoded'] = le.fit_transform(df['cabin'])
df['embarked_encoded'] = LabelEncoder().fit_transform(df['embarked'])
df = df.drop('cabin', axis=1)
df = df.drop('embarked', axis=1)



df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)
df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 42)
df_train= df_train.reset_index(drop=True)
df_test= df_test.reset_index(drop=True)
df_val= df_val.reset_index(drop=True)

y_train = df_train['survived']
y_test = df_test['survived']
y_val = df_val['survived']
X_train = df_train.drop(['survived','name','vpassengerid'], axis=1)
X_test = df_test.drop(['survived','name','vpassengerid'], axis=1)
X_val = df_val.drop(['survived','name','vpassengerid'], axis=1)


## Linear Regression
baseline_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])

baseline_model.fit(X_train, y_train)
baseline_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])

baseline_model.fit(X_train, y_train)
y_pred = baseline_model.predict(X_val)


## Decision Tree Regressor
dt_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('dt', DecisionTreeRegressor(random_state=42))
])
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_val)


## Random Regressor
rf_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ))
])
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_val)


### Gradient Boosting code
gb_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('gb', GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ))
])
gb_model.fit(X_train, y_train)
y_pred = gb_model.predict(X_val)

### KNN Regressor
knn_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor(n_neighbors=5, weights='distance'))
])
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_val)