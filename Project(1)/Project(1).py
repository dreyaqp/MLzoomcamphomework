#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor


# In[3]:


df= pd.read_csv(r"C:\Users\HP\Downloads\titanic.csv")


# In[4]:


df.head().T


# In[5]:


df.dtypes


# In[6]:


df.columns


# In[7]:


df.columns = df.columns.str.lower().str.replace(' ', '_')


# In[8]:


df.columns


# In[9]:


df.isnull().sum()


# In[10]:


df[df.select_dtypes(include='number').columns] = df.select_dtypes(include='number').fillna(0.0)
df[df.select_dtypes(include='object').columns] = df.select_dtypes(include='object').fillna("N/A")


# In[11]:


df.isnull().sum()


# In[12]:


df.head(20).T


# In[13]:


df = df.drop('ticket', axis=1)


# In[14]:


from sklearn.preprocessing import LabelEncoder

for col in ['sex']:
    df[col] = LabelEncoder().fit_transform(df[col])


# In[ ]:





# In[15]:


le = LabelEncoder()
df['cabin_encoded'] = le.fit_transform(df['cabin'])
df['embarked_encoded'] = LabelEncoder().fit_transform(df['embarked'])


# In[16]:


df = df.drop('cabin', axis=1)
df = df.drop('embarked', axis=1)


# In[17]:


df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)
df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 42)


# In[18]:


categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
numerical_columns = list(df.dtypes[df.dtypes == 'int64'].index)


# In[19]:


len(df_test),len(df_train),len(df_val)


# In[20]:


df.dtypes


# In[21]:


df_train= df_train.reset_index(drop=True)
df_test= df_test.reset_index(drop=True)
df_val= df_val.reset_index(drop=True)


# ## Scaling

# In[22]:


from sklearn.preprocessing import StandardScaler


# In[23]:


scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


# In[24]:


from sklearn.preprocessing import MinMaxScaler


# In[25]:


scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


# ## Splitting into x and y

# In[26]:


y_train = df_train['survived']
y_test = df_test['survived']
y_val = df_val['survived']


# In[27]:


X_train = df_train.drop(['survived','name','vpassengerid'], axis=1)
X_test = df_test.drop(['survived','name','vpassengerid'], axis=1)
X_val = df_val.drop(['survived','name','vpassengerid'], axis=1)


# ## Statistics summary

# In[28]:


df.describe()


# In[29]:


df.info()


# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[31]:


for col in numerical_columns:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()


# In[32]:


for col in numerical_columns:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


# In[33]:


target = 'survived'  
for col in numerical_columns:
    if col != target:
        plt.figure()
        sns.scatterplot(x=df[col], y=df[target])
        plt.title(f'{col} vs {target}')
        plt.show()


# In[34]:


X_train_dicts = X_train.to_dict(orient='records')
X_val_dicts = X_val.to_dict(orient='records')
X_test_dicts = X_test.to_dict(orient='records')


# In[35]:


model = RandomForestRegressor()
model.fit(X_train, y_train)
importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
importances.plot(kind='pie')
plt.title("Feature Importance")
plt.show()


# In[36]:


print(X_train.dtypes)


# ## Linear Regression

# In[37]:


X_train.columns.tolist()


# In[38]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression


# In[39]:


baseline_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])

baseline_model.fit(X_train, y_train)


# In[40]:


y_pred = baseline_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5                         
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)


# In[41]:


rmse, mae, r2


# ## Decision Tree Regressor

# In[42]:


from sklearn.tree import DecisionTreeRegressor


# In[43]:


dt_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('dt', DecisionTreeRegressor(random_state=42))
])


# In[44]:


dt_model.fit(X_train, y_train)


# In[45]:


y_pred = dt_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)


# In[46]:


rmse, mae, r2


# ## Random Regressor

# In[47]:


from sklearn.ensemble import RandomForestRegressor


# In[48]:


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


# In[49]:


rf_model.fit(X_train, y_train)


# In[50]:


y_pred = rf_model.predict(X_val)


# In[51]:


mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)


# In[52]:


rmse, mae, r2


# ### Gradient Boosting code

# In[53]:


from sklearn.ensemble import GradientBoostingRegressor


# In[54]:


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


# In[55]:


gb_model.fit(X_train, y_train)


# In[56]:


y_pred = gb_model.predict(X_val)


# In[57]:


mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)


# In[58]:


rmse, mae, r2


# ### KNN Regressor

# In[59]:


from sklearn.neighbors import KNeighborsRegressor


# In[60]:


knn_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor(n_neighbors=5, weights='distance'))
])


# In[61]:


knn_model.fit(X_train, y_train)


# In[62]:


y_pred = knn_model.predict(X_val)


# In[63]:


mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)


# In[64]:


rmse, mae, r2


# ### Gradient Boosting has the best RMSE and R^2

# In[65]:


gb = gb_model.named_steps['gb']


# In[66]:


feature_names = X_train.columns.tolist()


# In[67]:


importances = gb.feature_importances_
feat_imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

feat_imp_df


# In[67]:


importances = gb.feature_importances_
feat_imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

feat_imp_df


# In[68]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.barh(feat_imp_df['feature'], feat_imp_df['importance'])
plt.gca().invert_yaxis()  
plt.xlabel("Importance")
plt.title("Feature Importances - Gradient Boosting")
plt.show()


# In[ ]:


import joblib
joblib.dump(gb_model, "titanic_model.pkl")
print("Model saved as titanic_model.pkl")

# In[ ]:





