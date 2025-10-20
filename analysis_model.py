#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[69]:


import warnings
warnings.filterwarnings('ignore')


# In[70]:


pd.set_option("display.max_columns", None)


# In[71]:


sns.set_style("whitegrid")


# In[72]:


df = pd.read_csv("german_credit_data.csv")


# In[73]:


df.head()


# In[74]:


df["Risk"].value_counts()


# In[75]:


df.shape


# In[76]:


df.info()


# In[77]:


df.describe(include="all").T


# In[78]:


df["Job"].unique()


# In[79]:


df.isna().sum()


# In[80]:


df.duplicated().sum()


# In[81]:


df = df.dropna().reset_index(drop=True)


# In[82]:


df


# In[83]:


df.columns


# In[84]:


df.drop(columns = 'Unnamed: 0', inplace = True)


# In[85]:


df.columns


# In[86]:


df[["Age","Credit amount", "Duration"]].hist(bins = 20, edgecolor = "black")
plt.suptitle("Distribution of Numerical Features", fontsize = 14)
plt.show()


# In[87]:


plt.figure(figsize=(10,5))

for idx, col in enumerate(["Age","Credit amount","Duration"]):
    plt.subplot(1, 3, idx + 1)
    sns.boxplot(y=df[col], color="skyblue")
    plt.title(col)

plt.tight_layout()
plt.show()


# In[88]:


df.query("Duration >= 60")


# In[89]:


categorical_cols = ["Sex", "Job", "Housing", "Saving accounts", "Checking account", "Purpose"]


# In[90]:


plt.figure(figsize=[10,10])
for i, col in enumerate(categorical_cols):
    plt.subplot(3,3,i+1)
    sns.countplot(data= df, x = col, palette = "Set2", hue=col, legend=False, order = df[col].value_counts().index)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show    


# In[91]:


corr = df[["Age","Job","Credit amount","Duration"]].corr()


# In[92]:


corr


# In[93]:


sns.heatmap(corr, annot= True, cmap = "coolwarm", fmt=".2f")
plt.show


# In[94]:


df.groupby("Job")["Credit amount"].mean()


# In[95]:


df.groupby("Sex")["Credit amount"].mean()


# In[96]:


pd.pivot_table(df, values= "Credit amount", index = "Housing", columns = "Purpose")


# In[97]:


sns.scatterplot(data = df, x = "Age", y = "Credit amount", hue = "Sex", size = "Duration", alpha = 0.7, palette = "Set1")
plt.title("Credit amount vs Age coloured by Sex and sized by Duration")
plt.show


# In[98]:


sns.violinplot(data = df, x = "Saving accounts", y = "Credit amount", palette=sns.color_palette("pastel"))
plt.title("Credit Amount Distribution by Saving Accounts")
plt.show


# In[99]:


df["Risk"].value_counts(normalize = True) * 100


# In[100]:


plt.figure(figsize= (15,4))
for i, col in enumerate(["Age","Credit amount", "Duration"]):
    plt.subplot(1, 3, i+1)
    sns.boxplot(data = df, x = "Risk", y = col, palette = "Pastel2")
    plt.title(f"{col} by Risk")

plt.tight_layout()
plt.show    


# In[101]:


df.groupby("Risk")[["Age", "Credit amount", "Duration"]].mean()


# In[102]:


categorical_cols


# In[103]:


plt.figure(figsize= (15,10))
for i, col in enumerate(categorical_cols):
    plt.subplot(3, 3, i + 1)
    sns.countplot(data = df, x = col, hue = "Risk", palette = "Set1", order = df[col].value_counts().index)
    plt.title(f"{col} by Risk")
    plt.xticks(rotation = 45)

plt.tight_layout()
plt.show


# In[104]:


df.columns


# In[105]:


features = ["Age", "Sex", "Job", "Housing", "Saving accounts", "Checking account", "Credit amount", "Duration"]


# In[106]:


target = "Risk"


# In[107]:


df_model = df[features + [target]].copy()


# In[108]:


df_model.head()


# In[109]:


from sklearn.preprocessing import LabelEncoder
import joblib


# In[110]:


cat_cols = df_model.select_dtypes(include = "object").columns.drop("Risk")


# In[111]:


le_dict = {}


# In[112]:


cat_cols


# In[113]:


for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    le_dict[col] = le
    joblib.dump(le, f"{col}_encoder.pkl")


# In[114]:


le_target = LabelEncoder()


# In[115]:


target


# In[116]:


df_model[target] = le_target.fit_transform(df_model[target])


# In[117]:


df_model[target].value_counts()


# In[118]:


joblib.dump(le_target, "target_encorder.pkl")


# In[119]:


df_model.head()


# In[120]:


from sklearn.model_selection import train_test_split


# In[121]:


X = df_model.drop(target, axis = 1)


# In[122]:


y = df_model[target]


# In[123]:


X


# In[124]:


y


# In[125]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 1)


# In[126]:


X_train.shape


# In[127]:


X_test.shape


# In[128]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# In[129]:


def train_model(model, param_grid, X_train, y_trai, X_test, y_test):
    grid = GridSearchCV(model, param_grid, cv = 5, scoring = "accuracy", n_jobs = -1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return best_model, acc, grid.best_params_


# In[132]:


dt = DecisionTreeClassifier(random_state= 1, class_weight= "balanced")
dt_param_grid = {
    "max_depth" : [3,5,7,10,None],
    "min_samples_split" : [2,5,10],
    "min_samples_leaf" : [1,2,4]
}


# In[133]:


best_dt, acc_dt, params_dt = train_model(dt, dt_param_grid, X_train, y_train, X_test, y_test)


# In[134]:


print("Decision Tree Accuracy:", acc_dt)


# In[135]:


print("Best parameters", params_dt)


# In[136]:


rf = RandomForestClassifier(random_state=1, class_weight="balanced", n_jobs=-1)


# In[137]:


rf_param_grid = {
    "n_estimators": [100,200],
    "max_depth" : [5,7,10,None],
    "min_samples_split" : [2,5,10],
    "min_samples_leaf" : [1,2,4]
}


# In[139]:


best_rf, acc_rf, params_rf = train_model(rf, rf_param_grid, X_train, y_train, X_test, y_test)


# In[140]:


print("Random Forest Accuracy:", acc_rf)


# In[141]:


print("Best params", params_rf)


# In[148]:


et = ExtraTreesClassifier(random_state=1, class_weight="balanced",n_jobs= -1)


# In[149]:


et_param_grid = {
    "n_estimators": [100,200],
    "max_depth" : [5,7,10,None],
    "min_samples_split" : [2,5,10],
    "min_samples_leaf" : [1,2,4]
}


# In[151]:


best_et, acc_et, params_et = train_model(et, et_param_grid, X_train, y_train, X_test, y_test)


# In[152]:


print("Extra Trees Accuracy:", acc_et)


# In[154]:


print("Best params:", params_et)


# In[155]:


xgb = XGBClassifier(random_state = 1, scale_pos_weight= (y_train == 0).sum() / (y_train ==1).sum(), use_label_encoder = False, eval_metric = "logloss")


# In[156]:


xgb_param_grid = {
    "n_estimators": [100,200],
    "max_depth": [3,5,7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.7, 1],
    "colsample_bytree":[0.7,1]
}


# In[157]:


best_xgb, acc_xgb, params_xgb = train_model(xgb,xgb_param_grid, X_train, y_train, X_test, y_test)


# In[158]:


print("XGB accuracy:", acc_xgb)


# In[159]:


print("Best params:",params_xgb)


# In[160]:


best_et.predict(X_test)


# In[161]:


joblib.dump(best_et, "extra_trees_credit_model.pkl")


# In[ ]:




