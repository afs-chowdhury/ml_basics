# data ready 

#%%
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# modeling

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split , GridSearchCV


# setup random seed 

import numpy as np
np.random.seed(22)

# load data 

data = pd.read_csv("car-sales-extended-missing-data.csv")

# delete data with no labels

data.dropna(subset=["Price"] , inplace = True)


# setup features and make pipeline to transform them 

cat_features = ["Make", "Colour"]
cat_transformer = Pipeline(steps=[
    ("imputer" , SimpleImputer(strategy = "constant" , fill_value = "Missing")),
    ("one_hot" , OneHotEncoder(handle_unknown = "ignore"))
])


door_feature = ["Doors"]
door_transformer = Pipeline(steps=[
    ("imputer" , SimpleImputer(strategy = "constant", fill_value = 4))
])


num_feature = ["Odometer (KM)"]
num_transformer = Pipeline(steps=[
    ("imputer" , SimpleImputer(strategy = "mean"))
])


# setup a preprocessor 

preprocessor = ColumnTransformer(transformers= [
    ("cat", cat_transformer , cat_features),
    ("door" , door_transformer, door_feature),
    ("num" , num_transformer, num_feature)
])


# set up a pipeline for preprocessor and model 

model = Pipeline(steps =[
    
    ("preprocessor" , preprocessor),
    ("model" , RandomForestRegressor())
    
])



#split the data 

X = data.drop("Price" , axis = 1)
y = data["Price"]

X_train, X_test , y_train , y_test = train_test_split(X,y, test_size= 0.2)

# fit the model 

model.fit(X_train, y_train)  # according to the pipeline documentatoin , the final estimator only needs to implement fit 

# evaluate the model 

general_score = model.score(X_test, y_test)


# getup pipegrid for GridSearchCV

pipe_grid = {
    "preprocessor__num__imputer__strategy": ["mean", "median"], 
    "model__n_estimators" : [100, 1000],
    "model__max_depth":[None,5],
    "model__max_features" :["auto"],
    "model__min_samples_split" : [2,4]
}


gs_model = GridSearchCV(model, pipe_grid, cv = 5 , verbose =2)
gs_model.fit(X_train, y_train)

gs_score = gs_model.score(X_test, y_test)

print(f"The general score is   {general_score} and gs score is {gs_score}")