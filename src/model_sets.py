from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,\
    BaggingRegressor,ExtraTreesRegressor,RandomForestRegressor
from lightgbm import LGBMRegressor  
from sklearn.svm import SVR
  
n_estimators = [30,50,100,120,150,180,200,250]

# define different models#,
models = [
    ('SVR',SVR(max_iter=10000)),
    ('DT',DecisionTreeRegressor(random_state=42)),\
    ('ADBR',AdaBoostRegressor(random_state=42)), 
    ("GBR",GradientBoostingRegressor(random_state=42)),\
    ('BG',BaggingRegressor(random_state=42,n_jobs=-1)),
    ('ETR',ExtraTreesRegressor(random_state=42,n_jobs=-1)),\
    ('RF',RandomForestRegressor(n_jobs=-1,random_state=42)),
    ('LGBM',LGBMRegressor(n_jobs = -1,random_state = 42)),\
    ]

# set search parameters grid for different models
para_grids = { 
    'SVR':{'kernel':['linear','poly','rbf','sigmoid','precomputed']},\
    'DT':{'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']},\
    'ADBR':{'n_estimators':n_estimators,'learning_rate':[0.1,0.5,1,2],'loss':['linear','square','exponential']},\
    'GBR':{'n_estimators':n_estimators,'learning_rate':[0.1,0.5,1,2]},\
    'BG':{'n_estimators':[10,50,100]},\
    'ETR':{'n_estimators':n_estimators},\
    'RF':{'n_estimators':n_estimators},\
    'LGBM':{'num_leaves':[10,20,30,50],'learning_rate': [0.05,0.1,0.5,1],'n_estimators':n_estimators},\
    
    
    }