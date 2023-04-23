from sklearn.model_selection import GridSearchCV,cross_validate,GroupKFold
from  sklearn.metrics import mean_squared_error,r2_score
from sklearn.utils import shuffle
import numpy as np

def model_CV(train_x,train_y,groups:list,model,para_grid,**kwargs):
    '''
    This function is used to perform cross validation for the machine learning models.
    train_x: dataframe, the training features
    train_y: dataframe, the training labels
    groups: array, the group index for group splitting
    model: the machine learning model
    para_grid: dict, the parameter grid for grid search
    kwargs: dict, the key word arguments, default is {'n_splits':5}
    '''
    n_splits = kwargs.get('n_splits',5)
    out_cv = GroupKFold(n_splits = n_splits)
    result = GridSearchCV(model,para_grid,cv= out_cv.get_n_splits(groups =groups),
    scoring='neg_mean_squared_error', return_train_score=True,n_jobs=-1)
    result.fit(train_x,train_y)
    
    model_refit =model.set_params(**result.best_params_)
    train_cv = cross_validate(model_refit,train_x,train_y,groups = groups,cv =out_cv,scoring = ('r2', 'neg_mean_squared_error'))
    train_mse_cv = -np.round(train_cv['test_neg_mean_squared_error'].mean(),4)
    train_r2_cv = np.round(train_cv['test_r2'].mean(),4)
    
    return [train_r2_cv,train_mse_cv],result.best_params_

# model evaluation
def model_eval(model,test_x,test_y):
      
    test_pre = model.predict(test_x)
    test_r2 = r2_score(test_pre,test_y)
    test_mse = mean_squared_error(test_y,test_pre)
    return np.round(test_r2,4),np.round(test_mse,4)

# comparing different models
def model_comparison(train_df,test_df,model_list,para_grids,feature_list:list,gas_list:list,output:str='Adsorp(mmol/g)',**kwargs):
    groups_label =kwargs.get('groups','Index')
    gas_label = kwargs.get('gas_label','Label')
    input_feature = feature_list
    result_total = []
    for gas in gas_list:
        if gas =='total':
            train_df_com = train_df
            test_df_com = test_df
            train_x = train_df_com[input_feature]
            test_x = test_df_com[input_feature]
            train_y = train_df_com[output].values
            test_y = test_df_com[output].values
            groups = train_df_com[gas_label].values

            train_x, train_y, groups = shuffle(train_x, train_y, groups, random_state=42)
            for model_name, model in model_list:
                result, best_param = model_CV(train_x,train_y.squeeze(),groups,model,para_grids[model_name])
                model_refit = model.set_params(**best_param)
                model_refit.fit(train_x,train_y.squeeze())
                test_r2_total,test_mse_total = model_eval(model_refit,test_x,test_y.squeeze()) 
                for gases in gas_list[1:]:
                    test_df_com = test_df[test_df[gas_label]==gases]
                    test_xs = test_df_com[input_feature]
                    test_ys = test_df_com[output].values
                    test_r2,test_mse = model_eval(model_refit,test_xs,test_ys.squeeze()) 
                    result_total.append([gases,model_name+'_total',result[0],result[1],test_r2_total,test_mse_total,test_r2,test_mse,best_param])
                    print('Dataset {}, Algorithm {}, Test_r2 {}, Test_error {}'.format(gas,model_name+'_total',test_r2,test_mse))    
        else:
            train_df_com = train_df[train_df[gas_label]==gas]
            test_df_com = test_df[test_df[gas_label]==gas]
            train_x = train_df_com[input_feature]
            test_x = test_df_com[input_feature]
            train_y = train_df_com[output].values
            test_y = test_df_com[output].values
            groups = train_df_com[groups_label]
            train_x, train_y, groups = shuffle(train_x, train_y, groups, random_state=42)
            for model_name, model in model_list:
                result, best_param = model_CV(train_x,train_y.squeeze(),groups,model,para_grids[model_name])
                model_refit = model.set_params(**best_param)
                model_refit.fit(train_x,train_y.squeeze())
                test_r2,test_mse = model_eval(model_refit,test_x,test_y.squeeze()) 
                result_total.append([gas,model_name+'_separate',result[0],result[1],-1,-1, test_r2,test_mse,best_param])
                
                print('Dataset {}, Algorithm {}, Test_r2 {}, Test_error {}'.format(gas,model_name+'__separate',test_r2,test_mse))     
    return result_total