import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def pred_dataset(file_names:list, feature_set:list,skiprows:int=1,source_path:str='./CO2_adsorption/new_data',group_index:str='Index',**kwargs):
    '''
    This function is used to generate the dataset for building machine learning models, and only training and test dataset
    will generated from this function using group splitting.
    file_names: list, the list of file names, e.g., ['CO2-02-02-2022.xlsx','CH4-02-02-2022.xlsx']
    feature_set: list, the list of features, e.g., ['Pressure','Temperature','CO2 uptake']
    skiprows: int, the number of rows to skip, default is 1
    source_path: str, the path of the source data, default is './CO2_adsorption/new_data'
    group_index: str, the name of the column used for group splitting, default is 'Index'
    kwargs: dict, the key word arguments, default is {'split_ratio':0.2}
    '''
    split_ratio = kwargs.get('split_ratio',0.2)
    random_state = kwargs.get('random_state',42)
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for file_name in file_names:
        temp_data = pd.read_excel(os.path.join(source_path,file_name),skiprows = skiprows)
        temp_data = temp_data.dropna(axis=0,how = 'any',subset = feature_set)
        #temp_data = temp_data[temp_data['Pressure']>0.01]
        index = list(set(temp_data[group_index].values))
        _,test_index = train_test_split(index,test_size = split_ratio,random_state = random_state)
        train_x = temp_data.loc[~temp_data[group_index].isin( test_index)]
        test_x = temp_data.loc[temp_data[group_index].isin(test_index)]
        train_df = pd.concat([train_df,train_x],axis=0)
        test_df = pd.concat([test_df,test_x],axis =0)
    return train_df,test_df