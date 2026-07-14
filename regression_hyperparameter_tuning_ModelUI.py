# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:58:06 2019

@author: sakshij
"""

from .class_RunWithTimeout import RunWithTimeout
from .paths import (getDataFileName, getDataFilePath, getImagesPath, 
                    getMetaDataFileInfoPath, getResultsPath, getSampledDataPath, 
                    getStaticFilesPath)

import pandas as pd
import numpy as np

from os import path
import json
import ast

# Splitting Data
from sklearn.model_selection  import train_test_split

# Regression Models
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
#                              HistGradientBoostingRegressor
                              )

from sklearn.linear_model import (ARDRegression, BayesianRidge, LarsCV, LinearRegression,
                                  LogisticRegression, LogisticRegressionCV, PassiveAggressiveRegressor,
                                  RidgeCV, SGDRegressor
                                  )

# Evaluation Metrics
from sklearn.metrics import (mean_squared_error, mean_absolute_error, mean_squared_log_error
                             )

from sklearn.metrics import  (auc, make_scorer, recall_score, precision_recall_fscore_support,
                              accuracy_score, precision_score)

#Running different parameters in Grid
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone

#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)

# ignore all future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

###########################################################################################
#Paths Info

#def getDataFilePath():
#    file_info_path = r'C:\Users\sakshij\Documents\UI\django_test\practice_to_project\Data\\'
#    info_df = pd.read_csv(file_info_path + "File_Info.csv")
#    return info_df['Data_File_Path'][0]
#
#def getSampledDataPath():
#    file_info_path = r'C:\Users\sakshij\Documents\UI\django_test\practice_to_project\Data\\'
#    info_df = pd.read_csv(file_info_path + "File_Info.csv")
#    return info_df['Sampled_Data_File_Path'][0]
#
#def getResultsPath():
#    file_info_path = r'C:\Users\sakshij\Documents\UI\django_test\practice_to_project\Data\\'
#    info_df = pd.read_csv(file_info_path + "File_Info.csv")
#    return info_df['Results_Path'][0]
#
#def getDataFileName():
#    file_info_path = r'C:\Users\sakshij\Documents\UI\django_test\practice_to_project\Data\\'
#    info_df = pd.read_csv(file_info_path + "File_Info.csv")
#    return info_df['Data_File_Name'][0]
    
###########################################################################################
#Helper Functions

def validateRange(model_params_dict):
    """
    Function for validating if the numeric parameters' values are in the specified.
    Values passed as grid need not be validated as their validation is being done in 
    front end itself. 
    
    ###### THIS VALIDATION WILL ALSO BE DONE IN FRONT END, OR ATLEAST VIA ANOTHER API CALL ######
    """
    
    return model_params_dict

def getModelersDF(as_dict=False):
    model_names = [
             'DecisionTreeRegressor'
            ,'ExtraTreeRegressor'
            ,'AdaBoostRegressor'
            ,'BaggingRegressor'
            ,'ExtraTreesRegressor'
            ,'GradientBoostingRegressor'
            ,'RandomForestRegressor'
#            ,'VotingRegressor'
#            ,'HistGradientBoostingRegressor'
            
#            ,'ARDRegression'
#            ,'BayesianRidge'
#            ,'LarsCV'
            ,'LinearRegression'
#            ,'LogisticRegression'
#            ,'LogisticRegressionCV'
#            ,'PassiveAggressiveRegressor'
#            ,'RidgeCV'
#            ,'SGDRegressor'
            ]
    
    model_obj = [
             DecisionTreeRegressor(random_state=2019)
            ,ExtraTreeRegressor(random_state=2019)
            ,AdaBoostRegressor(random_state=2019)
            ,BaggingRegressor(random_state=2019)
            ,ExtraTreesRegressor(random_state=2019)
            ,GradientBoostingRegressor(random_state=2019)
            ,RandomForestRegressor(random_state=2019)
#            ,VotingRegressor(random_state=2019)
#            ,HistGradientBoostingRegressor(random_state=2019)
            
#            ,ARDRegression(random_state=2019)
#            ,BayesianRidge(random_state=2019)
#            ,LarsCV(random_state=2019)
            ,LinearRegression(fit_intercept=True)
#            ,LogisticRegressionCV(random_state=2019)
#            ,PassiveAggressiveRegressor(random_state=2019)
#            ,RidgeCV(random_state=2019)
#            ,SGDRegressor(random_state=2019)
            ]
    
    if as_dict == True:
        modeling_entity = dict(zip(model_names, model_obj))
    else:
        modeling_entity=pd.DataFrame()              
        modeling_entity['Name']=model_names
        modeling_entity['Object']=model_obj    
    
    return modeling_entity

def getFileName(file_path, file_prefix, file_extension):
    file_name = file_prefix + "_1000000"
    for n in range(1,1000000):
        if path.exists(file_path + file_prefix + "_" + str(n) + "." + file_extension) == False:
            file_name = file_prefix + "_" + str(n) + "." + file_extension
            break
    
    return file_name

def getModelParamType():
    """
    for parameters supporting both int and float, specify value as 
        ['int', 'float']
        
    First the parameters which can also take None are written
    The other parameters which take only numeric values are then written followed by a break
    only. 
    """
    model_param_type_dict = {}
    
    model_param_type_dict['BaggingRegressor'] = { 
    	"base_estimator" : ['choice'],
    	"bootstrap" : ['boolean'],
    	"bootstrap_features" : ['boolean'],
    	"oob_score" : ['boolean'],
    	"n_jobs" : ['int', 'None'],
    	"random_state" : ['int', 'None'],
    	"n_estimators" : ['int'],
    	"max_samples" : ['int', 'float'],
    	"max_features" : ['int', 'float']
    }
    
    model_param_type_dict['ExtraTreesRegressor'] = { 
    	"criterion" : ['choice'],
    	"bootstrap" : ['boolean'],
    	"oob_score" : ['boolean'],
    	"max_depth" : ['int', 'None'],
    	"n_jobs" : ['int', 'None'],
    	"random_state" : ['int', 'None'],
    	"max_samples" : ['int', 'float', 'None'],
    	"n_estimators" : ['int'],
    	"min_samples_split" : ['int', 'float'],
    	"min_samples_leaf" : ['int', 'float'],
    	"min_weight_fraction_leaf" : ['float'],
    	"min_impurity_decrease" : ['float'],
    	"min_impurity_split" : ['float'],
    	"ccp_alpha" : ['float']
    }
    
    model_param_type_dict['AdaBoostRegressor'] = { 
    	"base_estimator" : ['choice'],
    	"loss" : ['choice'],
    	"random_state" : ['int', 'None'],
    	"n_estimators" : ['int'],
    	"learning_rate" : ['float']
    }
    
    model_param_type_dict['GradientBoostingRegressor'] = { 
    	"loss" : ['choice'],
    	"criterion" : ['choice'],
    	"init" : ['choice'],
    	"max_depth" : ['int', 'None'],
    	"random_state" : ['int', 'None'],
    	"max_leaf_nodes" : ['int', 'None'],
    	"learning_rate" : ['float'],
    	"n_estimators" : ['int'],
    	"subsample" : ['float'],
    	"min_samples_split" : ['int', 'float'],
    	"min_samples_leaf" : ['int', 'float'],
    	"min_weight_fraction_leaf" : ['float'],
    	"min_impurity_decrease" : ['float'],
    	"min_impurity_split" : ['float'],
    	"alpha" : ['float'],
    	"validation_fraction" : ['float'],
    	"n_iter_no_change" : ['int', 'None'],
    	"tol" : ['float'],
    	"ccp_alpha" : ['float']
    }
    
    model_param_type_dict['RandomForestRegressor'] = { 
    	"criterion" : ['choice'],
    	"bootstrap" : ['boolean'],
    	"oob_score" : ['boolean'],
    	"max_depth" : ['int', 'None'],
    	"max_leaf_nodes" : ['int', 'None'],
    	"n_jobs" : ['int', 'None'],
    	"random_state" : ['int', 'None'],
    	"max_samples" : ['int', 'float', 'None'],
    	"n_estimators" : ['int'],
    	"min_samples_split" : ['int', 'float'],
    	"min_samples_leaf" : ['int', 'float'],
    	"min_weight_fraction_leaf" : ['float'],
    	"min_impurity_decrease" : ['float'],
    	"min_impurity_split" : ['float'],
    	"ccp_alpha" : ['float']
    }
    
    model_param_type_dict['ExtraTreeRegressor'] = { 
    	"criterion" : ['choice'],
    	"splitter" : ['choice'],
    	"max_depth" : ['int', 'None'],
    	"random_state" : ['int', 'None'],
    	"max_leaf_nodes" : ['int', 'None'],
    	"min_samples_split" : ['int', 'float'],
    	"min_samples_leaf" : ['int', 'float'],
    	"min_weight_fraction_leaf" : ['float'],
    	"min_impurity_decrease" : ['float'],
    	"min_impurity_split" : ['float'],
    	"ccp_alpha" : ['float']
    }
    
    model_param_type_dict['LinearRegression'] = { 
    	"fit_intercept" : ['boolean'],
    	"normalize" : ['boolean'],
    	"n_jobs" : ['int', 'None']
    }

    return model_param_type_dict

def classification_report_dict(model, X_test, y_test, model_name, params='Base Model'):
    y_pred = model.predict(X_test)
    
    model_report = {}
    
    y_pred = model.predict(X_test)
    
    model_report['Model_Name'] = model_name
    model_report['Model_Params'] = params
    model_report['Parameters'] = str(model)
    
    #Evaluation Metrics
    mse = mean_squared_error(y_test, y_pred)
    model_report['Mean_Squared_Error'] = mse
    
    rmse = np.sqrt(mse)
    model_report['Root_Mean_Squared_Error'] = rmse
    
#    msle = mean_squared_log_error(y_test, y_pred)
#    model_report['Mean_Squared_Log_Error'] = msle
    
    SS_Residual = sum((y_test-y_pred)**2)
    SS_Total = sum((y_test-np.mean(y_test))**2)
    r_squared = 1 - (float(SS_Residual))/SS_Total
    adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
    
    model_report['R_squared'] = r_squared
    model_report['Adjusted_r_squared'] = adjusted_r_squared
    
    mae = mean_absolute_error(y_test, y_pred)
    model_report['Mean_Absolute_Error'] = mae
    
    return model_report

#def getFileName(file_path, file_prefix, file_extension):
#    file_name = file_prefix + "_1000000"
#    for n in range(1,1000000):
#        if path.exists(file_path + file_prefix + "_" + str(n) + "." + file_extension) == False:
#            file_name = file_prefix + "_" + str(n) + "." + file_extension
#            break
#    
#    return file_name

def csvToExcel(excel_dict, file_name_prefix = None):
    #Path to save results in
    file_path = getResultsPath()
    
    #Get a unique filename 
    if file_name_prefix == None:
        file_name_prefix = "Results"
    
    file_name = getFileName(file_path, file_name_prefix, "xlsx")
    
    #initialize the excel writer
    writer = pd.ExcelWriter(file_path + file_name, engine='xlsxwriter')
    
    #now loop through and put each on a specific sheet
    for sheet_name in  excel_dict.keys(): # .use .items for python 3.X
        sheet_data = excel_dict[sheet_name]
        sheet_data.to_excel(writer, sheet_name = sheet_name, index=False)
    
    #critical last step
    writer.save()

#########################################################################################
def runGrid(base_model, model_params_dict, errors_dict, model_name, data_df, timeout):
    X = data_df.drop(['Target'], axis=1)
    y = data_df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    base_models_dict = getModelersDF(as_dict=True)

    param_grid = ParameterGrid(model_params_dict)
    total_models = len(param_grid)
    
#    print('model_params_dict : ', model_params_dict)
#    print('param_grid : ', param_grid)

    results_sampling = {}
    
    print("Running " + model_name + " :")
    counter = 0
    for params in param_grid:
#        print(params)
        params_run = True
        
        if 'base_estimator' in params.keys():
            params['base_estimator'] = base_models_dict[params['base_estimator']]
            
        model = clone(base_model)
        
        print("Running " + str(counter) + " of " + str(total_models))
        
        try:
            model.set_params(**params)
            model.fit(X_train, y_train)
            
            timeout_modelling = RunWithTimeout(model.fit, (X_train, y_train) )
            result_timeout_modelling = timeout_modelling.run(timeout)
#            print("Model with params " + str(params) + " run.")
            
#            res = classification_report_dict(model, X_test, y_test, model_name, str(params))
#            
#            results_sampling[str(params)] = res
            
        except Exception as e:
            params_run = False
            
            errors_dict[model_name + str(counter)] = {
                                    'Model_Name' : model_name,
                                    'Error' : str(e),
                                    'Parameters_Provided' : str(params),
                                    'ML_Model' : str(model)
                                }
            pass
        
        counter +=1 
        
        if (params_run == True) and (result_timeout_modelling != None):
            res = classification_report_dict(model, X_test, y_test, model_name, str(params))
            results_sampling[str(params)] = res
        
    results_sampling_df = pd.DataFrame().from_dict(results_sampling, orient='index')

    return results_sampling_df, errors_dict

def checkSuffix(param_name):
    param_type = ""
    param_name_to_take = param_name
    actual_param_name = param_name

    if param_name.endswith('_custom'):
        param_type = "custom"
        actual_param_name = param_name[:-len("_custom")]
        param_name_to_take = actual_param_name

    elif param_name.endswith('_min'):
        param_type = "grid"
        actual_param_name = param_name[:-len("_min")]

    elif param_name.endswith('_max'):
        param_type = "grid"
        actual_param_name = param_name[:-len("_max")]

    elif param_name.endswith('_step'):
        param_type = "grid"
        actual_param_name = param_name[:-len("_step")]
        
    elif param_name.endswith('_none_checked'):
        param_type = "none_checkbox"
        actual_param_name = param_name[:-len("_none_checked")]
        
    elif param_name.endswith('max_features'):
        param_type = "max_features"
        actual_param_name = param_name[:-len("max_features")]

    return param_name_to_take, actual_param_name, param_type

def isFloat(str_as_num):
    split_num = str_as_num.split('.')
    if len(split_num) == 2:
        if (split_num[0].isdigit() == True) and (split_num[0].isdigit() == True):
            return True
        
    return False

def convertToAppropriateNumber(str_as_num, param_dtype):
    changed_str = ''
    if param_dtype == ['int'] or (param_dtype == ['int', 'None']):
        try:
            changed_str = int(str_as_num)
        except:
            str_as_num = str_as_num.split('.')[0]
            changed_str = int(str_as_num)
    
    elif (param_dtype == ['float']) or (param_dtype == ['float', 'None']):
        changed_str = float(str_as_num)
    
    elif (param_dtype == ['int', 'float']) or (param_dtype == ['int', 'float', 'None']):
        if isFloat(str_as_num) == True:
            changed_str = float(str_as_num)
        else:
            changed_str = int(str_as_num)
    else:
        raise TypeError('Only int or float values required')
    
    return changed_str

def deStringifyRequiredValues(user_input):
    """
    converts stringified values to their normal dtype. Eg:
        "None   -> None
        "True"  -> True
        "False" -> False
    """
    changed_user_input = []
    for value in user_input:
        if value in ["None","True","False"]:
            changed_value = ast.literal_eval(value)
            changed_user_input.append(changed_value)
        else:
            changed_user_input.append(value)
    
    return changed_user_input

def createModelParamsDict(request_info_dict):
    model_params_dict = {}
    grid_param_names = {}
    param_names_with_none = {}

    models_list = request_info_dict['models_list']
#    models_list = request_info_dict['models_list'][0].split('__')
    print('models_list')
    print(models_list)

    # for creating the super keys of the model_params_dict
    for model_name in models_list:
        model_params_dict[model_name] = {}
        grid_param_names[model_name] = []
        param_names_with_none[model_name] = []

    model_param_type_dict = getModelParamType()

    # for assigning params to their respective model combinations
    for key in request_info_dict.keys():
        user_input = request_info_dict[key]
        print(key)
        print(user_input)
        
#        if 'min' in key:
#            break    
        
        key_is_a_param = True
        try:
            param_name, model_name = key.split('_000')[0], key.split('_000')[1]
#            model_name = model_name.split('00')[0]
            param_name_to_take, actual_param_name, param_type = checkSuffix(param_name)

        except:
            key_is_a_param = False
            pass
        
        if key_is_a_param == True:
            if param_type == "grid":
                grid_param_names[model_name].append(actual_param_name)
                user_input = ''.join(user_input[0].split(' '))
                
                if user_input != '':
                    param_dtype = model_param_type_dict[model_name][actual_param_name]
                    user_input = convertToAppropriateNumber(user_input, param_dtype)

            elif param_type == "custom":
                #converting stringified list to list of numbers
                user_input = ''.join(user_input[0].split(' '))
                
                if user_input != '':
                    param_dtype = model_param_type_dict[model_name][actual_param_name]
                    user_input = [convertToAppropriateNumber(x, param_dtype) for x in user_input.split(',')]
                
            elif param_type == "none_checkbox":
                #add actual param name to a list of parameter names where at the end value 'None'
                #will be appended to the corresponsing parameter list (if exists) or create one
                if user_input == ['true']:
                    param_names_with_none[model_name].append(actual_param_name)
                
            elif param_type == 'max_features':
                user_input = ''.join(user_input[0].split(' '))
                    
                if user_input != '':
                    converted_values_list = []
                    for value in user_input.split(','):
                        converted_value = ''
                        try:
                            converted_value = convertToNum(value)
                        except:
                            if value == 'None':
                                converted_value = None
                            else:
                                converted_value = value
                        converted_values_list.append(converted_value)
                    user_input = converted_values_list
                
            elif param_type == "":
                user_input = deStringifyRequiredValues(user_input)

            if (user_input != '') and (param_type != "none_checkbox"):
                model_params_dict[model_name][param_name_to_take] = user_input

    # for changing grid arguments of (min,max,step) into one    
    for model_name in models_list:
#        break
        if len(grid_param_names[model_name]) > 0:
            grid_param_names_list = list(set(grid_param_names[model_name]))

            for actual_param_name_0 in grid_param_names_list:
#                break
                min_val = model_params_dict[model_name][actual_param_name_0 + '_min']
                max_val = model_params_dict[model_name][actual_param_name_0 + '_max']
                step_val = model_params_dict[model_name][actual_param_name_0 + '_step']

                calc_range = np.arange(min_val, max_val, step_val).tolist()

                model_params_dict[model_name][actual_param_name_0] = calc_range

                model_params_dict[model_name].pop(actual_param_name_0 + '_min', None)
                model_params_dict[model_name].pop(actual_param_name_0 + '_max', None)
                model_params_dict[model_name].pop(actual_param_name_0 + '_step', None)

    for model_name in models_list:
#        break
        if len(param_names_with_none[model_name]) > 0:
            param_names_with_none_curr_model = list(set(param_names_with_none[model_name]))

            for actual_param_name_1 in param_names_with_none_curr_model:
                try:
                    model_params_dict[model_name][actual_param_name_1].append(None)
                except:
                    model_params_dict[model_name][actual_param_name_1] = [None]
            
            model_params_dict[model_name].pop(actual_param_name_1 + '_none_checked', None)
            
    model_params_dict['ranking_order'] = request_info_dict['ranking_order']
    model_params_dict['timeout'] = request_info_dict['timeout']
            
    print('dictionary created')
    print(model_params_dict)
        
    return model_params_dict

def convertParamsToString(model_params_dict):
    model_params_str = ""
    
#    model_params_str = str(model_params_dict)
    
    for model_name in model_params_dict.keys():
        if model_name != 'ranking_order' and model_name != 'timeout':
            model_params = model_params_dict[model_name]
            
            if model_params != {}:
                model_params_str += 'Model : ' + model_name + 'newline'
                print('model_params')
                print(model_params)
                for param_name in model_params:
                    param_value = model_params[param_name]
                    
                    model_params_str += 'tabspace' + param_name + ' : ' + str(param_value) + 'newline'
            
    return model_params_str            

def getBestModelsInfo(all_results_df, ranking_metrics, no_of_models):
#    all_results_df = pd.DataFrame().from_dict(all_results, orient='index')
    
    print('ranking_metrics')
    print(ranking_metrics)
    
    all_results_df.index = list(range(len(all_results_df)))
    all_results_df_sorted = all_results_df.sort_values(ranking_metrics, ascending = [False] * len(ranking_metrics))
    
    #'n' to be created as a user defined value later : Done
    best_model_info = all_results_df_sorted.head(no_of_models)[['Model_Name','Parameters']]
    
    best_model_info.index = list(range(len(best_model_info)))
    
    best_model_info_dict = best_model_info.to_dict(orient='index')
    
#    best_model_info_dict = {
#            'model_name' : best_model['Model_Name'].values[0],
#            'parameters' : best_model['Parameters'].values[0]
#            }
    
    return best_model_info_dict



def changeStringNoneToNone(model_params_dict):
    for key1 in model_params_dict.keys():
        if key1 != 'ranking_order':
            for key2 in model_params_dict[key1].keys():
                values = model_params_dict[key1][key2]
                
                if 'None' in values:
                    model_params_dict[key1][key2] = [x for x in values if x != 'None']
                    model_params_dict[key1][key2].append(None)
    return model_params_dict

def convertToNum(num_str):
    if '.' in num_str:
        return float(num_str)
    else:
        return int(num_str)

def stringifyNone(model_params_dict):
    print('model_params_dict')
    print(model_params_dict)
    
#    This part is not required for regression
#    for model_name in model_params_dict.keys():
#        if '00' in model_name:
#            display_model_name = model_name.replace('00',' - ')
#            model_params_dict[display_model_name] = model_params_dict[model_name]
#            model_params_dict.pop(model_name)
    
    for key1 in model_params_dict.keys():
        if key1 != 'ranking_order' and key1 != 'timeout':
            for key2 in model_params_dict[key1].keys():
                model_params_dict[key1][key2] = [x if x != None else 'None' for x in model_params_dict[key1][key2]]
    return model_params_dict
    
#########################################################################################
#Main Functions
def getRegressionParamsForTuning(request_info_dict):
    model_params_dict = createModelParamsDict(request_info_dict)
    with open('Regression_Tuning_Params_for_backend.json', 'w') as fp:
        json.dump(model_params_dict, fp)
        
    model_params_dict_with_stringified_none = stringifyNone(model_params_dict)
    with open('Regression_Tuning_Params.json', 'w') as fp:
        json.dump(model_params_dict_with_stringified_none, fp)
        
#    model_params_str = convertParamsToString(model_params_dict)
    
    return model_params_dict
    
def runRegressionParamGrid(request_info_dict):
    model_params_dict = ""
    
    try:
        with open('Regression_Tuning_Params_for_backend.json', 'r') as fp:
            model_params_dict = json.load(fp)
    except:
        message_code = "FILE_NOT_READ"
        raise FileNotFoundError('Parameters file not found')
    
    best_model_info_dict = "None"
    all_results = pd.DataFrame()
    grid_errors_dict = {}
    
    if model_params_dict != '':
        message_code = "SUCCESS"
        
#        model_params_dict = changeStringNoneToNone(model_params_dict)
        data_file_path = getDataFilePath()
        data_file_name = getDataFileName()
        models_dict = getModelersDF(as_dict=True)
        ranking_metrics = model_params_dict['ranking_order'][0]
        timeout = convertToNum(model_params_dict['timeout'][0])
        
        print('model_params_dict')
        print(model_params_dict)
        
        if type(ranking_metrics) == str:
            ranking_metrics = [ranking_metrics]
        
        data_df = pd.read_csv(data_file_path + data_file_name)
        
        grid_results_dict = {}
        for model_name in model_params_dict.keys():
            if model_name != 'ranking_order' and model_name != 'timeout':
                sheet_name = model_name
                    
                #Other parameters for running the grid
                base_model = models_dict[model_name]    
                
                #Running Grid for One model-sampling combination
                model_grid_results, grid_errors_dict = runGrid(base_model, model_params_dict[model_name], grid_errors_dict, model_name, data_df, timeout)
                grid_results_dict[sheet_name] = model_grid_results
                
                all_results = all_results.append(model_grid_results)
                
        csvToExcel(grid_results_dict, file_name_prefix = 'Regression_Grid_Results')
        
        ranking_metrics = ranking_metrics[0].split('__')
#        best_model_info = getBestModelsInfo(all_results, ranking_metrics)
    
        if all_results.empty:
                message_code = "TIMEOUT_NO_MODELS"
                
        else:
#            csvToExcel(grid_results_dict, file_name_prefix = 'Regression_Grid_Results')
            
            no_of_models = 3
            best_model_info_dict = getBestModelsInfo(all_results, ranking_metrics, no_of_models)
            
#            static_path = r'C:\Users\sakshij\Documents\UI\django_test\practice_to_project\frontend\service-frontend\src\app\Static\\'
            static_path = getResultsPath()
            
            with open(static_path + 'Regression_Best_Model.json', 'w') as fp:
                json.dump(best_model_info_dict, fp)
                
            pd.DataFrame().from_dict(grid_errors_dict, orient='index').to_csv(static_path + "Errors_in_Regression_Grid.csv", index=False)
    
        print('best_model_info')
        print(best_model_info_dict)
        
        return best_model_info_dict, message_code, len(all_results)