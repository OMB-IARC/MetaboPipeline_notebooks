


import numpy as np
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import time

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error

    

'''
Replace missing values in the dataframe to numpy NaNs

inputs :
- X : pandas dataframe, peakTable with only variable columns, no metadata
- na_values (default=None) : value of missing values we want to convert to numpy NaNs

return :
- X_ : pandas dataframe, same as input dataframe with initial missing values replaced with numpy NaNs
'''
def set_missing_value_to_Nan(X, na_values=None):
    
    X_ = X.copy()
    
    X_.replace(na_values, np.nan, inplace=True)
    
    return X_






'''
imputes missing values in peak table with constant value given as parameters

inputs :
    - X : peakTable with only variable columns, no metadata
    - const (default=0): value to impute in place of NaN
'''
def const_imputer(X, const=0):
    
    X_const = X.copy()
    X_const[X_const.isna()] = const
    return X_const



'''
For each feature, the missing values are imputed by the mean value of the non-missing values in that feature

input :
    - X : peakTable with only variable columns, no metadata
'''
def mean_imputer(X):
    
    imp = SimpleImputer(strategy='mean')
    imp.fit(X)

    X_mean = pd.DataFrame(imp.transform(X), columns=X.columns)
    return X_mean


'''
For each feature, the missing values are imputed by the median value of the non-missing values in that feature

input :
    - X : peakTable with only variable columns, no metadata
'''
def median_imputer(X):
    
    imp = SimpleImputer(strategy='median')
    imp.fit(X)

    X_median = pd.DataFrame(imp.transform(X), columns=X.columns)
    return X_median


'''
For each feature, the missing values are imputed by the most frequent value (rounded at 1.e-2) of the non-missing values in that feature

input :
    - X : peakTable with only variable columns, no metadata
'''
def mode_imputer(X):
    
    imp = SimpleImputer(strategy='most_frequent')
    imp.fit(round(X,2))

    X_most = pd.DataFrame(imp.transform(X), columns=X.columns)
    return X_most



'''
For each feature, the missing values are imputed by the minimum value of the non-missing values in that feature

input :
    - X : peakTable with only variable columns, no metadata
'''
def min_imputer(X):

    X_min = X.fillna(value=X.min())
    return X_min


'''
For each feature, the missing values are imputed by the half of the minimum value of the non-missing values in that feature

input :
    - X : peakTable with only variable columns, no metadata
'''
def half_min_imputer(X):

    X_half_min = X.fillna(value=X.min()/2)
    return X_half_min




'''
For each feature, the missing values are imputed using the MICE method (Multivariate Imputation by Chained Equations), inspired by the R MICE package.

input :
    - X : peakTable with only variable columns, no metadata
'''
def python_MICE_imputer(X, estimator):

    start_time = time.time()

    imp = IterativeImputer(estimator=estimator, max_iter=10, random_state=0, n_nearest_features=10)
    imp.fit(X)

    X_python_MICE = pd.DataFrame(imp.transform(X), columns=X.columns)
    
    print("----- {0:.1f} seconds -----".format(time.time() - start_time))
          
    return X_python_MICE




'''
"Each missing feature is imputed using values from n_neighbors nearest neighbors that have a value for the feature. The feature of the neighbors are averaged uniformly or weighted by distance to each neighbor. If a sample has more than one feature missing, then the neighbors for that sample can be different depending on the particular feature being imputed."

inputs :
    - X : peakTable with only variable columns, no metadata
    - n_neighbors (default=5) : number of neighbors to impute each feature
    - by (default='features') : allows to choose axis along which perform the imputation
'''
def KNN_imputer(X, n_neighbors=5, by='features'):
    
    imp = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
    
    if by=='samples':

        imp.fit(X)
        
        X_imp_KNN = pd.DataFrame(imp.transform(X), columns=X.columns)
        return X_imp_KNN
    
    elif by=='features':
        
        imp.fit(X.transpose())
        
        X_imp_KNN = pd.DataFrame(imp.transform(X.transpose())).transpose()
        X_imp_KNN.columns = X.columns
        
        return X_imp_KNN
    
    else:
        print('Wrong argument for <by>')

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
'''
The point of this function is to determine the best method to impute missing values in the peak table.

For that :
- we get the rate of missing values (<rate_NaN>) in the whole peak table (X passed as argument)
- we select only features with 0% of missing values from X (X_subset)
- we randomly remove <rate_NaN>% of values from X_subset
- we re-impute these values with all the available methods
- we compute the RMSE (Root Mean Squared Error) between the imputed values and the real values to know the most relevant method for the particular peak table

Arguments :
- X : peak table with only feature values (no metadata)
- path_peakTable_imputed : path to save imputed peak tables
- number_iterations (default=1) : number of times we compute the imputation (to decrease imputation variability for some methods)

Return :
- dataframe of ranked imputation methods (in decreasing order of RMSE)
'''
def choose_best_imputation_method(X, path_peakTable_imputed, number_iterations=1):
    
    print(150 * '#')
    print(66 * '#', 'Start processing', 66 * '#')
    print(150 * '#', '\n')
    
    print(f'--> Inputed dataframe has shape : {X.shape}\n')
    
    # Set path to save subset peak tables used to evaluate the best imputation method
    #path_peakTable_imputed_evaluate = path_peakTable_imputed + 'evaluate/'
    #if os.path.exists(path_peakTable_imputed_evaluate):
    #    print(f'--> Directory to save peak table for evaluation {path_peakTable_imputed_evaluate} already exists !\n')
    #else:
    #    os.makedirs(path_peakTable_imputed_evaluate)
    #    print(f'--> Directory to save peak table for evaluation {path_peakTable_imputed_evaluate} created !\n')


    # Get list of used methods
    list_X_imp = [X_imp for X_imp in os.listdir(path_peakTable_imputed) if X_imp.endswith('.csv')]
    list_methods = ['_'.join(X_imp.split('.')[0].split('_')[1:]) for X_imp in list_X_imp]
    #print(list_methods)
    #list_methods = [method for method in list_methods if method[:2] != 'R_']
    print(f'--> List of used imputation methods : {list_methods}\n')
    
    dict_RMSE = {key: [] for key in list_methods}
    #print(dict_RMSE)
    
    # Get the rate of NaN in the original peak table
    rate_NaN = X.isna().sum().sum()/X.size
    print(f'--> Rate of NaNs in the inputed peak table : {rate_NaN*100:.2f}%\n')
    
    
    
    # Prepare function for each imputation method
    def do_const_0(X_):
        return const_imputer(X_)

    def do_const_1(X_):
        return const_imputer(X_, 1)

    def do_mean(X_):
        return mean_imputer(X_)

    def do_median(X_):
        return median_imputer(X_)

    def do_mode(X_):
        return mode_imputer(X_)

    def do_min(X_):
        return min_imputer(X_)

    def do_half_min(X_):
        return half_min_imputer(X_)

    def do_python_MICE_BayesianRidge(X_):
        return python_MICE_imputer(X_, BayesianRidge())

    def do_python_MICE_DecisionTreeRegressor(X_):
        return python_MICE_imputer(X_, DecisionTreeRegressor(max_features='sqrt', random_state=0))

    def do_python_MICE_ExtraTreesRegressor(X_):
        return python_MICE_imputer(X_, ExtraTreesRegressor(n_estimators=10, random_state=0))

    def do_python_MICE_KNeighborsRegressor(X_):
        return python_MICE_imputer(X_, KNeighborsRegressor(n_neighbors=15))

    def do_KNN_features(X_):
        return KNN_imputer(X_, by='features')

    def do_KNN_samples(X_):
        return KNN_imputer(X_, by='samples')
    

    

    
    dict_methods = {
        'const_0': do_const_0,
        'const_1': do_const_1,
        'mean': do_mean,
        'median': do_median,
        'mode': do_mode,
        'min': do_min,
        'half_min': do_half_min,
        'python_MICE_BayesianRidge': do_python_MICE_BayesianRidge,
        'python_MICE_DecisionTreeRegressor': do_python_MICE_DecisionTreeRegressor,
        'python_MICE_ExtraTreesRegressor': do_python_MICE_ExtraTreesRegressor,
        'python_MICE_KNeighborsRegressor': do_python_MICE_KNeighborsRegressor,
        'KNN_features': do_KNN_features,
        'KNN_samples': do_KNN_samples
    }



    

    
    # Loop over the specified number of iterations
    for i in range(number_iterations):
        
        print(150 * '-')
        print(f'{70 * "-"} Iteration {i+1} {(70 - len(str(i+1)) -2) * "-"}')
        
        subset = X.dropna(axis=1)
        subset_with_NaN = subset.copy() # copy subset that we will later fill with NaNs
        
        print(f'--> Subset dataframe has shape : {subset_with_NaN.shape}\n')
        
        
        # We randomly remove <rate_NaN>% of values in <subset_with_NaN>
        # TODO : Maybe we could remove a random % of values ? Maybe a random number from a normal distribution centered on <rate_NaN>% ?
        ix = [(row, col) for row in range(subset_with_NaN.shape[0]) for col in range(subset_with_NaN.shape[1])]
        for row, col in random.sample(ix, int(round(rate_NaN * len(ix)))):
            subset_with_NaN.iat[row, col] = np.nan
        
        
        for method in list_methods:
            print(f'Processing : {method}..........')
            curr_X_imp = dict_methods[method](subset_with_NaN)
            dict_RMSE[method].append(mean_squared_error(subset, curr_X_imp))
            
            del curr_X_imp
            
            print()
        

        print()
        print(f'--> After iteration {i+1}, dict_RMSE updated')
        #print(dict_RMSE)
        
        del subset
        del subset_with_NaN
        
        print(150 * '-')
        print('\n\n')
       
    
    dict_RMSE_mean = {key:sum(dict_RMSE[key])/len(dict_RMSE[key]) for key in dict_RMSE.keys()}
    print(dict_RMSE_mean)
    
    sorted_dict_RMSE_mean = dict(sorted(dict_RMSE_mean.items(),
                                        key=lambda item: item[1],
                                        reverse=True))
    
    
    print(150 * '#')
    
    return pd.DataFrame(sorted_dict_RMSE_mean.items(), columns=['Method', 'RMSE'])













'''
Plot barplot of RMSE values for each imputation method.

Input:
- dF_RMSE : dataframe of ranked imputation methods (in decreasing order of RMSE), output of choose_best_imputation_method function
'''
def plot_RMSE_results(df_RMSE):
    
    plt.figure(figsize=(15, 10))
    ax = sns.barplot(x='Method', y='RMSE', data=df_RMSE);
    plt.xticks(rotation=90)
    ax.set_title('RMSE for each imputation method', fontsize=16)
    ax.bar_label(ax.containers[0], fmt='%.3f');