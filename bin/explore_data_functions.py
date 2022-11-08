import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA

import scipy
from scipy.stats import shapiro   
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon

import time

import seaborn as sns

import statsmodels.stats.multitest

import random

import cimcb_lite as cb



'''
plots 3D PCA

input :
    - X : peakTable with only variable columns, no metadata
''' 
def PCA_3D(X, target):
    
    pca = PCA(n_components=3)
    components = pca.fit_transform(X)

    total_var = pca.explained_variance_ratio_.sum() * 100

    fig = px.scatter_3d(
        components, x=0, y=1, z=2, color=target,
        title=f'Total Explained Variance: {total_var:.2f}%',
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'},
        opacity=0.7
    )
    fig.show()
    
    

'''
plots 2D PCA, with all paired of principal components

inputs :
    - X : peakTable with only variable columns, no metadata
    - dimensions (default=3) : number of principal components
'''
def PCA_paired(X, target, dimensions=3):
    
    pca = PCA()
    components = pca.fit_transform(X)
    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    fig = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(dimensions),
        color=target
    )
    fig.update_traces(diagonal_visible=False)
    fig.show()
    
    
    
    
    
    
'''
Plots PCA with number of dimensions passed as argument.

If paired argument is set to False and number of dimensions is 2, we call function of cimcb_lite module.
If paired argument is set to False and number of dimensions is 3, we create a 3D plot with matplotlib.
If paired argument is set to True and whatever the number of dimensions, we plot a pairplot of all dimensions.

Inputs :
    - df : peakTable with only variable columns, no metadata
    - target : pandas series to plot differents colors on the PCA
    - dimensions (default=3) : int, number of principal components
    - paired (default=False) : boolean, choose type of plot
'''
def plot_PCA(df, target, dimensions=3, paired=False):
    
    # Copy input dataframe to avoid overwrite original dataframe
    df_copy = df.copy()
    
    # PCA display classic
    if not paired:
        
        # Assert dimensions passed as argument is 2 or 3
        assert (dimensions==2 or dimensions==3), '<dimensions> has to be 2 or 3'

        # PCA with 2 components
        if dimensions==2:
            if plot_infos_missing_values(df_copy, na_values=0, plot=False) == 0:
                cb.plot.pca(df_copy, pcx=1, pcy=2, group_label=target)
            else:
                print('NaNs in peak table, please impute before using PCA')

        # PCA with 3 components
        if dimensions==3:
            if plot_infos_missing_values(df_copy, na_values=0, plot=False) == 0:
                PCA_3D(df_copy, target)
            else:
                print('NaNs in peak table, please impute before using PCA')
                
    else:
        if plot_infos_missing_values(df_copy, na_values=0, plot=False) == 0:
            PCA_paired(df_copy, target, dimensions=dimensions)
        else:
            print('NaNs in peak table, please impute before using PCA')
            
    
    
    
    
    
    
    
    
    
    


    
    
'''
Test of normality with Shapiro-Wilk test : scipy.stats.shapiro

performs shapiro test for each column in dataframe

input :
    - X : peakTable with only variable columns, no metadata
return :
    - dataframe with statistic and pvalue for each variable (dataframe columns)
'''   
def shapiro_test_df(X):

    infos = []

    for col in X.columns:

        curr_infos = []

        shapiro_test = shapiro(X[col][X[col].notna()])
        curr_infos.extend([col, shapiro_test.statistic, shapiro_test.pvalue])

        infos.append(np.array(curr_infos))

    infos = pd.DataFrame(np.array(infos))
    infos.columns = ['Compounds', 'shapiro_score', 'pvalue']
    infos.index = infos['Compounds']
    infos = infos.drop(['Compounds'], axis=1)
    infos = infos.apply(pd.to_numeric, errors='coerce')

    return infos




'''
Paired t-test : scipy.stats.ttest_rel

For each column in dataframe, performs paired t-test between 'Incident' and 'Non-case' groups

input :
    - peakTable_HILIC_POS : whole peak table with variable and metadata
    - X : peakTable with only variable columns, no metadata
return :
    - dataframe with statistic and pvalue for each variable (dataframe columns)
'''  
def paired_ttest_df(peakTable_HILIC_POS, X):
    
    t0 = time.time()

    infos = []

    for variable in X.columns:

        curr_var = peakTable_HILIC_POS[['Groups', 'MatchCaseset', variable]]

        val_incident = []
        val_non_case = []
        case_id = []

        for elt in np.unique(curr_var['MatchCaseset']):

            curr_case = curr_var[curr_var['MatchCaseset'] == elt]

            case_id.append(elt)
            val_incident.append(curr_case[curr_case['Groups'] == 'Incident'][variable].values[0])
            val_non_case.append(curr_case[curr_case['Groups'] == 'Non-case'][variable].values[0])

        df_var = pd.concat([pd.Series(case_id), pd.Series(val_incident), pd.Series(val_non_case)], axis=1)
        df_var.columns = ['MatchCaseset', 'Incident', 'Non-case']


        curr_ttest = ttest_rel(df_var['Incident'].values, df_var['Non-case'].values, nan_policy='omit')
        curr_ttest_values = [variable, curr_ttest.statistic, curr_ttest.pvalue]

        infos.append(curr_ttest_values)


    infos = pd.DataFrame(np.array(infos))
    infos.columns = ['Variable', 'statistic', 'pvalue']
    infos.index = infos['Variable']
    infos = infos.drop(['Variable'], axis=1)
    infos = infos.apply(pd.to_numeric, errors='coerce')
    
    print('Time to compute : {0}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))))

    return infos




'''
Paired t-test : scipy.stats.ttest_rel
Wilcoxon test : scipy.stats.wilcoxon

For each column in dataframe :
- test with shapiro test if differences between paires are normally distributed
- if yes, performs paired t-test between 'Incident' and 'Non-case' groups
- if no, performs wilcoxon test between 'Incident' and 'Non-case' groups

input :
    - peakTable_HILIC_POS : whole peak table with variable and metadata
    - X : peakTable with only variable columns, no metadata
    - alpha_shapiro : threshold for shapiro test
return :
    - dataframe with boolean for normal distribution, applied test, statistic and pvalue for each variable (dataframe columns)
'''
def paired_test_t_or_Wilcoxon(peakTable_HILIC_POS, X, alpha_shapiro=0.05):
    
    t0 = time.time()

    infos = []

    for variable in X.columns:

        curr_var = peakTable_HILIC_POS[['Groups', 'MatchCaseset', variable]]

        val_incident = []
        val_non_case = []
        case_id = []
        val_diff = []

        for elt in np.unique(curr_var['MatchCaseset']):

            curr_case = curr_var[curr_var['MatchCaseset'] == elt]

            case_id.append(elt)
            val_incident.append(curr_case[curr_case['Groups'] == 'Incident'][variable].values[0])
            val_non_case.append(curr_case[curr_case['Groups'] == 'Non-case'][variable].values[0])
            val_diff.append((curr_case[curr_case['Groups'] == 'Incident'][variable].values[0]) - (curr_case[curr_case['Groups'] == 'Non-case'][variable].values[0]))
            
        df_var = pd.DataFrame(list(zip(case_id, val_incident, val_non_case, val_diff)), columns = ['MatchCaseset', 'Incident', 'Non-case', 'Diff'])

        if scipy.stats.shapiro(df_var['Diff'].values).pvalue > alpha_shapiro:
            
            # difference between paires is normally distributed so paired t-test (parametric)
            normally_distibuted = True
            test_applied = 'Paired t-test'
            
            curr_ttest = ttest_rel(df_var['Incident'].values, df_var['Non-case'].values, nan_policy='omit')
            curr_ttest_values = [variable, normally_distibuted, test_applied, curr_ttest.statistic, curr_ttest.pvalue]
            
        else:
            
            # difference between paires isn't normally distributed so Wilcoxon test (non-parametric)
            normally_distibuted = False
            test_applied = 'Wilcoxon'
            
            curr_ttest = wilcoxon(df_var['Diff'].values)
            curr_ttest_values = [variable, normally_distibuted, test_applied, curr_ttest.statistic, curr_ttest.pvalue]
            
        infos.append(curr_ttest_values)


    infos = pd.DataFrame(np.array(infos))
    infos.columns = ['Variable', 'NormallyDistributed', 'TestApplied', 'statistic', 'pvalue']
    infos.index = infos['Variable']
    infos = infos.drop(['Variable'], axis=1)
    infos['statistic'] = infos['statistic'].apply(pd.to_numeric, errors='coerce')
    infos['pvalue'] = infos['pvalue'].apply(pd.to_numeric, errors='coerce')
    
    
    print('Time to compute : {0}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))))

    return infos






   
'''
Plot the relative log abundance for each variable and each sample
input :
    - X : peakTable with only variable columns, no metadata
'''
def plot_relative_log_abundance(X):
    
    plt.figure(figsize=(50, 30))

    plt.suptitle('Relative log abundance', fontsize=50)

    plt.subplot(2, 1, 1)
    plt.subplots_adjust(top=0.92)
    rel_log_abundance_metabolite = np.log(X) - np.log(X).median()
    
    xlabel_fontsize = min(3000/X.shape[1], 30)
    boxplot = sns.boxplot(data=rel_log_abundance_metabolite)
    boxplot.set_xticklabels(boxplot.get_xticklabels(), rotation=90, fontsize=xlabel_fontsize)
    plt.title('Based on metabolites', fontsize=40)
    
    plt.subplot(2, 1, 2)
    rel_log_abundance_sample = np.log(X) - np.array(np.log(X).median(axis=1)).reshape(X.median(axis=1).shape[0], 1)
    boxplot = sns.boxplot(data=rel_log_abundance_sample.transpose())
    boxplot.set_xticklabels(boxplot.get_xticklabels(), rotation=90, fontsize=18)
    plt.title('Based on samples', fontsize=40)

    plt.subplots_adjust(hspace=0.4)
    plt.show()
    
    
    
    
    
    

    
    
    
    
def remove_correlated_features(df, threshold=0.95, method='spearman'):
    
    t0 = time.time()
    
    print(f'Initial shape : {df.shape}')
    
    cor_matrix = df.corr(method=method).abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    df1 = df.drop(to_drop, axis=1)
    
    print(f'Final shape : {df1.shape}')
    
    print(f'\nTime to compute : {time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - t0))}')
    print('\n', 150 * '#')
    print(3 * '\n')
    
    return df1







def plot_infos_missing_values(X, na_values=None, plot=True):
    
    if plot:
        print(150 * '#')
    
    if isinstance(na_values, type(None)):
        tot_nb_mv = X.isna().sum().sum()
        heatmap = X.isna()
    else:
        tot_nb_mv = (X == na_values).sum().sum()
        heatmap = (X == na_values)
    
    if plot:
        print(f'Considering {na_values} as missing values :')
        print(f'- Total number of missing values : {tot_nb_mv}')
        print(f'- Percent of missing values : {tot_nb_mv / X.size * 100 :.2f}%\n')
    
    if tot_nb_mv != 0 and plot:
        
        # Heatmap missing values
        plt.figure(figsize=(15,8))
        sns.heatmap(~heatmap, cbar=False)
        plt.xlabel('features', fontsize=12)
        plt.ylabel('samples', fontsize=12)
        plt.title('Heatmap of missing values\n(Dark points correspond to missing values)', fontsize=18)
        plt.show()
        print('\n')
        
        # Part of samples missing each compound
        perc_mv = heatmap.sum() / X.shape[0]
        plt.figure(figsize=(15,8))
        plt.plot(perc_mv.sort_values().values, color='b', linewidth=3)
        plt.xlabel('N° of the compound', fontsize=12)
        plt.ylabel('Part of samples missing the compound', fontsize=12)
        plt.title('The part of samples missing each compound', fontsize=18)
        plt.grid(linestyle='--', linewidth=1)
        plt.show()
        print('\n')
    
    else:
        return tot_nb_mv
        
    print(150 * '#')
    
    
    
    
    
    
    
def plot_feature_types(peakTable):
    
    print(150 * '#')
    
    print(f'Data types : \n{peakTable.dtypes.value_counts()}\n')
    
    plt.figure(figsize=(8,4))
    ax = sns.countplot(x=peakTable.dtypes)
    ax.bar_label(ax.containers[0], fontsize=12)
    plt.show()
    
    print(150 * '#')
    
    
    
    
    
    
# threshold argument correspond to the absolute value of correlation above which we enlighten the point of the second plot
def plot_correlation_matrix(X, threshold=0.9, method='spearman', annot=False, fmt=None):
    
    print(150 * '#')
    
    corr = X.corr(method=method)
    
    # Plot correlation matrix
    plt.figure(figsize=(20,10))
    sns.heatmap(corr, annot=annot, fmt=fmt)
    plt.title('Correlation matrix', fontsize=18)
    plt.show()
    print('\n')
    
    # Plot heatmap of highly correlated features
    plt.figure(figsize=(20,10))
    sns.heatmap(corr.abs() > threshold)
    plt.title('Heatmap of highly correlated features', fontsize=18)
    plt.show()
    print('\n')
    
    print(150 * '#', '\n')
    
    
    
    
    
    
    
    
    
def randomly_remove_values_in_df(peakTable, part_removed=0.1):
    
    peakTable_with_Nan = peakTable.copy()

    for i in range(int(part_removed * peakTable.size)):
    
        peakTable_with_Nan.iloc[random.randrange(0, peakTable.shape[0]), random.randrange(0, peakTable.shape[1])] = np.nan
        
    return peakTable_with_Nan





def plot_target(target):
    
    print(150 * '#')
    
    print(f'Target values : \n{target.value_counts()}\n')
    
    plt.figure(figsize=(8,4))
    ax = sns.countplot(x=target);
    ax.bar_label(ax.containers[0]);
    plt.title('Countplot of target values', fontsize=18)
    plt.show()
    
    print(150 * '#', '\n')
    
    
    
    
    
    
    
    
    
    
    
def plot_pairplot_distributions(X, target, nb_features=None, list_features=None):

    print(150 * '#')
    
    if (isinstance(nb_features, type(None)) and isinstance(list_features, type(None))):
        nb_features = 5
        X_subset = X.iloc[:, :nb_features]
    elif (not isinstance(nb_features, type(None)) and isinstance(list_features, type(None))):
        X_subset = X.iloc[:, :nb_features]
    elif (isinstance(nb_features, type(None)) and not isinstance(list_features, type(None))):
        X_subset = X.loc[:, list_features]
    else:
        assert (isinstance(nb_features, type(None)) or isinstance(list_features, type(None))),\
            '<nb_features> and <list_features> cannot be both passed as argument'
    
    print()
    
    fig = plt.figure(figsize=(10,10));

    data = pd.concat([target, X_subset], axis=1)
    g = sns.pairplot(data=data, hue=target.name);

    for ax in g.axes.flatten():
        # rotate x axis labels
        ax.set_xlabel(ax.get_xlabel(), rotation = 45);
        # rotate y axis labels
        ax.set_ylabel(ax.get_ylabel(), rotation = 45);
        # set y labels alignment
        ax.yaxis.get_label().set_horizontalalignment('right');

    g.fig.suptitle(t='Pairplot of chosen features distributions', y=1.05, fontsize=18);
    plt.show();
    
    print(150 * '#', '\n')
    
    
    
    
    
    
    
    
    
    
    
def plot_hist_boxplot_distributions(X, target, nb_features=None, list_features=None):

    print(150 * '#')
    
    if (isinstance(nb_features, type(None)) and isinstance(list_features, type(None))):
        nb_features = 5
        X_subset = X.iloc[:, :nb_features]
    elif (not isinstance(nb_features, type(None)) and isinstance(list_features, type(None))):
        X_subset = X.iloc[:, :nb_features]
    elif (isinstance(nb_features, type(None)) and not isinstance(list_features, type(None))):
        X_subset = X.loc[:, list_features]
    else:
        assert (isinstance(nb_features, type(None)) or isinstance(list_features, type(None))),\
            '<nb_features> and <list_features> cannot be both passed as argument'
    
    print()
    
    data = pd.concat([target, X_subset], axis=1)
    
    for col in [col for col in X_subset.columns]:

        print(120 * '-')
        
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(data=data, x=data[col], hue=target.name, kde=True, element='step')
        plt.title('Histogram')

        plt.subplot(1, 2, 2)
        sns.boxplot(data=data, x=target.name, y=col)
        plt.title('Boxplot')

        plt.suptitle(col, fontsize=16)

        plt.show()
        
    print(120 * '-', '\n')
    print(150 * '#')
    
    
    
    
    
    
    
    
    
def plot_nb_unique_qualitative_metadata(metadata):
    
    print(150 * '#')

    unique_values = metadata[metadata.dtypes[metadata.dtypes == 'object'].index].nunique().sort_values()

    plt.figure(figsize=(8,4))
    ax = sns.barplot(x=unique_values.index, y=unique_values.values);
    ax.bar_label(ax.containers[0], fontsize=12)
    plt.show()
    
    print(150 * '#')
    