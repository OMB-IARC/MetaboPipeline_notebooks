

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.multitest


'''
This function tests, for each column of inputed dataframe, if the feature is normally distributed.
We apply both Shapiro and d'Agostino tests : both tests has to be significant for the feature to be considered normally distributed.

Inputs :
- X : pandas dataframe, only containing features intensities
- alpha (default=0.05) : float, significance level for normality test
- display_result (default=False) : boolean, prints or not results of all tests for all features.

Output :
- pandas dataframe, containing 2 columns :
    - Features : feature name
    - Normally distributed : given the alpha parameter, whether the feature normally distributed or not
'''
def normality_test_features(X, alpha=0.05, display_results=False):

    def Shapiro_Wilk_test(feature):
        
        from scipy.stats import shapiro
        return shapiro(feature)
    
    def dAgostino_test(feature):
        
        from scipy.stats import normaltest
        return normaltest(feature)
    
    #def Kolmogorov_Smirnov_test(feature):
        
    #    from scipy.stats import kstest
    #    return kstest(feature, 'norm')
    
    
    results = []
    
    for i, col in enumerate(X):
    
        # Shapiro-Wilk test
        stat1, p1 = Shapiro_Wilk_test(X.iloc[:,i])
        # D’Agostino’s K^2 test
        stats2, p2 = dAgostino_test(X.iloc[:,i])
        # Kolmogorov-Smirnov test
        #stats3, p3 = Kolmogorov_Smirnov_test(X.iloc[:,i])
        
        if display_results:
            print(100 * '#')
            print(col, ':')
            
            print(f'\tShapiro-Wilk test :\n\t\tStatistics={stat1:.3f}, p={p1}')
            print(f'\tD’Agostino’s K^2 test :\n\t\tStatistics={stats2:.3f}, p={p2}')
            #print(f'\tKolmogorov-Smirnov test :\n\t\tStatistics={stats3:.3f}, p={p3}')
        
            print(100 * '#', '\n')
        
        # Check if significant for alpha threshold passed as argument
        #if (p1 < alpha) or (p2 < alpha) or (p3 < alpha):
        if (p1 < alpha) or (p2 < alpha):
        #if p1 < alpha:
            normally_distributed = False
            #print(f'\nFeature {col} is normally distributed')
        else:
            normally_distributed = True
            #print(f'\nFeature {col} is not normally distributed')
        
        results.append([col, normally_distributed])
        
        
        
    return pd.DataFrame(results, columns=['Features', 'Normally distributed'])















'''
This function test if feature values are significantly different between the two classes passed as parameters.

First, each class in each column of X is tested to check if it comes from a normally distributed population, to apply either parametrical or non-parametrical test later on.
Then, each column of X is tested to check if the two classes distributions are from populations with equal variances.
Finally, we apply the adapted two-sample location test to check :
- if the distributions of both populations (classes) have equal means (independant t-test and Welch's t-test) or,
- if the distributions of both populations (classes) are identical (Mann–Whitney U test).


Inputs :
- X : pandas dataframe, only containing features intensities
- target : pandas series, has to have a length equal to number of rows in X, containing target values for each row in X
- alpha_normality_test (default=0.05) : float, significance level for normality test
- alpha_variance_equality_test (default=0.05) : float, significance level for variance equality test
- alpha_stat_test (default=0.05) : float, significance level for final statistical test test
- display_result (default=False) : boolean, prints or not results of all tests for all features.

Output :
- pandas dataframe, containing 6 columns :
    - Features : feature name
    - Normally distributed : given the alpha_normality_test parameter, whether both classes are normally distributed or not
    - Variance equality test : applied test for variance equality
    - Equal variances : given the alpha_variance_equality_test parameter, whether variances of classes distributions or equal or not
    - Two-sample location test : applied test to check significant differences between the two classes
    - H0 rejected : given the alpha_stat_test parameter, whether we reject H0 or not
'''
def features_tests(X, target, alpha_normality_test=0.05, alpha_variance_equality_test=0.05, alpha_stat_test=0.05, display_results=False):
    
    results = []
    
    uniques_target = np.unique(target)

    index_1 = target[target == uniques_target[0]].index
    X_1 = X.iloc[index_1, :]

    index_2 = target[target == uniques_target[1]].index
    X_2 = X.iloc[index_2, :]
    
    ### Test normality for both target classes
    normally_distibuted_1 = normality_test_features(X_1, alpha=alpha_normality_test)
    normally_distibuted_2 = normality_test_features(X_2, alpha=alpha_normality_test)
    

    for i, col in enumerate(X):
        
        results_currvar = [col]
        
        X_1_currvar = X_1[col]
        X_2_currvar = X_2[col]
        
        if display_results:
            print(100 * '-')
            print(col, ':')
        
        normal_currvar = normally_distibuted_1.iloc[0]['Normally distributed'] and normally_distibuted_2.iloc[0]['Normally distributed']
        results_currvar.append(normal_currvar)
        
        ### If normal :
        if normal_currvar:
            
            ### Test variance equality
            # normal : Barlett's test --> scipy.stats.bartlett
            from scipy.stats import bartlett
            variance_applied_test = "Barlett's test"
            results_currvar.append(variance_applied_test)
            stat, p = bartlett(X_1_currvar, X_2_currvar)
            equal_var = p > alpha_variance_equality
            results_currvar.append(equal_var)
            if display_results: print(f"\t- Test variance equality, Levene's test :\n\t\tstatistics={stat:.4f}, p-value={p} --> equal_var={equal_var}")
            
            if equal_var:
                # if normal and equal variances : Student t-test independant -->  scipy.stats.ttest_ind(..., equal_var=True)
                applied_test = "Independant t-test"
                results_currvar.append(applied_test)
                from scipy.stats import ttest_ind
                stat, p = ttest_ind(X_1_currvar, X_2_currvar, equal_var=True)
                results_currvar.extend([stat, p, alpha_stat_test])
                signif_diff = p < alpha_stat_test
                results_currvar.append(signif_diff)
                if display_results: print(f"\t- Two-sample location test, {applied_test} :\n\t\tstatistics={stat:.4f}, p-value={p} --> signif_diff={signif_diff}")
            else:
                # if normal and not equal variances : Welch's t-test -->  scipy.stats.ttest_ind(..., equal_var=False)
                applied_test = "Welch's t-test"
                results_currvar.append(applied_test)
                from scipy.stats import ttest_ind
                stat, p = ttest_ind(X_1_currvar, X_2_currvar, equal_var=False)
                results_currvar.extend([stat, p, alpha_stat_test])
                signif_diff = p < alpha_stat_test
                results_currvar.append(signif_diff)
                if display_results: print(f"\t- Two-sample location test, {applied_test} :\n\t\tstatistics={stat:.4f}, p-value={p} --> signif_diff={signif_diff}")
        
        
        ### If not normal :
        else:
             
            ### Test variance equality
            # not normal : Levene's test --> scipy.stats.levene
            from scipy.stats import levene
            variance_applied_test = "Levene's test"
            results_currvar.append(variance_applied_test)
            stat, p = levene(X_1_currvar, X_2_currvar)
            equal_var = p > alpha_variance_equality_test
            results_currvar.append(equal_var)
            if display_results: print(f"\t- Test variance equality, Levene's test :\n\t\tstatistics={stat:.4f}, p-value={p} --> equal_var={equal_var}")
            
            if equal_var:
                # if not normal and equal variances : Mann-Whitney U test --> scipy.stats.mannwhitneyu
                applied_test = "Mann-Whitney U test"
                results_currvar.append(applied_test)
                from scipy.stats import mannwhitneyu
                stat, p = mannwhitneyu(X_1_currvar, X_2_currvar)
                results_currvar.extend([stat, p, alpha_stat_test])
                signif_diff = p < alpha_stat_test
                results_currvar.append(signif_diff)
                if display_results: print(f"\t- Two-sample location test, {applied_test} :\n\t\tstatistics={stat:.4f}, p-value={p} --> signif_diff={signif_diff}")
        
                
                
            else:
                # if not normal and not equal variances : Welch's t-test -->  scipy.stats.ttest_ind(..., equal_var=False)
                # (the test is moderately robust against unequal variance if sample sizes are close : differ by a ratio of 3 or less)
                applied_test = "Welch's t-test"
                results_currvar.append(applied_test)
                from scipy.stats import ttest_ind
                stat, p = ttest_ind(X_1_currvar, X_2_currvar, equal_var=False)
                results_currvar.extend([stat, p, alpha_stat_test])
                signif_diff = p < alpha_stat_test
                results_currvar.append(signif_diff)
                if display_results: print(f"\t- Two-sample location test, {applied_test} :\n\t\tstatistics={stat:.4f}, p-value={p} --> signif_diff={signif_diff}")
        
        
        
        results.append(results_currvar)
        
        if display_results: print(100 * '-')

    
    return pd.DataFrame(results, columns=['Features', 'Normally distributed', 'Variance equality test', 'Equal variances', 'Two-sample location test', 'statistic', 'pvalue', 'alpha', 'H0 rejected']), alpha_stat_test












def get_significant_features(features_tests_results, corrected=False):
    
    col = 'H0 rejected'
    if corrected: col += ' corrected'
    
    part_signif = len(features_tests_results[features_tests_results[col]]) / len(features_tests_results) * 100
    print(f"With alpha={features_tests_results['alpha'][0]}, {part_signif:.2f}% of feature ({len(features_tests_results[features_tests_results[col]])}/{len(features_tests_results)}) are considered significantly different between the two classes.\n")
    
    return features_tests_results[features_tests_results['H0 rejected']]














    
'''
Plot the histogram of pvalues
input :
    - df : dataframe output of function paired_test_t_or_Wilcoxon
    - alpha : chosen threshold for alpha
'''  
def plot_hist_pvalue(df, alpha=0.05, plot_corrected=False):
    
    if plot_corrected:
        
        fig, ax = plt.subplots(1, 2, figsize=(24, 8))

        freq, bins, _ = ax[0].hist(df['pvalue'], np.arange(0, 1, alpha), color='coral', edgecolor='black', alpha=0.5, label=f'pvalue > {alpha}')
        count, _ = np.histogram(df['pvalue'], bins)
        for x,y,num in zip(bins, freq, count):
            if num != 0:
                ax[0].text(x+alpha/3, y+1, num, fontsize=10) # x,y,str
        ax[0].set_xticks(np.arange(0, 1, 0.05))
        ax[0].set_title('Histogram of p-values', fontsize=20)

        # add green on bin with pvalue < alpha
        hist2 = ax[0].hist(df['pvalue'][df['pvalue'] < alpha], np.arange(0, 1, alpha), color='mediumseagreen', edgecolor='black', alpha=1, label=f'pvalue < {alpha}')

        ax[0].legend(loc='upper right', prop={"size":15})   
        
        
        
        freq, bins, _ = ax[1].hist(df['pvalue corrected'], np.arange(0, 1, alpha), color='coral', edgecolor='black', alpha=0.5, label=f'pvalue corrected > {alpha}')
        count, _ = np.histogram(df['pvalue corrected'], bins)
        for x,y,num in zip(bins, freq, count):
            if num != 0:
                ax[1].text(x+alpha/3, y+1, num, fontsize=10) # x,y,str
        ax[1].set_xticks(np.arange(0, 1, 0.05))
        ax[1].set_title('Histogram of corrected p-values', fontsize=20)

        # add green on bin with pvalue < alpha
        hist2 = ax[1].hist(df['pvalue corrected'][df['pvalue corrected'] < alpha], np.arange(0, 1, alpha), color='mediumseagreen', edgecolor='black', alpha=1, label=f'pvalue corrected < {alpha}')

        ax[1].legend(loc='upper right', prop={"size":15})  

        
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        freq, bins, _ = ax.hist(df['pvalue'], np.arange(0, 1, alpha), color='coral', edgecolor='black', alpha=0.5, label=f'pvalue > {alpha}')
        count, _ = np.histogram(df['pvalue'], bins)
        for x,y,num in zip(bins, freq, count):
            if num != 0:
                ax.text(x+alpha/3, y+1, num, fontsize=10) # x,y,str
        ax.set_xticks(np.arange(0, 1, 0.05))
        plt.title('Histogram of p-values', fontsize=20)

        # add green on bin with pvalue < alpha
        hist2 = ax.hist(df['pvalue'][df['pvalue'] < alpha], np.arange(0, 1, alpha), color='mediumseagreen', edgecolor='black', alpha=1, label=f'pvalue < {alpha}')

        plt.legend(loc='upper right', prop={"size":15})       
        

    plt.show()
    
    
    
    
    
    
    
    
    
    

    
    
    
'''
Correct the list of pvalues using the inputed correction type.
input :
    - df : dataframe output of function features_tests
    - correction: name of the multiple testing correction to use (values: Bf, H-BF (by default), Benj-Hoch)
    - alpha : thresold of significance level
return :
    - input df with some columns added with the corrected p-values
'''
def pvalue_correction(df, correction = 'Benjamini-Hochberg', alpha = 0.05):

    ech_size = df.shape[0]
    updated = df.copy()
    
    if correction == 'Bonferroni':
        
        updated['alpha corrected'] = alpha / ech_size
        updated['H0 rejected corrected'] = updated['pvalue'] < updated['alpha corrected']
    
    elif correction == 'Holm-Bonferroni':
        
        updated_sorted = updated.sort_values('pvalue', ascending=True)
        updated_sorted['Rank'] = np.arange(ech_size, 0, -1)
        updated_sorted['alpha corrected'] = updated['alpha'] / updated_sorted['Rank']
        updated_sorted['H0 rejected corrected'] = updated_sorted['pvalue'] < updated_sorted['alpha corrected']
    
        updated = updated_sorted.reindex(updated.index)
        
    elif correction == 'Benjamini-Hochberg':
        
        updated_sorted = updated.sort_values('pvalue', ascending=True)
        updated_sorted['Rank'] = np.arange(1, ech_size + 1)
        updated_sorted['alpha corrected'] = updated_sorted['Rank'] / ech_size * alpha
        updated_sorted['H0 rejected corrected'] = updated_sorted['pvalue'] < updated_sorted['alpha corrected']
    
        updated = updated_sorted.reindex(updated.index)
    
    elif correction == 'FDR':
        
        Rejects, AdjustedPValues = statsmodels.stats.multitest.fdrcorrection(df['pvalue'], alpha=alpha, method='indep', is_sorted=False)
        updated = pd.concat([updated, pd.DataFrame({'pvalue corrected': AdjustedPValues, 'H0 rejected corrected': Rejects}, index=df.index)], axis=1)
        
    else:
        print("Correction not valid")
        updated = df
        
    return updated











'''
Plot the p-values for each variable
input :
    - df : output of function pvalue_correction
'''
def plot_pvalue(df):
    
    df = df.sort_values(by='pvalue')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30,6))
    fig.suptitle('p-value for each feature', fontsize=20, y=1.05)    
    
    # whole plot
    ax1.plot(df['pvalue'].values, color='b', linewidth=2, label='pvalue')
    ax1.plot(df['alpha'].values, color='r', linewidth=1, label='alpha')
    ax1.plot(df['alpha corrected'].values, color='g', linewidth=1, label='alpha corrected')
    ax1.set_xlabel('N° of the compound', fontsize=12)
    ax1.set_ylabel('pvalue', fontsize=12)
    ax1.legend(loc='upper left', prop={'size': 12})
    ax1.set_title('whole plot', fontsize=15)
    ax1.grid(linestyle='--', linewidth=1)
    
    
    # zoom on features under alpha
    subset1 = df.loc[:df[df['H0 rejected'] == False].index[2], :]
    
    ax2.plot(subset1['pvalue'].values, color='b', linewidth=2, label='pvalue')
    ax2.plot(subset1['alpha'].values, color='r', linewidth=2, label='alpha')
    ax2.plot(subset1['alpha corrected'].values, color='g', linewidth=2, label='alpha corrected')
    ax2.set_xlabel('N° of the compound', fontsize=12)
    ax2.set_ylabel('pvalue', fontsize=12)
    ax2.legend(loc='upper left', prop={'size': 12})
    ax2.set_title('zoom on features under alpha', fontsize=15)
    ax2.grid(linestyle='--', linewidth=1)
    
    
    # zoom on features under alphaCorrected
    subset2 = df.loc[:df[df['H0 rejected corrected'] == False].index[0], :]
    
    ax3.plot(subset2['pvalue'].values, color='b', linewidth=2, label='p-value')
    ax3.plot(subset2['alpha corrected'].values, color='g', linewidth=2, label='alpha corrected')
    ax3.set_xlabel('N° of the compound', fontsize=12)
    ax3.set_ylabel('pvalue', fontsize=12)
    ax3.ticklabel_format(useOffset=False, style='plain')
    ax3.legend(loc='upper left', prop={'size': 12})
    ax3.set_title('zoom on features under alpha corrected', fontsize=15)
    ax3.grid(linestyle='--', linewidth=1)
    
    

    
 