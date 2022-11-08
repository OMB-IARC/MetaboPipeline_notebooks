
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


import time






############################################################################################################################################
################################################################### PCA ####################################################################
############################################################################################################################################
from sklearn.decomposition import PCA

'''
Perform PCA on a peak table.


input :
    - X : peakTable with only variable columns, no metadata
    - n_components : number of components for the PCA
    - part_explained_variance : part of initial explained variance we want to keep in the reduced dimensions
    - concat_metadata (default=False) : if set to True, the returned dataframe df is concatenated with metadata passed as argument
    - metadata (default=None) : required if concat_metadata is set to True, corresponds to metadata dataframe
    - scale_before_PCA (default=False) : whether to apply standard scaler to the input peak table X before performing PCA
return :
    - df : dataframe with PCA components as columns and samples as rows
    - explained_variance : dataframe of % explained variance and cumulative sum for each component
plot :
    - lineplot of cumulative sum of explained variance
        x-axis : number of components
        y-axis : cumulative sum
'''
def perform_PCA(X, n_components=None, part_explained_variance=None, concat_metadata=False, metadata=None, scale_before_PCA=False):
    
    
    # Apply StandardScaler() if not already done
    #if (X.mean().mean() < 0.01) & (0.99 < X.std().mean() < 1.01):
    #    X_std = X
    #else:
    #    X_std = StandardScaler().fit_transform(X)
    if scale_before_PCA:
        X = StandardScaler().fit_transform(X)

    
    # Assertion
    assert not (isinstance(n_components, type(None)) and isinstance(part_explained_variance, type(None))),\
        '<n_components> or <part_explained_variance> has to be defined'
    
    assert (isinstance(n_components, type(None)) or isinstance(part_explained_variance, type(None))),\
        '<n_components> and <part_explained_variance> cannot be both passed as argument'
    
    if concat_metadata:
        # assert metadata != None
        assert not isinstance(metadata, type(None)),\
            'If <merge_metadata> is set to True, <metadata> has to be provided'
    
        
    # Perform PCA analysis
    if not isinstance(n_components, type(None)):
        PCA_model = PCA(n_components=n_components)
    else:
        PCA_model = PCA(n_components=part_explained_variance)
    #principal_components = PCA_model.fit_transform(X_std)
    principal_components = PCA_model.fit_transform(X)
    

    # Prepare column names
    col_names = ['PC' + str(i+1) for i in range(principal_components.shape[1])]
    index = X.index.values
    
    
    # Prepare df dataframe to return
    # Concat metadata if set to True
    if concat_metadata:
        subset = pd.concat([metadata, pd.DataFrame(principal_components, columns = col_names, index = index)], axis=1)
    else:
        subset = pd.DataFrame(principal_components, columns = col_names, index = index)
    
    # Get percent of explained variance
    explained_variance = PCA_model.explained_variance_ratio_
    explained_variance_format = [round(num, 2) for num in explained_variance * 100]

    # Compute cumulative sum of explained variance percents
    cum_sum = np.cumsum(explained_variance)
    cum_sum_format = [round(num, 2) for num in cum_sum * 100]
    
    # Prepare explained_variance dataframe to return
    explained_variance = pd.DataFrame(list(zip(explained_variance_format, cum_sum_format)), columns = [['Explained variance (%)', 'Cumulative sum']], index = col_names)
    
    
    # Plot cumulative sum of explained variance
    fig, ax = plt.subplots(figsize=(12,8))
    xi = np.arange(1, explained_variance.shape[0] + 1, step=1)
    y = np.cumsum(PCA_model.explained_variance_ratio_)

    plt.plot(xi, y, marker='o', linestyle='--', color='b')
    plt.ylim(0, 1.05)
    plt.text(x=PCA_model.n_components_, y=y[-1] + 0.03, s=f'PC{PCA_model.n_components_}\n({y[-1]*100:.2f}%)',
             color='b', fontsize=12)

    plt.xlabel('Number of Components', fontsize=14)
    plt.ylabel('Cumulative variance', fontsize=14)
    
    if not isinstance(n_components, type(None)):
        plt.title(f'Explained variance with {n_components} components in PCA model', fontsize=18)
    else:
        plt.title(f'The number of components needed to explain {part_explained_variance*100:.0f}% of total variance', fontsize=18)
        plt.axhline(y=part_explained_variance, color='r', linestyle='-')
        plt.text(x=0.5, y=part_explained_variance - 0.05, s=f'{part_explained_variance}% cut-off threshold', color='r', fontsize=12)

    plt.show()
    
    
    # Print information about number of dimensions before and after PCA
    print(f'Initial number of dimension : {X.shape[1]}')
    print(f'Final number of dimension : {explained_variance.shape[0]}')
    
    
    return subset, explained_variance







############################################################################################################################################
################################################################# k-best ###################################################################
############################################################################################################################################

from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectPercentile



def select_best_features(X, y, k=None, alpha=None, concat_metadata=False, metadata=None):
    
    if k==None:
        k = X.shape[1]
    if alpha==None:
        alpha = 2
    
    # Assertions
    if concat_metadata:
        # assert metadata != None
        assert not isinstance(metadata, type(None)),\
            'If <merge_metadata> is set to True, <metadata> has to be provided'
    
    # Compute the ANOVA F-value
    f_statistic, p_values = f_classif(X, y)

    # Create dataframe with f_score and p_value for each feature
    df_f_classif = pd.DataFrame(list(zip(X.columns.values, f_statistic, p_values)), columns=['features', 'F-value', 'p-value'])\
    .sort_values(by ='F-value', ascending=False)
    #df_f_classif = df_f_classif.round({'F-value': 3, 'p-value': 4})
    
    # Subset df_f_classif based on k and p_value
    df_f_classif = df_f_classif.iloc[:k,:]
    df_f_classif = df_f_classif[df_f_classif['p-value'] < alpha]
    
    # Subset X with selected features
    subset = X[df_f_classif['features'].values]
    
    # Concat metadata if set to True
    if concat_metadata:
        subset = pd.concat([metadata, subset], axis=1)
        
    print(f'Initial number of dimension : {X.shape[1]}')
    print(f'Final number of dimension : {df_f_classif.shape[0]}')

    
    return subset, df_f_classif






def select_percentile_features(X, y, percentile=100, concat_metadata=False, metadata=None):

    # Compute the ANOVA F-value
    f_statistic, p_values = f_classif(X, y)

    # Create dataframe with f_score and p_value for each feature
    df_f_classif = pd.DataFrame(list(zip(X.columns.values, f_statistic, p_values)), columns=['features', 'F-value', 'p-value'])\
    .sort_values(by ='F-value', ascending=False)
    #df_f_classif = df_f_classif.round({'F-value': 3, 'p-value': 4})
    
    # Subset X with selected features
    selector = SelectPercentile(f_classif, percentile=percentile)
    X_new = selector.fit_transform(X, y)
    columns = X.columns.values
    subset_columns = selector.get_support()
    kept_features = columns[subset_columns]
    subset = X[kept_features]
    
    # Subset df_f_classif based on ketp features
    df_f_classif = df_f_classif[df_f_classif['features'].isin(kept_features)]
    
    # Concat metadata if set to True
    if concat_metadata:
        subset = pd.concat([metadata, subset], axis=1)
        
    print(f'Initial number of dimension : {X.shape[1]}')
    print(f'Final number of dimension : {df_f_classif.shape[0]}')

    
    return subset, df_f_classif






def plot_pvalue_fvalue(feature_scores):
    
    if len(feature_scores) > 75:
        print('Dataframe has too much features to display plots (75 features is the maximum length accepted).')

    else:
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(data=feature_scores, x='features', y='p-value');
        plt.xticks(rotation=90)
        ax.set_title('p-value for each selected feature', fontsize=16)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.show()

        plt.figure(figsize=(12, 8))
        ax = sns.barplot(data=feature_scores, x='features', y='F-value');
        plt.xticks(rotation=90)
        ax.set_title('ANOVA F-value for each selected feature', fontsize=16)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.show()






############################################################################################################################################
################################################################## t-SNE ###################################################################
############################################################################################################################################
from sklearn.manifold import TSNE


def perform_tSNE(X, n_components=2, concat_metadata=False, metadata=None, scale_before_tSNE=False, target=None):
    
    # Apply StandardScaler() if not already done
    #if (X.mean().mean() < 0.01) & (0.99 < X.std().mean() < 1.01):
    #    X_std = X
    #else:
    #    X_std = StandardScaler().fit_transform(X)
    if scale_before_tSNE:
        X = StandardScaler().fit_transform(X)
        
        
    # Assertion
    assert (n_components==2 or n_components==3),\
        '<n_components> has to be either 2 or 3'
        
    
    # Perform t-SNE
    start_time = time.time()
    X_tSNE = TSNE(n_components=n_components, learning_rate='auto', init='random', random_state=0).fit_transform(X)
    print(f'Time to compute t-SNE (input dataframe of shape {X.shape}) : {time.time() - start_time :.2f} seconds\n')
    
    # Prepare column names
    col_names = ['tSNE' + str(i+1) for i in range(n_components)]
    index = X.index.values
    
    
    # Prepare df dataframe to return
    # Concat metadata if set to True
    if concat_metadata:
        df_tSNE = pd.concat([metadata, pd.DataFrame(X_tSNE, columns=col_names, index=index)], axis=1)
    else:
        df_tSNE = pd.DataFrame(X_tSNE, columns=col_names, index=index)

    
    # Plot t-SNE dimensions
    if n_components == 2:
        
        print(100 * '-')
        if isinstance(target, type(None)):
            plt.figure(figsize=(12,8))
            sns.scatterplot(x=0, y=1, data=pd.DataFrame(X_tSNE),
                            s=50, legend='full', palette='deep');
            plt.xlabel('t-SNE (1st dimension)', fontsize=14)
            plt.ylabel('t-SNE (2nd dimension)', fontsize=14)
            plt.title(f't-SNE first two components', fontsize=18)
            plt.show()
            print(100 * '-')
        else:
            plt.figure(figsize=(12,8))
            sns.scatterplot(x=0, y=1, data=pd.DataFrame(X_tSNE), hue=target,
                            s=50, legend='full', palette='deep');
            plt.xlabel('t-SNE (1st dimension)', fontsize=14)
            plt.ylabel('t-SNE (2nd dimension)', fontsize=14)
            plt.title(f't-SNE first two components, colored by {target.name}', fontsize=18)
            plt.show()
            print(100 * '-')
            
    elif n_components == 3:
        
        print(100 * '-', '\n')
        if isinstance(target, type(None)):
            fig = plt.figure(figsize=(15,10))
            ax = fig.add_subplot(111, projection = '3d')

            scatter = ax.scatter(
                xs=X_tSNE[:,0], 
                ys=X_tSNE[:,1], 
                zs=X_tSNE[:,2],
                cmap='rainbow',
                s=50,
                alpha=1
            )
            ax.set_xlabel('t-SNE (1st dimension)')
            ax.set_ylabel('t-SNE (2nd dimension)')
            ax.set_zlabel('t-SNE (3rd dimension)')
            ax.set_title(f't-SNE first 3 components', fontsize=18)
            
            plt.show()
            
        else :
            fig = plt.figure(figsize=(15,10))
            ax = fig.add_subplot(111, projection = '3d')

            le = LabelEncoder()
            le.fit(target)
            encoded_labels = le.transform(target)
            le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f'Corresponding labels for each class in legend : {le_name_mapping}\n')

            scatter = ax.scatter(
                xs=X_tSNE[:,0], 
                ys=X_tSNE[:,1], 
                zs=X_tSNE[:,2],
                c=encoded_labels,
                cmap='rainbow',
                s=50,
                alpha=1
            )
            ax.set_xlabel('t-SNE (1st dimension)')
            ax.set_ylabel('t-SNE (2nd dimension)')
            ax.set_zlabel('t-SNE (3rd dimension)')
            ax.set_title(f't-SNE first 3 components, colored by {target.name}', fontsize=18)

            legend1 = ax.legend(*scatter.legend_elements(), loc='upper right', title=target.name)
            ax.add_artist(legend1)

            plt.show()

        print(100 * '-', '\n')
            
    return df_tSNE















############################################################################################################################################
################################################################## UMAP ###################################################################
############################################################################################################################################
from umap import UMAP


def perform_UMAP(X, n_components=2, concat_metadata=False, metadata=None, scale_before_UMAP=False, target=None):
    
    # Apply StandardScaler() if not already done
    #if (X.mean().mean() < 0.01) & (0.99 < X.std().mean() < 1.01):
    #    X_std = X
    #else:
    #    X_std = StandardScaler().fit_transform(X)
    if scale_before_UMAP:
        X = StandardScaler().fit_transform(X)
        
        
    # Assertion
    assert (n_components==2 or n_components==3),\
        '<n_components> has to be either 2 or 3'
        
    
    # Perform UMAP
    start_time = time.time()
    X_UMAP = UMAP(n_components=n_components, init='random', random_state=0).fit_transform(X)
    print(f'Time to compute UMAP (input dataframe of shape {X.shape}) : {time.time() - start_time :.2f} seconds\n')

    
    
    # Prepare column names
    col_names = ['UMAP' + str(i+1) for i in range(n_components)]
    index = X.index.values
    
    
    # Prepare df dataframe to return
    # Concat metadata if set to True
    if concat_metadata:
        df_UMAP = pd.concat([metadata, pd.DataFrame(X_UMAP, columns=col_names, index=index)], axis=1)
    else:
        df_UMAP = pd.DataFrame(X_UMAP, columns=col_names, index=index)

    
    # Plot t-SNE dimensions
    if n_components == 2:
        
        print(100 * '-')
        if isinstance(target, type(None)):
            plt.figure(figsize=(12,8))
            sns.scatterplot(x=0, y=1, data=pd.DataFrame(X_UMAP),
                            s=50, legend='full', palette='deep');
            plt.xlabel('UMAP (1st dimension)', fontsize=14)
            plt.ylabel('UMAP (2nd dimension)', fontsize=14)
            plt.title(f'UMAP first two components', fontsize=18)
            plt.show()
            print(100 * '-')
        else:
            plt.figure(figsize=(12,8))
            sns.scatterplot(x=0, y=1, data=pd.DataFrame(X_UMAP), hue=target,
                            s=50, legend='full', palette='deep');
            plt.xlabel('UMAP (1st dimension)', fontsize=14)
            plt.ylabel('UMAP (2nd dimension)', fontsize=14)
            plt.title(f'UMAP first two components, colored by {target.name}', fontsize=18)
            plt.show()
            print(100 * '-')
            
    elif n_components == 3:
        
        print(100 * '-', '\n')
        if isinstance(target, type(None)):
            fig = plt.figure(figsize=(15,10))
            ax = fig.add_subplot(111, projection = '3d')

            scatter = ax.scatter(
                xs=X_UMAP[:,0], 
                ys=X_UMAP[:,1], 
                zs=X_UMAP[:,2],
                cmap='rainbow',
                s=50,
                alpha=1
            )
            ax.set_xlabel('UMAP (1st dimension)')
            ax.set_ylabel('UMAP (2nd dimension)')
            ax.set_zlabel('UMAP (3rd dimension)')
            ax.set_title(f'UMAP first 3 components', fontsize=18)
            
            plt.show()
            
        else :
            fig = plt.figure(figsize=(15,10))
            ax = fig.add_subplot(111, projection = '3d')

            le = LabelEncoder()
            le.fit(target)
            encoded_labels = le.transform(target)
            le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f'Corresponding labels for each class in legend : {le_name_mapping}\n')

            scatter = ax.scatter(
                xs=X_UMAP[:,0], 
                ys=X_UMAP[:,1], 
                zs=X_UMAP[:,2],
                c=encoded_labels,
                cmap='rainbow',
                s=50,
                alpha=1
            )
            ax.set_xlabel('UMAP (1st dimension)')
            ax.set_ylabel('UMAP (2nd dimension)')
            ax.set_zlabel('UMAP (3rd dimension)')
            ax.set_title(f'UMAP first 3 components, colored by {target.name}', fontsize=18)

            legend1 = ax.legend(*scatter.legend_elements(), loc='upper right', title=target.name)
            ax.add_artist(legend1)

            plt.show()

        print(100 * '-', '\n')
            
    return df_UMAP