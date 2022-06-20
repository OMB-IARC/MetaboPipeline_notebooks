'''
Perform PCA on a peak table

input :
    - X : peakTable with only variable columns, no metadata
    - n_components : number of components for the PCA
    - part_explained_variance : part of initial explained variance we want to keep in the reduced dimensions
    - concat_metadata (default=False) : if set to True, the returned dataframe df is concatenated with metadata passed as argument
    - metadata (default=None) : required if concat_metadata is set to True, corresponds to metadata dataframe
return :
    - df : dataframe with PCA components as columns and samples as rows
    - explained_variance : dataframe of % explained variance and cumulative sum for each component
plot :
    - lineplot of cumulative sum of explained variance
        x-axis : number of components
        y-axis : cumulative sum
'''
def perform_PCA(X, n_components=None, part_explained_variance=None, concat_metadata=False, metadata=None):
    
    
    # Apply StandardScaler() if not already done
    if (X.mean().mean() < 0.01) & (0.99 < X.std().mean() < 1.01):
        X_std = X
    else:
        X_std = StandardScaler().fit_transform(X)

    
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
    principal_components = PCA_model.fit_transform(X_std)
    

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
    fig, ax = plt.subplots()
    xi = np.arange(1, explained_variance.shape[0] + 1, step=1)
    y = np.cumsum(PCA_model.explained_variance_ratio_)

    plt.ylim(0, 1.05)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')
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
    
    return subset, explained_variance
















def perform_tSNE(X, metadata, n_components=2, targets_plot=[], concat_metadata=False):
    
    # Apply StandardScaler() if not already done
    if (X.mean().mean() < 0.01) & (0.99 < X.std().mean() < 1.01):
        X_std = X
    else:
        X_std = StandardScaler().fit_transform(X)
        
        
    # Assertion
    assert (n_components==2 or n_components==3),\
        '<n_components> has to be either 2 or 3'
        
    
    # Perform t-SNE
    start_time = time.time()
    X_tSNE = TSNE(n_components=n_components, learning_rate='auto', init='random', random_state=0).fit_transform(X_std)
    print(f'Time to compute t-SNE (input dataframe of shape {X_std.shape}) : {time.time() - start_time :.2f} seconds\n')
    
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
        if targets_plot==[]:
            plt.figure(figsize=(12,8))
            sns.scatterplot(x=0, y=1, data=pd.DataFrame(X_tSNE),
                            s=50, legend='full', palette='deep');
            plt.xlabel('t-SNE (1st dimension)', fontsize=14)
            plt.ylabel('t-SNE (2nd dimension)', fontsize=14)
            plt.title(f't-SNE first two components', fontsize=18)
            plt.show()
            print(100 * '-')
        else:
            for target in targets_plot:

                plt.figure(figsize=(12,8))
                sns.scatterplot(x=0, y=1, data=pd.DataFrame(X_tSNE), hue=metadata[target],
                                s=50, legend='full', palette='deep');
                plt.xlabel('t-SNE (1st dimension)', fontsize=14)
                plt.ylabel('t-SNE (2nd dimension)', fontsize=14)
                plt.title(f't-SNE first two components, colored by {target}', fontsize=18)
                plt.show()
                print(100 * '-')
            
    elif n_components == 3:
        
        print(100 * '-', '\n')
        if targets_plot==[]:
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
            for target in targets_plot:

                fig = plt.figure(figsize=(15,10))
                ax = fig.add_subplot(111, projection = '3d')

                le = LabelEncoder()
                le.fit(metadata[target])
                encoded_labels = le.transform(metadata[target])
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
                ax.set_title(f't-SNE first 3 components, colored by {target}', fontsize=18)

                legend1 = ax.legend(*scatter.legend_elements(), loc='upper right', title=target)
                ax.add_artist(legend1)

                plt.show()

            print(100 * '-', '\n')
            
    return df_tSNE




















def perform_UMAP(X, metadata, n_components=2, targets_plot=[], concat_metadata=False):
    
    # Apply StandardScaler() if not already done
    if (X.mean().mean() < 0.01) & (0.99 < X.std().mean() < 1.01):
        X_std = X
    else:
        X_std = StandardScaler().fit_transform(X)
        
        
    # Assertion
    assert (n_components==2 or n_components==3),\
        '<n_components> has to be either 2 or 3'
        
    
    # Perform UMAP
    start_time = time.time()
    X_UMAP = UMAP(n_components=n_components, init='random', random_state=0).fit_transform(X_std)
    print(f'Time to compute UMAP (input dataframe of shape {X_std.shape}) : {time.time() - start_time :.2f} seconds\n')

    
    
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
        if targets_plot==[]:
            plt.figure(figsize=(12,8))
            sns.scatterplot(x=0, y=1, data=pd.DataFrame(X_UMAP),
                            s=50, legend='full', palette='deep');
            plt.xlabel('UMAP (1st dimension)', fontsize=14)
            plt.ylabel('UMAP (2nd dimension)', fontsize=14)
            plt.title(f'UMAP first two components', fontsize=18)
            plt.show()
            print(100 * '-')
        else:
            for target in targets_plot:

                plt.figure(figsize=(12,8))
                sns.scatterplot(x=0, y=1, data=pd.DataFrame(X_UMAP), hue=metadata[target],
                                s=50, legend='full', palette='deep');
                plt.xlabel('UMAP (1st dimension)', fontsize=14)
                plt.ylabel('UMAP (2nd dimension)', fontsize=14)
                plt.title(f'UMAP first two components, colored by {target}', fontsize=18)
                plt.show()
                print(100 * '-')
            
    elif n_components == 3:
        
        print(100 * '-', '\n')
        if targets_plot==[]:
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
            for target in targets_plot:

                fig = plt.figure(figsize=(15,10))
                ax = fig.add_subplot(111, projection = '3d')

                le = LabelEncoder()
                le.fit(metadata[target])
                encoded_labels = le.transform(metadata[target])
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
                ax.set_title(f'UMAP first 3 components, colored by {target}', fontsize=18)

                legend1 = ax.legend(*scatter.legend_elements(), loc='upper right', title=target)
                ax.add_artist(legend1)

                plt.show()

            print(100 * '-', '\n')
            
    return df_UMAP