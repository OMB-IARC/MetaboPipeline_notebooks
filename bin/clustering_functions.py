
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from random_functions import *




############################################################################################################################################
################################################################# k-means ##################################################################
############################################################################################################################################
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

def perform_kmeans(X, n_clusters=2, target=None, col_prefix=''):
    
    X_ = X.copy()
    
    # Perform k-means
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, init='k-means++', random_state=0)
    kmeans.fit(X_)
    
    if X_.shape[1] > 2:
        print('Inputed dataframe has more than 2 dimensions so we cannot plot results of kmeans clustering.\nPlease use two-dimensional dataframe if you want to see the results plotted.')
        return kmeans
    
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = (X_.max() - X_.min()).max() / 1000
    #h = 0.01  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = X_.iloc[:, 0].min() - 1, X_.iloc[:, 0].max() + 1
    y_min, y_max = X_.iloc[:, 1].min() - 1, X_.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    # Set figure
    plt.figure(figsize=(12, 8))
    
    
    # Prepare cmap to fill with chosen colors
    #cmap = plt.get_cmap('jet')
    #cmap = plt.get_cmap('Pastel2')
    # Set3
    
    # Prepare cmap to fill with chosen colors
    #cmap = plt.get_cmap('jet')
    hex_list = ['#0091ad', '#3fcdda', '#83f9f8', '#d6f6eb', '#fdf1d2', '#f8eaad', '#faaaae', '#d16f6f']
    cmap=get_continuous_cmap(hex_list)
    bounds = np.arange(0, n_clusters)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    
    # Fill the areas with selected colors
    plt.imshow(
        Z,
        interpolation='nearest',
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=cmap,
        aspect='auto',
        origin='lower',
    )
    
    
    # Plot points with their 2 PCA components
    # and color based on sample group
    points = X_.copy()
    points.columns = ['x', 'y']
    palette = 'Set2'
    if not isinstance(target, type(None)):
        points['target'] = target.values
        points['target'] = points['target'].fillna('NaN')
        sns.scatterplot(x='x',
                        y='y',
                        data=points,
                        hue='target',
                        palette=palette,
                        s=50)
        plot_title = f'K-means clustering on peak table samples\nPoints colored by {target.name}\nCentroids are marked with white cross'
    else:
        sns.scatterplot(x='x',
                        y='y',
                        data=points,
                        palette=palette,
                        s=50)
        plot_title = f'K-means clustering on peak table samples\nCentroids are marked with white cross'

    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker='x',
        s=300,
        linewidths=3,
        color='w',
        zorder=1,
    )
    
    
    # Set figure characteristics
    plt.title(plot_title, fontsize=15)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.xlabel(f'{col_prefix} 1st component')
    plt.ylabel(f'{col_prefix} 2nd component')
    plt.show()
    

    print(f'Inertia of k-means model : {kmeans.inertia_ :.2f}')

    if not isinstance(target, type(None)) and (len(np.unique(target))==n_clusters):
        labels_true = target
        labels_true = LabelEncoder().fit_transform(labels_true)
        labels_predict = kmeans.labels_
        print(f'Adjusted rand index of k-means model : {metrics.adjusted_rand_score(labels_true, labels_predict) :.3f}')
        

    return kmeans
        
    
    
    
    
    
    
def plot_inertia(X, target=None, max_clusters=None, log=False):
    
    X_ = X.copy()
    
    if isinstance(max_clusters, type(None)):
        n_clusters = X_.shape[0]
    else:
        n_clusters = max_clusters
            
        
    inertia = []
    
    #if not isinstance(target, type(None)):
    #    rand_score = []
    #    labels_true = target.values
    #    labels_true = LabelEncoder().fit_transform(labels_true)

        
    for i in range(1, n_clusters):

        kmeans = KMeans(n_clusters=i, n_init=1, init='k-means++', random_state=0).fit(X_)
        
        if log:
            curr_inertia = np.log(kmeans.inertia_)
        else:
            curr_inertia = kmeans.inertia_
        
        inertia.append(curr_inertia)

        #labels_predict = kmeans.labels_
        
        #if not isinstance(target, type(None)):
        #    rand_score.append(metrics.adjusted_rand_score(labels_predict, labels_true))


    #print(f'Maximum rand score is {max(rand_score):.5f}, obtained with {rand_score.index(max(rand_score))} clusters\n')


    plt.figure(figsize=(10,6))
    plt.scatter(np.arange(1, n_clusters), inertia)
    plt.plot(np.arange(1, n_clusters), inertia)
    log_ = '(log value)' if log else ''
    plt.title(f'K-means inertia {log_} depending on number of clusters')
    plt.show()

    #if not isinstance(target, type(None)):
    #    plt.figure(figsize=(10,6))
    #    plt.scatter(np.arange(1, n_clusters), rand_score)
    #    plt.plot(np.arange(1, n_clusters), rand_score)
    #    plt.title('K-means rand score depending on number of clusters')
    #    plt.show()  
    
    
    

    
    

############################################################################################################################################
########################################################### Affinity propagation ###########################################################
############################################################################################################################################
from sklearn.cluster import AffinityPropagation
from itertools import cycle


def perform_affinity_propagation(X, target=None):
    
    X_ = X.copy()
    af = AffinityPropagation(random_state=0).fit(X_)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)

    if not isinstance(target, type(None)):
        labels_true = target
        labels_true = LabelEncoder().fit_transform(labels_true)
        
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
        print(
            "Adjusted Mutual Information: %0.3f"
            % metrics.adjusted_mutual_info_score(labels_true, labels)
        )
        print(
            "Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(X_, labels, metric="sqeuclidean")
        )
    
    
    
    plt.figure(figsize=(12, 8))

    X_ = np.array(X_)

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        cluster_center = X_[cluster_centers_indices[k]]
        plt.plot(X_[class_members, 0], X_[class_members, 1], col + ".")
        plt.plot(
            cluster_center[0],
            cluster_center[1],
            'o',
            markerfacecolor=col,
            markeredgecolor='k',
            markersize=14,
        )
        for x in X_[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

    plt.title(f'Estimated number of clusters : {n_clusters_}', fontsize=18)
    plt.show()
    
    return af
    
    
    
    
    
    
############################################################################################################################################
################################################################ Mean shift ################################################################
############################################################################################################################################
from sklearn.cluster import MeanShift, estimate_bandwidth


def perform_mean_shift(X):
    
    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(X, quantile=0.2)

    X_ = X.copy()
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X_)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print(f'Estimated number of clusters : {n_clusters_}')
    
    
    
    plt.figure(figsize=(12, 8))

    X_ = np.array(X_)

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X_[my_members, 0], X_[my_members, 1], col + ".")
        plt.plot(
            cluster_center[0],
            cluster_center[1],
            'o',
            markerfacecolor=col,
            markeredgecolor='k',
            markersize=14,
        )
    plt.title(f'Estimated number of clusters : {n_clusters_}', fontsize=18)
    plt.show()
    
    return ms











############################################################################################################################################
######################################################### Hierarchical clustering ##########################################################
############################################################################################################################################
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering


def plot_dendrogram(X, hline=None):
    
    plt.figure(figsize=(30,10))

    dendrogram = shc.dendrogram(shc.linkage(X, 'ward'),
                                orientation='top',
                                distance_sort='descending',
                                show_leaf_counts=True)

    if not isinstance(hline, type(None)):
        plt.axhline(y=hline, c='k')

    plt.title('Dendrogram', fontsize=18)
    plt.xticks(fontsize=9)
    plt.show()
    
    
    
    
def perform_hierarchical_clustering(X, n_clusters, target=None):

    X_ = X.copy()

    hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    hc.fit(X_)
    
    if not isinstance(target, type(None)):
        X_ = pd.concat([X_, target], axis=1)

    plt.figure(figsize=(12, 8))
    
    if not isinstance(target, type(None)):
        #plt.scatter(x=X_.iloc[:,0], y=X_.iloc[:,1], c=hc.labels_, cmap='rainbow')
        sns.scatterplot(x=X_.iloc[:,0], y=X_.iloc[:,1], hue=hc.labels_,
                        s=100, style=target, palette='rainbow')
    else:
        #plt.scatter(x=X_.iloc[:,0], y=X_.iloc[:,1], c=hc.labels_, cmap='rainbow')
        sns.scatterplot(x=X_.iloc[:,0], y=X_.iloc[:,1], hue=hc.labels_,
                        s=100, palette='rainbow')
    
    plt.show()
    
    return hc
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
############################################################################################################################################
################################################################# DBSCAN ###################################################################
############################################################################################################################################
from sklearn.cluster import DBSCAN


def perform_DBSCAN(X, target=None, eps=0.5, min_samples=5):
    
    X_ = X.copy()

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters_ == 0:
        return 'DBSCAN detects all points as noise/outliers'
    
    n_noise_ = list(labels).count(-1)
    

    if not isinstance(target, type(None)):
        labels_true = target
        labels_true = LabelEncoder().fit_transform(labels_true)


        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
        print(
            "Adjusted Mutual Information: %0.3f"
            % metrics.adjusted_mutual_info_score(labels_true, labels)
        )
        print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
        
    
    plt.figure(figsize=(12, 8))

    X_ = np.array(X_)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X_[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = X_[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f'Estimated number of clusters : {n_clusters_}', fontsize=18)
    plt.show()
    
    return db





def get_eps_optimal_value(df):

    import math
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=5)

    nbrs = neigh.fit(df)
    distances, indices = nbrs.kneighbors(df)

    # Plotting K-distance Graph
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.figure(figsize=(12, 8))
    plt.plot(distances)
    plt.title('K-distance Graph',fontsize=20)
    plt.xlabel('Data Points sorted by distance',fontsize=14)
    plt.ylabel('Epsilon',fontsize=14)
    plt.show()

    opt_eps = math.ceil(max(distances) * 10) / 10
    
    print(f'Optimal value to pass DBSCAN model is : {opt_eps}')
    
    return opt_eps






############################################################################################################################################
################################################################# OPTICS ###################################################################
############################################################################################################################################
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec


def perform_OPTICS(X, list_eps=[0.5, 1, 2], min_samples=5):
    
    X_ = X.copy()

    clust = OPTICS(min_samples=min_samples, cluster_method='dbscan')

    # Run the fit
    clust.fit(X_)

    labels_050 = cluster_optics_dbscan(
        reachability=clust.reachability_,
        core_distances=clust.core_distances_,
        ordering=clust.ordering_,
        eps=list_eps[0],
    )
    labels_100 = cluster_optics_dbscan(
        reachability=clust.reachability_,
        core_distances=clust.core_distances_,
        ordering=clust.ordering_,
        eps=list_eps[1],
    )
    labels_200 = cluster_optics_dbscan(
        reachability=clust.reachability_,
        core_distances=clust.core_distances_,
        ordering=clust.ordering_,
        eps=list_eps[2],
    )

    space = np.arange(len(X_))
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]
    
    
    
    
    plt.figure(figsize=(20, 10))

    X_ = np.array(X_)

    G = gridspec.GridSpec(2, 4)
    ax1 = plt.subplot(G[0, :])
    ax2 = plt.subplot(G[1, 0])
    ax3 = plt.subplot(G[1, 1])
    ax4 = plt.subplot(G[1, 2])
    ax5 = plt.subplot(G[1, 3])

    # Reachability plot
    colors = ["g.", "r.", "b.", "y.", "c."]
    for klass, color in zip(range(0, 5), colors):
        Xk = space[labels == klass]
        Rk = reachability[labels == klass]
        ax1.plot(Xk, Rk, color, alpha=0.3)
    ax1.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3)
    ax1.plot(space, np.full_like(space, list_eps[2], dtype=float), "k-", alpha=0.5)
    ax1.plot(space, np.full_like(space, list_eps[1], dtype=float), "k--", alpha=0.5)
    ax1.plot(space, np.full_like(space, list_eps[0], dtype=float), "k-.", alpha=0.5)
    ax1.set_ylabel("Reachability (epsilon distance)")
    ax1.set_title("Reachability Plot")

    # OPTICS
    colors = ["g.", "r.", "b.", "y.", "c."]
    for klass, color in zip(range(0, 5), colors):
        Xk = X_[clust.labels_ == klass]
        ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    ax2.plot(X_[clust.labels_ == -1, 0], X_[clust.labels_ == -1, 1], "k+", alpha=0.1)
    ax2.set_title("Automatic Clustering\nOPTICS")

    # DBSCAN at list_eps[0]
    colors = ["g", "greenyellow", "olive", "r", "b", "c"]
    for klass, color in zip(range(0, 6), colors):
        Xk = X_[labels_050 == klass]
        ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker=".")
    ax3.plot(X_[labels_050 == -1, 0], X_[labels_050 == -1, 1], "k+", alpha=0.1)
    ax3.set_title(f"Clustering at {list_eps[0]} epsilon cut\nDBSCAN")

    # DBSCAN at list_eps[1]
    colors = ["g.", "m.", "y.", "c."]
    for klass, color in zip(range(0, 4), colors):
        Xk = X_[labels_100 == klass]
        ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    ax4.plot(X_[labels_100 == -1, 0], X_[labels_100 == -1, 1], "k+", alpha=0.1)
    ax4.set_title(f"Clustering at {list_eps[1]} epsilon cut\nDBSCAN")

    # DBSCAN at list_eps[2]
    colors = ["g.", "m.", "y.", "c."]
    for klass, color in zip(range(0, 4), colors):
        Xk = X_[labels_200 == klass]
        ax5.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    ax5.plot(X_[labels_200 == -1, 0], X_[labels_200 == -1, 1], "k+", alpha=0.1)
    ax5.set_title(f"Clustering at {list_eps[2]} epsilon cut\nDBSCAN")

    plt.tight_layout()
    plt.show()
    
    
    return clust