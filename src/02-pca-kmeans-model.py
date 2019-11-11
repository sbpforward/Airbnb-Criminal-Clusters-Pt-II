import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# KMeans
from sklearn.cluster import KMeans


def few_categorical_df(df):
    '''
    Create DataFrame consisting of fewer, select numerical values.
    
    Parameters
    ----------
    df: pandas.DataFrame  

    Returns
    ----------
    df: pandas.DataFrame 
    '''
    few_categorical_df = df[['price', 'minimum_nights', 'maximum_nights','review_scores_rating','host_loc_denver', 
                            'needs_license','room_type_Entire home/apt', 'room_type_Private room',
                            'room_type_Shared room', 'current_license', 'list_loc_denver']]
    return few_categorical_df

def y_target_vals(df):
    '''
    Create array consisting of target values.
    
    Parameters
    ----------
    df: pandas.DataFrame 

    Returns
    ----------
    arr: array
    '''
    y_target_vals = df.loc[:,['is_violating']].values
    return y_target_vals

def PCA_fewer_categoricals(df):
    '''
    Conduct PCA on few, select numerical values and plots explained variance.
    
    Parameters
    ----------
    df: pandas.DataFrame 
    
    Returns
    ----------
    df: pandas.DataFrame 
    plt: images
    '''
    features = ['price', 'minimum_nights', 'maximum_nights','review_scores_rating',
            'host_loc_denver', 'needs_license','room_type_Entire home/apt', 
            'room_type_Private room','room_type_Shared room', 'current_license', 
            'list_loc_denver']

    # Separating out the feature
    x = few_categorical_df.loc[:, features].values

    # Separating out the target
    y = df.loc[:,['is_violating']].values

    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=5)
    few_principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = few_principalComponents
                , columns = ['principal component 1', 'principal component 2',
                            'principal component 3', 'principal component 4',
                            'principal component 5'])
    few_pcaDf = pd.concat([principalDf, df[['is_violating']]], axis = 1)

    total_variance = np.sum(pca.explained_variance_)
    cum_variance = np.cumsum(pca.explained_variance_)
    prop_var_expl = cum_variance/total_variance

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(prop_var_expl, color='red', linewidth=2, label='Explained variance')
    ax.axhline(0.9, label='90% goal', linestyle='--', color="black", linewidth=1)
    ax.set_ylabel('Cumulative Explained Variance', fontsize=20)
    ax.set_xlabel('# of Principal Components', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.legend()
    # plt.show()
    plt.savefig('explained-variance.png')

    return (few_pcaDf,few_principalComponents)

def plot_PCA(df, title):
    '''
    Plot PCA visuals.
    
    Parameters
    ----------
    df: pd.DataFrame

    Returns
    ----------
    plt: image 
    '''
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 20)
    ax.set_ylabel('Principal Component 2', fontsize = 20)
    ax.set_title(f'{title}', fontsize = 20)
    targets = [0,1]
    colors = ['r', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = df['is_violating'] == target
        ax.scatter(df.loc[indicesToKeep, 'principal component 1']
                , df.loc[indicesToKeep, 'principal component 2']
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.savefig(f'PCA_{title}.png')
    # plt.show()

def plot_KMeans(few_principalComponents):
    '''
    Enter PCA component values into KMeans model. Plots scatter image of the model and
    cluster count by target.
    
    Parameters
    ----------
    df: pd.DataFrame

    Returns
    ----------
    plt: image
    '''
    kmeans = KMeans(n_clusters=3)
    X_clustered_num = kmeans.fit_predict(few_principalComponents)

    # Plot the scatter diagram with centroids
    plt.figure(figsize = (7,7))
    plt.scatter(few_principalComponents[:,0], few_principalComponents[:,1], c=kmeans.labels_, cmap='rainbow', alpha=0.2)
    plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
    plt.title('KMeans Clusters (3)', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('kmeans-clusters.png')
    # plt.show()

    # Plot cluster by target
    plt.figure(figsize=(10,5))
    fewkmeans = kmeans.labels_
    kmeans_y = y_target_vals
    kmeans_y = kmeans_y.reshape(4601)
    fewKMeansDF = pd.DataFrame({'kmeans_cluster':fewkmeans, 'target':kmeans_y})
    sns.barplot(x='kmeans_cluster',y='target',data=fewKMeansDF)
    plt.title('Clusters by Target', fontsize=20)
    plt.ylabel('% Violation',fontsize=20)
    plt.xlabel('KMeans Cluster',fontsize=20)
    plt.xticks(fontsize=19)
    plt.tight_layout()
    plt.savefig('clusters-by-target.png')
    # plt.show()
    return fewKMeansDF

def save(df):
    df.to_pickle('../data/pickled_kmeans_df')
    return df

if __name__ == '__main__':
    df = pd.read_pickle('../data/pickled_listings_df')

    few_categorical_df = few_categorical_df(df)
    y_target_vals = y_target_vals(df)

    few_categorical_df, few_principalComponents = PCA_fewer_categoricals(df)
    plot_PCA(few_categorical_df,'Fewer Categorical Features')

    fewKMeansDF = plot_KMeans(few_principalComponents)

    save(fewKMeansDF)