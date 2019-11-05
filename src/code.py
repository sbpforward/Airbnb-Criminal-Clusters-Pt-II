import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# KMeans
from sklearn.cluster import KMeans

def num_only_df(df):
    '''
    Create DataFrame consisting of only numerical values.
    
    Parameters
    ----------
    df: pandas.DataFrame 

    Returns
    ----------
    df: pandas.DataFrame 
    '''
    num_only_df = df[['price', 'weekly_price', 'monthly_price', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 
                'minimum_nights', 'maximum_nights', 'review_scores_rating', 
                'calculated_host_listings_count_entire_homes','calculated_host_listings_count_private_rooms', 
                'calculated_host_listings_count_shared_rooms', 'host_loc_denver', 'is_superhost', 'needs_license', 
                'in_top_10_neighbourhood', 'room_type_Entire home/apt', 'room_type_Private room',
                'room_type_Shared room', 'current_license', 'list_loc_denver']]
    return num_only_df

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

def PCA_all_numeric_categoricals(df):
    '''
    Conduct PCA on all numerical values.
    
    Parameters
    ----------
    df: pandas.DataFrame 

    Returns
    ----------
    df: pandas.DataFrame 
    '''
    features = ['price', 'weekly_price', 'monthly_price', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'minimum_nights',
    'maximum_nights', 'review_scores_rating', 'calculated_host_listings_count_entire_homes',
    'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms', 'host_loc_denver',
    'is_superhost', 'needs_license', 'in_top_10_neighbourhood', 'room_type_Entire home/apt', 'room_type_Private room',
    'room_type_Shared room', 'current_license', 'list_loc_denver']

    # Separating out the features
    x = num_only_df.loc[:, features].values

    # Separating out the target
    y = df.loc[:,['is_violating']].values

    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=5)
    num_only_principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = num_only_principalComponents
                , columns = ['principal component 1', 'principal component 2',
                            'principal component 3', 'principal component 4',
                            'principal component 5'])
    all_num_pcaDf = pd.concat([principalDf, df[['is_violating']]], axis = 1)
    return all_num_pcaDf

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
    ax.set_ylabel('cumulative prop. of explained variance')
    ax.set_xlabel('number of principal components')
    ax.legend()
    # plt.show()
    # plt.savefig('explained-variance.png')

    return few_pcaDf

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
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
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
    # plt.savefig(f'PCA_{title}.png')
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
    kmeans = KMeans(n_clusters=8)
    X_clustered_num = kmeans.fit_predict(few_principalComponents)

    # Plot the scatter diagram
    plt.figure(figsize = (7,7))
    plt.scatter(few_principalComponents[:,0],few_principalComponents[:,1], 
                c=kmeans.labels_,cmap='rainbow', alpha=0.5) 
    plt.title('KMeans Clusters (8)')
    # plt.savefig('kmeans-clusters.png')
    # plt.show()

    # Plot cluster by target
    plt.figure(figsize=(10,5))
    fewkmeans = kmeans.labels_
    kmeans_y = y_target_vals
    kmeans_y = kmeans_y.reshape(4511)
    fewKMeansDF = pd.DataFrame({'kmeans_cluster':fewkmeans, 'target':kmeans_y})
    sns.barplot(x='kmeans_cluster',y='target',data=fewKMeansDF)
    plt.title('Clusters by Target')
    plt.savefig('clusters-by-target.png')
    # plt.show()

if __name__ == '__main__':
    df = pd.read_pickle('../data/pickled_listings_df')

    num_only_df = num_only_df(df)
    few_categorical_df = few_categorical_df(df)
    y_target_vals = y_target_vals(df)

    all_num_pcaDf = PCA_all_numeric_categoricals(df)
    plot_PCA(all_num_pcaDf,'All Features')

    few_categorical_df, few_principalComponents = PCA_fewer_categoricals(df)
    plot_PCA(few_categorical_df,'Fewer Categorical Features')

    plot_KMeans(few_principalComponents)