import numpy as np 
import pandas as pd 

def merge(df1,df2):
    df = df1.join(df2)    
    return df

def flask_df(df):

    flask_df = df[['kmeans_cluster','listing_url', 'list_loc_denver', 'host_id',
                    'host_loc_denver','host_url','needs_license','current_license', 
                    'minimum_nights', 'maximum_nights']]
    return flask_df

def sort(df):
    flask_df = df.sort_values(by=['kmeans_cluster', 'host_id', 'current_license'])
    return  flask_df

def save(df):
    df.to_pickle('../data/flask_df_pickle')
    return df

def save_json(df):
    df.to_json('../data/flask_df.json')
    return df


if __name__ == '__main__':
    df_master = pd.read_pickle('../data/pickled_listings_df')
    df_kmeans = pd.read_pickle('../data/pickled_kmeans_df') 
    df = merge(df_master,df_kmeans)
    
    flask_df = flask_df(df)
    flask_df = sort(flask_df)
    save(flask_df)
    save_json(flask_df)

