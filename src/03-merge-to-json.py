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

def convert_cols_yes_no(df):
    flask_df['list_loc_denver'] = df['list_loc_denver'].map(lambda x: 'Yes' if x == 1.0 else 'No' )
    flask_df['host_loc_denver'] = df['host_loc_denver'].map(lambda x: 'Yes' if x == 1.0 else 'No' )
    flask_df['needs_license'] = df['needs_license'].map(lambda x: 'Yes' if x == 1.0 else 'No' )
    flask_df['current_license'] = df['current_license'].map(lambda x: 'Yes' if x == 1.0 else 'No' )
    return flask_df

def likelihood(df):
    # flask_df.kmeans_cluster.replace(['Very', 'Somewhat', 'Not'], [0, 1, 2], inplace=True)
    # flask_df.kmeans_cluster.replace(to_replace=dict(Very=0, Somewhat=1, Not=2), inplace=True)
    # flask_df['kmeans_cluster'].map({0:'Very', 1:'Somewhat', 2:'Not'})

    flask_df['kmeans_cluster'] = flask_df['kmeans_cluster'].replace(regex=0, value='Very')
    flask_df['kmeans_cluster'] = flask_df['kmeans_cluster'].replace(regex=1, value='Somewhat')
    flask_df['kmeans_cluster'] = flask_df['kmeans_cluster'].replace(regex=2, value='Not')

    return flask_df

def sort(df):
    flask_df = df.sort_values(by=['kmeans_cluster', 'host_id', 'current_license'])
    return  flask_df

def rename_cols(df):
    flask_df.columns = ["Likelihood", "Listing URL", "Listing in Denver", "Host ID",
                        "Host in Denver", "Host URL", "Requires License", "Current License",
                        "Minimum Nights", "Maximum Nights"]
    return flask_df

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
    flask_df = convert_cols_yes_no(flask_df)
    # flask_df = likelihood(flask_df)
    flask_df = sort(flask_df)
    flask_df = rename_cols(flask_df)

    save(flask_df)
    save_json(flask_df)

