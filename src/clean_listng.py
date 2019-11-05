import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Clean():
    
    def __init__(self, df):
        self.df = df  

    def select_cols(self, columns_to_keep):
        '''
        Returns pandas DataFrame with desired columns.
        
        Parameters
        ----------
        df: pandas.DataFrame
            DataFrame produced by the listings.csv file.
        cols: list

        Returns
        ----------
        df: pandas.DataFrame
            DataFrame that consists only of the columns passed through.

        '''
        columns_to_drop = []
        for x in self.df.columns:
            if x not in columns_to_keep:
                columns_to_drop.append(x)
        self.df.drop(columns_to_drop, inplace=True, axis=1)

    def to_float(self, cols):
        '''
        Converts specifified column to float type.

        Parameters
        ----------
        df: pandas.DataFrame
            Passes the DataFrame that was recently updated to have the
            desired columns.

        cols: list
            List of strings of the column names that need to be 
            converted to float.
            =============
            price
            weekly_price
            monthly_price
            =============

        Returns
        ----------
        df: pandas.DataFrame
            Updated DataFrame that updates the 'price' column datatype from a
            string to a float datatype.
        '''
        for c in cols:
            self.df[c] = self.df[c].replace({'\$':'', ',':''}, regex = True).astype(float)

    ##### WHERE HOT-ENCODE/STANDARDIZE STARTS 

    def NaN_to_None(self, cols):
        '''
        Covert NaN's to 'none'.
        
        Parameters
        ----------
        df: pandas.DataFrame
        cols: list

        Returns
        ----------
        df: pandas.DataFrame
        '''
        for col in cols:
            self.df[col].fillna('none', inplace=True)

    def NaN_to_zero(self):
        '''
        Covert NaN's to zeros.
        
        Parameters
        ----------
        df: pandas.DataFrame

        Returns
        ----------
        df: pandas.DataFrame
        '''
        self.df['review_scores_rating'].fillna(value=0, inplace=True)
        self.df['bedrooms'].fillna(value=0, inplace=True)
        self.df['bathrooms'].fillna(value=0, inplace=True)  
        self.df['beds'].fillna(value=0, inplace=True)     

    def host_in_Denver(self):
        '''
        Hot encode if the host location is in Denver (1) or not (0).

        Parameters
        ----------
        df: pandas.DataFrame

        Returns
        ----------
        df: pandas.DataFrame
        '''
        self.df['host_loc_denver'] = self.df['host_location'].map(lambda x: 1.0 if x == 'Denver, Colorado, United States' else 0.0)

    def true_false_hot_enconde(self):
        '''
        Hot encodes if the host is a superhost an if listing requires a license — 
        both (1) if yes, (0) if no.
        
        Parameters
        ----------
        df: pandas.DataFrame

        Returns
        ----------
        df: pandas.DataFrame
        '''
        self.df['is_superhost'] = self.df['host_is_superhost'].map(lambda x: 1.0 if x == 't' else 0.0) 
        self.df['needs_license'] = self.df['requires_license'].map(lambda x: 1.0 if x == 't' else 0.0) 

    def in_top_10_neighbourhood(self):
        '''
        Hot encode if the listing location is in the one of the top 10 highest listed neighbourhoods.
        (1) if yes, (0) if no.
        
        Parameters
        ----------
        df: pandas.DataFrame
        cols: 

        Returns
        ----------
        df: pandas.DataFrame
        '''
        df_prop_type_per_hood = self.df.groupby(['neighbourhood_cleansed','room_type']).size().to_frame('count').reset_index()
        df_hoodtop10 = df_prop_type_per_hood.groupby(['neighbourhood_cleansed'])['count'].sum().sort_values(ascending=False)
        df_hoodtop10 = df_hoodtop10.iloc[0:10].reset_index()
        top10neighborhoods = df_hoodtop10['neighbourhood_cleansed'].tolist()
        self.df['in_top_10_neighbourhood'] = self.df['neighbourhood_cleansed'].map(lambda x: 1.0 if x in top10neighborhoods else 0.0) 

    def listing_location(self):
        '''
        Hot encode if the listing location is in Denver.
        (1) if yes, (0) if no.

        Parameters
        ----------
        df: pandas.DataFrame 

        Returns
        ----------
        df: pandas.DataFrame 
        '''
        self.df['list_loc_denver'] = self.df['city'].map(lambda x: 1.0 if x == 'Denver' else 0.0) 

    def fill_NaN_pricing(self):
        '''
        Fill NaN values with the estimated pricing for the weekly and monthly rates based on 
        the daily rate.
        
        Parameters
        ----------
        df: pandas.DataFrame 

        Returns
        ----------
        df: 
        '''
        self.df['weekly_price'].fillna(value=self.df['price']*7, inplace=True)
        self.df['monthly_price'].fillna(value=self.df['price']*30, inplace=True)

    def room_type_dummies(self):
        '''
        Get dummy variables for the three different room type options available — 
        Entire home/apt, Private Room, Shared Room.
        
        Parameters
        ----------
        df: pandas.DataFrame 

        Returns
        ----------
        df: pandas.DataFrame 
        '''
        self.df = pd.get_dummies(self.df, columns=['room_type'])

    def current_license(self):
        '''
        Create new column that identifies whether or not the license number is on the site and current.
        
        Parameters
        ----------
        df: pandas.DataFrame 

        Returns
        ----------
        df: pandas.DataFrame 
        '''
        self.df['license'].fillna(0, inplace=True)
        searchString = "2019"
        find_current_df = self.df.loc[df['license'].str.contains(searchString, regex=False, na=False)]
        current_lst = find_current_df['license'].tolist()
        self.df['current_license'] = self.df['license'].map(lambda x: 1.0 if x in current_lst else 0.0) 

    def drop_cols(self):
        '''
        Remove duplicate columns orginally need to create the new hot-encoded or dummy columns.
        
        Parameters
        ----------
        df: pandas.DataFrame 

        Returns
        ----------
        df: pandas.DataFrame 
        '''
        self.df.drop(columns=['host_location', 'host_is_superhost', 'city', 'requires_license', 'license'], axis=1, inplace=True)

    def add_violation_col(self):
        '''
        Create column that will display confirmed listings that are in violation of Denver's Short-Term Rental regulations.
        
        Parameters
        ----------
        df: pandas.DataFrame 

        Returns
        ----------
        df: pandas.DataFrame 
        '''
        self.df['is_violating'] = self.df['listing_url'].map(lambda x: 1.0 if x in violater else 0.0) 

    def standardize_pricing(self,cols):
        '''
        Standardize pricing columns
        
        Parameters
        ----------
        df: pandas.DataFrame 
        cols: 

        Returns
        ----------
        df: pandas.DataFrame
        '''
        features = self.df[cols]
        scaler = StandardScaler().fit(features.values)
        features = scaler.transform(features.values)
        self.df[cols] = features

    def clean_roundup(self):
        self.host_in_Denver()
        self.true_false_hot_enconde()
        self.in_top_10_neighbourhood()
        self.listing_location()
        self.fill_NaN_pricing()
        self.NaN_to_zero()
        self.room_type_dummies()
        self.current_license()
        self.drop_cols()
        self.add_violation_col()

    def save(self):
        self.df.to_pickle('../data/pickled_listings_df')

if __name__ == '__main__':
    df = pd.read_csv('../data/listings.csv')

    clean = Clean(df)

    columns_to_keep = ['id', 'listing_url', 'summary', 'space', 'description', 'notes', 'access', 'interaction', 
                       'house_rules', 'host_id', 'host_url','host_location', 
                       'host_about', 'host_is_superhost', 'neighbourhood_cleansed', 'city',
                       'price','weekly_price','monthly_price', 'room_type','accommodates','bathrooms', 
                       'bedrooms', 'beds', 'minimum_nights', 'maximum_nights','review_scores_rating', 
                       'requires_license','license', 'calculated_host_listings_count_entire_homes', 
                       'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms']

    float_cols = ['price','weekly_price','monthly_price']
    text_cols = ['summary', 'space', 'description', 'notes', 'access', 'interaction', 'house_rules', 'host_about']
    violater = ['https://www.airbnb.com/rooms/2086', 'https://www.airbnb.com/rooms/36026536',
                'https://www.airbnb.com/rooms/30991941', 'https://www.airbnb.com/rooms/36110171', 
                'https://www.airbnb.com/rooms/283162', 'https://www.airbnb.com/rooms/21281617', 
                'https://www.airbnb.com/rooms/27987428', 'https://www.airbnb.com/rooms/2139342',
                'https://www.airbnb.com/rooms/1488774', 'https://www.airbnb.com/rooms/1153002',
                'https://www.airbnb.com/rooms/2599115', 'https://www.airbnb.com/rooms/2827887',
                'https://www.airbnb.com/rooms/3495498', 'https://www.airbnb.com/rooms/4137490',
                'https://www.airbnb.com/rooms/4556950', 'https://www.airbnb.com/rooms/4671735',
                'https://www.airbnb.com/rooms/19214092', 'https://www.airbnb.com/rooms/4753876',
                'https://www.airbnb.com/rooms/4896381', 'https://www.airbnb.com/rooms/5070640',
                'https://www.airbnb.com/rooms/5402378', 'https://www.airbnb.com/rooms/35678320',
                'https://www.airbnb.com/rooms/35683218', 'https://www.airbnb.com/rooms/35828919',
                'https://www.airbnb.com/rooms/21753893', 'https://www.airbnb.com/rooms/5696654',
                'https://www.airbnb.com/rooms/8797683', 'https://www.airbnb.com/rooms/37746342',
                'https://www.airbnb.com/rooms/6288460', 'https://www.airbnb.com/rooms/6370140',
                'https://www.airbnb.com/rooms/7475742', 'https://www.airbnb.com/rooms/7859145',
                'https://www.airbnb.com/rooms/8211278', 'https://www.airbnb.com/rooms/8366762',
                'https://www.airbnb.com/rooms/8555795', 'https://www.airbnb.com/rooms/8739814',
                'https://www.airbnb.com/rooms/8829680', 'https://www.airbnb.com/rooms/8884899',
                'https://www.airbnb.com/rooms/8951180', 'https://www.airbnb.com/rooms/8989473',
                'https://www.airbnb.com/rooms/9165337', 'https://www.airbnb.com/rooms/20419783',
                'https://www.airbnb.com/rooms/9372481', 'https://www.airbnb.com/rooms/9591731',
                'https://www.airbnb.com/rooms/9169634', 'https://www.airbnb.com/rooms/10426535',
                'https://www.airbnb.com/rooms/10088031', 'https://www.airbnb.com/rooms/9237825',
                'https://www.airbnb.com/rooms/9330036', 'https://www.airbnb.com/rooms/9409334',
                'https://www.airbnb.com/rooms/9532575', 'https://www.airbnb.com/rooms/12172692',
                'https://www.airbnb.com/rooms/11377981', 'https://www.airbnb.com/rooms/21899331',
                'https://www.airbnb.com/rooms/11358422', 'https://www.airbnb.com/rooms/11240044',
                'https://www.airbnb.com/rooms/11191271', 'https://www.airbnb.com/rooms/11148272',
                'https://www.airbnb.com/rooms/11070296', 'https://www.airbnb.com/rooms/10995273',
                'https://www.airbnb.com/rooms/15743145', 'https://www.airbnb.com/rooms/10706175',
                'https://www.airbnb.com/rooms/10392285', 'https://www.airbnb.com/rooms/18000300',
                'https://www.airbnb.com/rooms/10190798', 'https://www.airbnb.com/rooms/9856869',
                'https://www.airbnb.com/rooms/9796646', 'https://www.airbnb.com/rooms/9795040',
                'https://www.airbnb.com/rooms/22307026', 'https://www.airbnb.com/rooms/9734548',
                'https://www.airbnb.com/rooms/9633450', 'https://www.airbnb.com/rooms/12711245',
                'https://www.airbnb.com/rooms/12681403', 'https://www.airbnb.com/rooms/12791280',
                'https://www.airbnb.com/rooms/13550337', 'https://www.airbnb.com/rooms/13377190',
                'https://www.airbnb.com/rooms/13140326', 'https://www.airbnb.com/rooms/12874653',
                'https://www.airbnb.com/rooms/13688138', 'https://www.airbnb.com/rooms/13656290',
                'https://www.airbnb.com/rooms/13877553', 'https://www.airbnb.com/rooms/14654146',
                'https://www.airbnb.com/rooms/14902544', 'https://www.airbnb.com/rooms/15097005',
                'https://www.airbnb.com/rooms/15237689', 'https://www.airbnb.com/rooms/38982793',
                'https://www.airbnb.com/rooms/19141160', 'https://www.airbnb.com/rooms/15583685',
                'https://www.airbnb.com/rooms/15585225', 'https://www.airbnb.com/rooms/15641776',
                'https://www.airbnb.com/rooms/15680276', 'https://www.airbnb.com/rooms/15745694',
                'https://www.airbnb.com/rooms/15807599', 'https://www.airbnb.com/rooms/27545807',
                'https://www.airbnb.com/rooms/14125469', 'https://www.airbnb.com/rooms/16497996',
                'https://www.airbnb.com/rooms/16443175', 'https://www.airbnb.com/rooms/16392236',
                'https://www.airbnb.com/rooms/16299372']
    standardize_cols = ['price', 'weekly_price', 'monthly_price']

    clean.select_cols(columns_to_keep)
    clean.to_float(float_cols)
    clean.NaN_to_None(text_cols)
    clean.clean_roundup()
    clean.standardize_pricing(standardize_cols)
    clean.save()