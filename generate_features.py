#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:29:50 2017

@author: tingzhu
"""
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# generate data features for fitting
# type1 feature simply combines customer and product data
# type2 feature consolidates customer and product data
def generate_features(customer_features,product_features, last_assortment,next_assortment):
    
    # prepare dataframe, including the use of one-hot encoding
    customer_df=create_customer_features(customer_features)
    product_df=create_product_features(product_features)
    
    # prepare data features for last month
    features_type1_last, features_type2_last=combine_df(last_assortment, customer_df, product_df)
    
    # prepare data features for next month
    features_type1_next, features_type2_next = combine_df_nextmonth(next_assortment,customer_df, product_df)
    
    return features_type1_last, features_type2_last,features_type1_next, features_type2_next
    

def create_customer_features(customer_features):
    
    # one-hot the age feature
    age_df=pd.get_dummies(customer_features['age_bucket'],prefix='age')

    # relabel "is_returning_customer" to 1 or 0
    label_encoder = LabelEncoder()
    costumer_return_feature = label_encoder.fit_transform(list(customer_features['is_returning_customer']))
    costumer_return_df=pd.DataFrame(costumer_return_feature)
    costumer_return_df.columns = ['is_returning_customer']
    
    # one-hot the favorite_genres feature
    customer_genre_df = onehot_favorite_genres(customer_features['favorite_genres'])
    
    # combine features
    customer_df=pd.concat([customer_features['customer_id'],age_df,costumer_return_df,customer_genre_df],axis=1)

    if len(customer_features) != len(customer_df):
        print('ERROR: Something wrong with concatenanting dfs for final customer_features')
    
    return customer_df

# one-hot each user's favorite genres 
def onehot_favorite_genres(favorite_genres):
    
    string2list_genre=[customer_genre.replace("'",'').replace('[','').replace(']','').lstrip().split(',') for customer_genre in favorite_genres]
    flat_list_genre=[item.lstrip() for customer in string2list_genre for item in customer]
   
    num_label=np.arange(len(set(flat_list_genre)))
    genre_dict=dict(zip(set(flat_list_genre),num_label))

    total_num_genre=len(genre_dict)
    genre_onehot=[]

    for row in string2list_genre:
        temp=np.array([genre_dict[i.lstrip()] for i in row])
        genre_onehot_row=[0]*total_num_genre
        for i in temp:
            genre_onehot_row[i]=1
        genre_onehot.append(genre_onehot_row)
        
    col_names=['customer_'+key for key in genre_dict]
    customer_genre_df=pd.DataFrame(genre_onehot,columns=col_names)
    
    customer_genre_df=customer_genre_df.rename(columns = {'customer_':'customer_noEntry'})
    
    return customer_genre_df

# prepare product features
def create_product_features(product_features):
    
    # normalize the length of books
    length_df=normalize_bookLength(product_features['length'],1)
    
    #one-hot the "difficulty" feature
    difficulty_df=pd.get_dummies(product_features['difficulty'],prefix='difficulty')
    
    # one-hot the "genre" feature
    genre_df=pd.get_dummies(product_features['genre'],prefix='product')
    
    # tranform "fiction" feature to 1/0
    label_encoder = LabelEncoder()
    fiction_feature = label_encoder.fit_transform(list(product_features['fiction']))
    fiction_feature_df=pd.DataFrame(fiction_feature )
    fiction_feature_df.columns = ['fiction']
    
    # combine features
    product_df = pd.concat([product_features['product_id'],length_df,difficulty_df,fiction_feature_df,genre_df], axis=1)
    return product_df

# normalize the book length feature
# concern that some book lengths are much bigger than the rest
# option to use log(book length)
def normalize_bookLength(bookLength, useLog):

    orig_length=np.array(bookLength)
    
    if useLog==1:
        norm_length=np.log(orig_length)
        norm_length=((norm_length-min(norm_length))/(max(norm_length)-min(norm_length)))
    else:
        norm_length=((orig_length-min(orig_length))/(max(orig_length)-min(orig_length)))

    product_length=pd.DataFrame(norm_length,columns=['length'])
    return product_length

# generate data features by simply combining customer and product features
# also find matches of product_genre with customer's favorite_genre
def combine_df(assortment, customer_df, product_df):
    
    step1 = assortment.merge(customer_df, left_on='customer_id', right_on='customer_id', how='left')
    result = step1.merge(product_df, left_on='product_id', right_on='product_id', how='left')
    result_type1 = result
    
    # type2 data feature engineering:
    # create a new feature (matched_picks) which is 1 if the book genre is one of customer's favorite genres
    
    customer_choice=result.iloc[:,12:23]
    product_picked=result.iloc[:,31:]
    
    col_names = customer_choice.columns.values 
    matched_picks=pd.DataFrame(columns=col_names,index=np.arange(len(customer_choice)))

    for i in np.arange(len(customer_choice)):
        product_picked_name=product_picked.iloc[i][product_picked.iloc[i]==1].index[0]
        customer_product_match_name='customer_'+product_picked_name.split('_')[1]
    
        if customer_choice[customer_product_match_name][i]==1:
            matched_picks.loc[i,customer_product_match_name]=1
        
        else:
            matched_picks.loc[i,customer_product_match_name]=0
        
    matched_picks.fillna(value=0,inplace=True)
    new_col_names=['matched_'+ name.split('_')[1] for name in col_names]
    matched_picks.columns=new_col_names
    
    part1=result.iloc[:,:12]
    part2=result.iloc[:,23:30]
    
   
    result_type2 = pd.concat([part1,part2,matched_picks],axis=1)
    
    return result_type1, result_type2

# generate data features by simply combining customer and product features
# also find matches of product_genre with customer's favorite_genre
# same as combine_df, but without the column of "purchsed" in next_month_assortment
def combine_df_nextmonth(next_month_assortment, customer_df, product_df):
    
    step1 = next_month_assortment.merge(customer_df, left_on='customer_id', right_on='customer_id', how='left')
    result = step1.merge(product_df, left_on='product_id', right_on='product_id', how='left')
    result_type1 = result
 
    # type2 data feature engineering:
    # create a new feature (matched_picks) which is 1 if the book genre is one of customer's favorite genres
    
    customer_choice=result.iloc[:,11:22]
    product_picked=result.iloc[:,30:]
    
    col_names = customer_choice.columns.values 
    matched_picks=pd.DataFrame(columns=col_names,index=np.arange(len(customer_choice)))

    for i in np.arange(len(customer_choice)):
        product_picked_name=product_picked.iloc[i][product_picked.iloc[i]==1].index[0]
        customer_product_match_name='customer_'+product_picked_name.split('_')[1]
    
        if customer_choice[customer_product_match_name][i]==1:
            matched_picks.loc[i,customer_product_match_name]=1
        
        else:
            matched_picks.loc[i,customer_product_match_name]=0
        
    matched_picks.fillna(value=0,inplace=True)
    new_col_names=['matched_'+ name.split('_')[1] for name in col_names]
    matched_picks.columns=new_col_names
    
    part1=result.iloc[:,:11]
    part2=result.iloc[:,22:29]
    
   
    result_type2 = pd.concat([part1,part2,matched_picks],axis=1)
    
    # potentially the feature "is_returning_customer" should be updated since 
    # it is another month now
    # result_type2['is_returning_customer']=1
    
    return result_type1, result_type2