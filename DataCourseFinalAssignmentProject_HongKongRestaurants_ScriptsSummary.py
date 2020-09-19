# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 23:00:57 2020

@author: Siwei Luo

Combination of the Scripts for the final assignment of IBM DataScience Course
Project: Where is a restaurant located in Hong Kong City?
"""
#--------------------------------------------------------------------------------------------------------
"""
First Part: Data Scraper
"""
#--------------------------------------------------------------------------------------------------------
#%%
#load packages

#import packages
import pandas as pd
import math
from bs4 import BeautifulSoup
import requests
import json
from pandas.io.json import json_normalize
from geopy.distance import geodesic


#Define a function extracting the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']

#Set Foursquare credential
CLIENT_ID = 'IRKXXFZ3W5AWF25ZVSUVTXCZPCCZROGG3JH024WGIY3DYJNR' # your Foursquare ID
CLIENT_SECRET = 'DAMJQDDXDTMVPPR13DPM0KX3XKMMGJK40FFP1XFMWE4Z4ACK' # your Foursquare Secret
VERSION = '20200101' # Foursquare API version
#set fixed URL parameter LIMIT
LIMIT = 50

#%%
#Get the venue category hierarchy ---DOWNLOADING---
url = 'https://api.foursquare.com/v2/venues/categories?client_id={}&client_secret={}&v={}'.format(CLIENT_ID, CLIENT_SECRET, VERSION)
results = requests.get(url).json()
categories = results['response']['categories']
#%%
#get the hierarchical category ID
df_hierarchical_cat = pd.DataFrame(columns = ['categories','c5','c4','c3','c2','c1'])
for c1 in categories:
    c1_id = c1['id']
    h_row = pd.DataFrame([[c1_id,c1_id,c1_id,c1_id,c1_id,c1_id]],columns = ['categories','c5','c4','c3','c2','c1'])
    df_hierarchical_cat = df_hierarchical_cat.append(h_row)
    for c2 in c1['categories']:
        c2_id = c2['id']
        h_row = pd.DataFrame([[c2_id,c2_id,c2_id,c2_id,c2_id,c1_id]],columns = ['categories','c5','c4','c3','c2','c1'])
        df_hierarchical_cat = df_hierarchical_cat.append(h_row)
        if len(c2['categories'])>0:
            for c3 in c2['categories']:
                c3_id = c3['id']
                h_row = pd.DataFrame([[c3_id,c3_id,c3_id,c3_id,c2_id,c1_id]],columns = ['categories','c5','c4','c3','c2','c1'])
                df_hierarchical_cat = df_hierarchical_cat.append(h_row)
                if len(c3['categories'])>0:
                    for c4 in c3['categories']:
                        c4_id = c4['id']
                        h_row = pd.DataFrame([[c4_id,c4_id,c4_id,c3_id,c2_id,c1_id]],columns = ['categories','c5','c4','c3','c2','c1'])
                        df_hierarchical_cat = df_hierarchical_cat.append(h_row)
                        if len(c4['categories'])>0:
                            for c5 in c4['categories']:
                                c5_id = c5['id']
                                h_row = pd.DataFrame([[c5_id,c5_id,c4_id,c3_id,c2_id,c1_id]],columns = ['categories','c5','c4','c3','c2','c1'])
                                df_hierarchical_cat = df_hierarchical_cat.append(h_row)
                                if len(c5['categories'])>0:
                                    categories_id = c5['id']
                                    h_row = pd.DataFrame([[categories_id,c5_id,c4_id,c3_id,c2_id,c1_id]],columns = ['categories','c5','c4','c3','c2','c1'])
                                    df_hierarchical_cat = df_hierarchical_cat.append(h_row)
                                else:
                                    categories_id = c5_id
                                    h_row = pd.DataFrame([[categories_id,c5_id,c4_id,c3_id,c2_id,c1_id]],columns = ['categories','c5','c4','c3','c2','c1'])
                                    df_hierarchical_cat = df_hierarchical_cat.append(h_row)
                        else:
                            categories_id = c4_id
                            c5_id = c4_id
                            h_row = pd.DataFrame([[categories_id,c5_id,c4_id,c3_id,c2_id,c1_id]],columns = ['categories','c5','c4','c3','c2','c1'])
                            df_hierarchical_cat = df_hierarchical_cat.append(h_row)
                else:
                    categories_id = c3_id
                    c5_id = c3_id
                    c4_id = c3_id
                    h_row = pd.DataFrame([[categories_id,c5_id,c4_id,c3_id,c2_id,c1_id]],columns = ['categories','c5','c4','c3','c2','c1'])
                    df_hierarchical_cat = df_hierarchical_cat.append(h_row)
        else:
            categories_id = c2_id
            c5_id = c2_id
            c4_id = c2_id
            c3_id = c2_id
            h_row = pd.DataFrame([[categories_id,c5_id,c4_id,c3_id,c2_id,c1_id]],columns = ['categories','c5','c4','c3','c2','c1'])
            df_hierarchical_cat = df_hierarchical_cat.append(h_row)
df_hierarchical_cat.reset_index(inplace = True, drop = True)
#%%
df_hierarchical_cat.to_csv('C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/hierarchical_cat_id.csv')

#%%
###Define recursive downloading function
def RD(lat,lng,radius,categoryID,LIMIT,CLIENT_ID,CLIENT_SECRET,VERSION,f_t=0):
    #define URL
    url = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&radius={}&v={}&limit={}&categoryId={}'.format(CLIENT_ID,CLIENT_SECRET,lat,lng,radius,VERSION,LIMIT,categoryID)
    #download results
    try:
        results = requests.get(url).json()
        response = results['response']
        #If there is at least a required venue
        if len(response)>=1:
            venues = results['response']['venues']
            if len(venues)>=1:
                #transform to dataframe
                df_venues = json_normalize(venues)
                #data frame wraggling
                df_venues['categories'] = df_venues.apply(get_category_type, axis=1)
                df_venues = df_venues.loc[:,['id','name','categories','location.lat','location.lng']]
                df_venues = df_venues[df_venues['categories']!='Moving Target']
                df_venues.columns = ['ID','name','categories','latitude','longitude']
            else:
                df_venues = pd.DataFrame(columns = ['ID','name','categories','latitude','longitude'])
        else:
            df_venues = pd.DataFrame(columns = ['ID','name','categories','latitude','longitude'])
        return df_venues
    except:
        f_t = f_t + 1
        if f_t <= 30:
            print('failed!...Retry:',f_t)
            return RD(lat,lng,radius,categoryID,LIMIT,CLIENT_ID,CLIENT_SECRET,VERSION,f_t)
        else:
            print('failed more than 30 times! Please check and restart the downloading process from the returned dataframe!')
            df_venues = pd.DataFrame(['Failed!...radius=',radius,categoryID,lat,lng],columns = ['ID','name','categories','latitude','longitude'])

#Function for progressively creating dynamic map grids and downloading
def MGD(n,e,s,w,categoryID,LIMIT,CLIENT_ID,CLIENT_SECRET,VERSION):
    venues_summary = pd.DataFrame(columns = ['ID','name','categories','latitude','longitude'])
    lat = s + 0.5*(n-s)
    lng = w + 0.5*(e-w)
    radius = ((max(n-s,e-w)/360)*(2*math.pi*6379000))/(2**0.5) #use maximum earth radius
    df_venues = RD(lat,lng,radius,categoryID,LIMIT,CLIENT_ID,CLIENT_SECRET,VERSION)
    df_maxlen = len(df_venues)
    d = 1
    while df_maxlen==LIMIT and d<=4:
        dt = 2**d
        sep1 = (n-s)/dt
        sep2 = (e-w)/dt
        radius1 = radius/dt
        print('dividing d=',d,'partsNo dt=',dt)
        list_df = []
        for i in range(dt): #north and south
            for j in range(dt): #east and west
                s1 = s + sep1*i
                n1 = s + sep1*(i+1)
                w1 = w + sep2*j
                e1 = w + sep2*(j+1)
                lat1 = s1 + 0.5*(n1-s1)
                lng1 = w1 + 0.5*(e1-w1)
                df_venues = RD(lat1,lng1,radius1,categoryID,LIMIT,CLIENT_ID,CLIENT_SECRET,VERSION)
                list_df.append(df_venues)
        df_maxlen = 0
        for df in list_df:
            df_maxlen = max(len(df),df_maxlen)
        d = d + 1
    if d > 1:
        for df in list_df:
            venues_summary = venues_summary.append(df)
    else:
        venues_summary = df_venues
    return venues_summary
            
#%%
#set total range of the Hong Kong city
north = 22.344567
east = 114.222713
south = 22.273698
west = 114.117974

#Create a dataframe to save the downloaded venues
All_venues = pd.DataFrame(columns = ['ID','name','categories','latitude','longitude'])

#separate the map into 36*18 sectors
sep_la = (north - south)/18
sep_lo = (east - west)/36

#%% --- MAIN DOWNLOADING PROCESS ---
#Download the venues by c1 category
for cat in df_hierarchical_cat['c1'].unique():
    for j in range(36):
        for i in range(18):
            s = south + sep_la*i
            n = south + sep_la*(i+1)
            w = west + sep_lo*j
            e = west + sep_lo*(j+1)
            All_venues = All_venues.append(MGD(n,e,s,w,cat,LIMIT,CLIENT_ID,CLIENT_SECRET,VERSION))
        All_venues.drop_duplicates(inplace=True)
        All_venues.to_csv('C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/ResNB_venues/DownloadVenuesBackup_cat_{}_Lng_j_{}.csv'.format(cat,j),encoding='utf-8-sig')

#%%
###Download number of likes and calculate the distance between the venues

###Calculate the distance between the venues
#%%
#load venues data
All_venues = pd.read_csv('C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/ResNB_venues/DownloadVenuesBackup_cat_4d4b7105d754a06379d81259_Lng_j_35.csv',index_col = 0)

#load category name
df_cat_name = pd.read_csv('C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/hierarchical_cat.csv',index_col = 0)
df_cat_name = df_cat_name.reset_index(drop=True)

#add hierarchical categories to the venues dataframe
AV_cat = All_venues.merge(df_cat_name,on='categories',how='left')

#subset the restaurant dataframe
df_res = AV_cat[AV_cat['c2'].str.contains('Restaurant')]
df_res = df_res.reset_index(drop=True)

#%%
#Reload df_res & df_likes
df_res = pd.read_csv('C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/df_res_n_likes.csv',index_col = 0)

#%% ---CALCULATION---Run in multiple kernels---
#Distance computation
#initialize a list for distances for a restaurant
list_df_dist = []
#Estimated distance in radian (limit of 1 km, approaximate earth radius as 6378 km, applied bias 5%)
limit_r = (1*1.05)/(2*math.pi*6378)*360

for i in range(0,len(df_res)):
    list_dist_res = []
    for j in range(len(AV_cat)):
        ID_res = df_res['ID'][i]
        ID_nb = AV_cat['ID'][j]
        name_nb = AV_cat['name'][j]
        lat1 = df_res['latitude'][i]
        lng1 = df_res['longitude'][i]
        lat2 = AV_cat['latitude'][j]
        lng2 = AV_cat['longitude'][j]
        radian_dist1 = max(abs(lat1 - lat2),abs(lng1 - lng2))
        if radian_dist1 < limit_r:
            radian_dist2 = (lat1-lat2)**2 + (lng1-lng2)**2
            if radian_dist2 < limit_r**2:
                coord1 = (lat1,lng1)
                coord2 = (lat2,lng2)
                dist = geodesic(coord1,coord2).kilometers
                list_dist_res.append([ID_res,ID_nb,name_nb,dist])
    df_dist_res = pd.DataFrame(list_dist_res,columns=['ID_res','ID_nb','name_nb','geod_dist'])
    df_dist_res_nb = df_dist_res[df_dist_res['geod_dist']<=1]
    list_df_dist.append(df_dist_res_nb)
    if i%100 == 0:
        print('res loop No.:',i)

#%% ---Load Calculation results from multiple kernels
df_nb_dist0 = pd.read_csv('C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/ResNB_venues/NBvenues_distance_0_4000.csv',index_col = 0)
df_nb_dist1 = pd.read_csv('C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/ResNB_venues/NBvenues_distance_4000_8000.csv',index_col = 0)
df_nb_dist2 = pd.read_csv('C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/ResNB_venues/NBvenues_distance_8000_e.csv',index_col = 0)

#%%
df_nb_dist_list = [df_nb_dist0,df_nb_dist1,df_nb_dist2]
df_nb_dist = pd.concat(df_nb_dist_list,ignore_index=True)
del(df_nb_dist0,df_nb_dist1,df_nb_dist2)
#%%
df_nb_dist.to_csv('C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/ResNB_venues/NBvenues_distance.csv',encoding = 'utf-8-sig')
###Download number of likes of the restaurants

###Download number of likes of the restaurants
#%%
#define function to get likes of a restaurant
def RDL(ID,CLIENT_ID, CLIENT_SECRET, VERSION,f_t=0):
    url = 'https://api.foursquare.com/v2/venues/{}/likes?client_id={}&client_secret={}&v={}'.format(ID,CLIENT_ID, CLIENT_SECRET, VERSION)
    try:
        results = requests.get(url).json()
        likes = results['response']['likes']['count']
        return [ID,likes]
    except:
        f_t = f_t + 1
        if f_t <= 30:
            print('failed!...Retry:',f_t)
            return RDL(ID,CLIENT_ID, CLIENT_SECRET, VERSION,f_t)
        else:
            print('failed more than 30 times! Please check and restart the downloading process from the returned dataframe!')
            return [ID,'failed']

#%% ---DOWNLOADING PROCESS---
#initialize a likes data frame
list_likes = []
#Download No. of likes
for ID in df_res['ID']:
    list_likes.append(RDL(ID,CLIENT_ID,CLIENT_SECRET,VERSION))
#transform results into dataframe
df_likes = pd.DataFrame(list_likes,columns=['ID','likes'])

#%%
type_error = []
for i in range(len(df_likes)):
    if type(df_likes['likes'][i])!=int:
        type_error.append([df_likes['ID'][i],df_likes['likes'][i]])
#%%
df_res[df_res['ID']==type_error[0][0]]
#%%
df_likes['likes'][df_likes['ID']==type_error[0][0]] = 0
df_likes['likes'][df_likes['ID']==type_error[0][0]]
#%%
df_res = df_res.merge(df_likes,on='ID')

#%%
df_res.to_csv('C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/df_res_n_likes.csv',encoding = 'utf-8-sig')

#%%
#--------------------------------------------------------------------------------------------------------
"""
Second Part: Data wrangling
"""
#--------------------------------------------------------------------------------------------------------
#%%
##import packages

#basic
import pandas as pd
import numpy as np
#dealing with large dataset
import dask.dataframe as dd
#get current time
from datetime import datetime
#For PCA etc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#For visulization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib._color_data as mcd
#For clustering
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist #point pair distance computation
from sklearn.cluster import KMeans
#For correlation
from scipy.stats import spearmanr
from scipy.stats import pearsonr
#Mapping
import folium
import webbrowser

#%%
#load data
df_res = pd.read_csv('C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/df_res_n_likes.csv',index_col = 0)
df_cat = pd.read_csv('C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/hierarchical_cat.csv',index_col = 0)
All_venues = pd.read_csv('C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/ResNB_venues/DownloadVenuesBackup_cat_4d4b7105d754a06379d81259_Lng_j_35.csv',index_col = 0)
#append hierarchical categories to the venues
All_venues = All_venues.merge(df_cat,on = 'categories')
#load distance data
df_dist = pd.read_csv('C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/ResNB_venues/NBvenues_distance.csv')
df_dist = df_dist.drop(labels = df_dist.columns[0],axis=1)

#%% separate the large distance dataframe into dataframe pieces
list_df_dist = []
for i in range(len(df_res)):
    res_ID = df_res['ID'][i]
    df = df_dist[df_dist['ID_res']==res_ID]
    list_df_dist.append(df)
    if i%100 == 0:
        print(datetime.now().strftime("%H:%M:%S"),'transform loop:',i)

#-------------------From segemented dataframe list-----------------------------
#%%
#category number counting data
#categories summery
cat_list = list(All_venues['categories'].unique())
col_ana = ['ID_res','name_res','category_res'] + cat_list
len(col_ana)

#%%
#within 2000 meters
df_count_cat = pd.DataFrame(columns=col_ana)
for i in range(len(df_res)):
    res_ID = df_res['ID'][i]
    res_name = df_res['name'][i]
    res_category = df_res['categories'][i]
    df_dist_cat = list_df_dist[i]
    df_dist_cat = df_dist_cat[df_dist_cat['geod_dist']<=2] #define distance of nearby
    df_dist_cat = df_dist_cat[df_dist_cat['geod_dist']>0]
    df_dist_cat.columns = ['ID_res','ID','geod_dist']
    df_dist_cat = df_dist_cat.merge(All_venues,on = 'ID',how = 'left')
    count_row = [res_ID,res_name,res_category]
    for cat in cat_list:
        df = df_dist_cat[df_dist_cat['categories']==cat]
        cat_num = len(df)
        count_row.append(cat_num)
    df_count_row = pd.DataFrame([count_row],columns = col_ana)
    df_count_cat = df_count_cat.append(df_count_row)
    if i%100 == 0:
        print(datetime.now().strftime("%H:%M:%S"),'transform loop:',i)
df_count_cat.reset_index(inplace=True,drop=True)

#%%
#build available hierarchical categories in Hong Kong
df_cat_hk = pd.DataFrame(cat_list,columns = ['cat_hk'])
df_cat_hk = df_cat_hk.merge(df_cat , how = 'left',left_on = 'cat_hk',right_on = 'categories')

#%%
#Transform basic category to c2 and c1
#categories summery
c2_list = list(All_venues['c2'].unique())
col_ana_c2 = ['ID_res','name_res','category_res'] + c2_list
df_count_c2 = pd.DataFrame(index = df_res.index, columns = col_ana_c2)
df_count_c2['ID_res'] = df_res['ID']
df_count_c2['name_res'] = df_res['name']
df_count_c2['category_res'] = df_res['categories']
for i in range(len(c2_list)):
    c2_name = c2_list[i]
    sub_cat = list(df_cat_hk.loc[df_cat_hk['c2'] == c2_name,'categories'])
    df_c2 = df_count_cat[sub_cat]
    df_count_c2[c2_name] = df_c2.sum(axis = 1)
    
#%%
#To c1
c1_list = list(All_venues['c1'].unique())
col_ana_c1 = ['ID_res','name_res','category_res'] + c1_list
df_count_c1 = pd.DataFrame(index = df_res.index, columns = col_ana_c1)
df_count_c1['ID_res'] = df_res['ID']
df_count_c1['name_res'] = df_res['name']
df_count_c1['category_res'] = df_res['categories']
for i in range(len(c1_list)):
    c1_name = c1_list[i]
    sub_cat = list(df_cat_hk.loc[df_cat_hk['c1'] == c1_name,'categories'])
    df_c1 = df_count_cat[sub_cat]
    df_count_c1[c1_name] = df_c1.sum(axis = 1)

#%%
#category nearest data
#within 2000 meters
mask_dist = 2
df_ndist_cat = pd.DataFrame(columns=col_ana)
for i in range(len(df_res)):
    res_ID = df_res['ID'][i]
    res_name = df_res['name'][i]
    res_category = df_res['categories'][i]
    df_dist_cat = list_df_dist[i]
    df_dist_cat = df_dist_cat[df_dist_cat['geod_dist']<=2] #define distance of nearby
    df_dist_cat = df_dist_cat[df_dist_cat['geod_dist']>0]
    df_dist_cat.columns = ['ID_res','ID','geod_dist']
    df_dist_cat = df_dist_cat.merge(All_venues,on = 'ID',how = 'left')
    dist_row = [res_ID,res_name,res_category]
    for cat in cat_list:
        df_dist_cat_cat = df_dist_cat[df_dist_cat['categories']==cat]
        if len(df_dist_cat_cat) > 0:
            cat_dist = min(df_dist_cat_cat['geod_dist'])
        else:
            cat_dist = mask_dist
        dist_row.append(cat_dist)
    df_dist_row = pd.DataFrame([dist_row],columns = col_ana)
    df_ndist_cat = df_ndist_cat.append(df_dist_row)
    if i%100 == 0:
        print(datetime.now().strftime("%H:%M:%S"),'transform loop:',i)
df_ndist_cat.reset_index(inplace=True,drop=True)

#%%
#Transform basic category to c2 and c1
#categories summery
c2_list = list(All_venues['c2'].unique())
col_ana_c2 = ['ID_res','name_res','category_res'] + c2_list
df_ndist_c2 = pd.DataFrame(index = df_res.index, columns = col_ana_c2)
df_ndist_c2['ID_res'] = df_res['ID']
df_ndist_c2['name_res'] = df_res['name']
df_ndist_c2['category_res'] = df_res['categories']
for i in range(len(c2_list)):
    c2_name = c2_list[i]
    sub_cat = list(df_cat_hk.loc[df_cat_hk['c2'] == c2_name,'categories'])
    df_c2 = df_ndist_cat[sub_cat]
    df_ndist_c2[c2_name] = df_c2.min(axis = 1)

#%%
#To c1
c1_list = list(All_venues['c1'].unique())
col_ana_c1 = ['ID_res','name_res','category_res'] + c1_list
df_ndist_c1 = pd.DataFrame(index = df_res.index, columns = col_ana_c1)
df_ndist_c1['ID_res'] = df_res['ID']
df_ndist_c1['name_res'] = df_res['name']
df_ndist_c1['category_res'] = df_res['categories']
for i in range(len(c1_list)):
    c1_name = c1_list[i]
    sub_cat = list(df_cat_hk.loc[df_cat_hk['c1'] == c1_name,'categories'])
    df_c1 = df_ndist_cat[sub_cat]
    df_ndist_c1[c1_name] = df_c1.min(axis = 1)

#------------Data preparation completed--------------
#%%
#--------------------------------------------------------------------------------------------------------
"""
Third Part: Clustering analysis with nearby venues within 2000 meters
"""
#--------------------------------------------------------------------------------------------------------
#%% ---------------------------------2000m Venues C1 Basic Category analysis---------------------------------2000m Venues C1 Basic Category analysis---------------------------------2000m Venues C1 Basic Category analysis
#PCA of c1 COUNTS

#scaling and centering
X = np.array(df_count_c1.iloc[:,3:13])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#PCA
pca = PCA(n_components = 3)
#calculate transformed PCs
pca_count_c1 = pca.fit_transform(X_scaled)
df_PC_count = pd.DataFrame(data = pca_count_c1, columns = ['pc1','pc2','pc3'])
pc1 = df_PC_count['pc1']
pc2 = df_PC_count['pc2']
pc3 = df_PC_count['pc3']
#check the explained variance
pca.explained_variance_ratio_

#PC1, PC2, PC3 explains 0.70649292, 0.12889269, 0.06626713 of the total variance

#%%
#2D Scatter plot of PCs

fig = plt.figure(figsize = (8,8))
fig.tight_layout()

ax = fig.add_subplot(2,2,1)
ax.scatter(pc1,pc2,s = 10)
ax.set_xlabel('PC 1 ({:.1%})'.format(pca.explained_variance_ratio_[0]), fontsize = 15,labelpad=2)
ax.set_ylabel('PC 2 ({:.1%})'.format(pca.explained_variance_ratio_[1]), fontsize = 15,labelpad=-5)

ax = fig.add_subplot(2,2,2)
ax.scatter(pc1,pc3,s = 10)
ax.set_xlabel('PC 1 ({:.1%})'.format(pca.explained_variance_ratio_[0]), fontsize = 15,labelpad=2)
ax.set_ylabel('PC 3 ({:.1%})'.format(pca.explained_variance_ratio_[2]), fontsize = 15,labelpad=-5)

ax = fig.add_subplot(2,2,3)
ax.scatter(pc2,pc3,s = 10)
ax.set_xlabel('PC 2 ({:.1%})'.format(pca.explained_variance_ratio_[1]), fontsize = 15,labelpad=2)
ax.set_ylabel('PC 3 ({:.1%})'.format(pca.explained_variance_ratio_[2]), fontsize = 15,labelpad=-5)

#%%
#3D scatter plot of PCs

fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1,projection = '3d')

ax.scatter(pc1,pc2,pc3,c='r',marker='o')

ax.set_xlabel('PC 1 ({:.1%})'.format(pca.explained_variance_ratio_[0]), fontsize = 15)
ax.set_ylabel('PC 2 ({:.1%})'.format(pca.explained_variance_ratio_[1]), fontsize = 15)
ax.set_zlabel('PC 3 ({:.1%})'.format(pca.explained_variance_ratio_[2]), fontsize = 15)

plt.show()

#PCA indicated 6-7 clusters

#%%
#Use DBSCAN to extract the clusters in the PCA

X = pca_count_c1

D_cluster = DBSCAN(eps = 0.45, min_samples = 100).fit(X)

D_ClassNumber = len(np.unique(D_cluster.labels_))-1
print('class count:',D_ClassNumber)

#add class to df_PC_count
df_PC_count['D_class'] = D_cluster.labels_

#3D scatter plot of PCs
#Plot DBSCAN clusters in PCA plots

fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1,projection = '3d')

colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'pink', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray', 'beige']

for i in range(-1,D_ClassNumber):
    df_cluster = df_PC_count[df_PC_count['D_class']==i]
    color = colors[i+1]
    ax.scatter(df_cluster['pc1'],df_cluster['pc2'],df_cluster['pc3'],c=color,marker='o',s=10)

ax.set_xlabel('PC 1 ({:.1%})'.format(pca.explained_variance_ratio_[0]), fontsize = 15)
ax.set_ylabel('PC 2 ({:.1%})'.format(pca.explained_variance_ratio_[1]), fontsize = 15)
ax.set_zlabel('PC 3 ({:.1%})'.format(pca.explained_variance_ratio_[2]), fontsize = 15)

plt.show()

#%%
#Check details of the classes！

df_c1_class = df_count_c1
df_c1_class['D_class']=D_cluster.labels_
df_c1_class['likes']=df_res['likes']

#Group by DBSCAN classes
df_c1_D_grouped = df_c1_class.groupby(by='D_class').mean()

df_c1_D_grouped['res_counts'] = df_c1_class.groupby(by='D_class').count()['ID_res']

#%%
df_c1_D_grouped.to_csv('C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/Report/c1_count_D_groups.csv',encoding='utf-8-sig')

#%% Map the results

#add the group to the restaurant dataframe
df_res['group_c1cd'] = df_c1_class['D_class']+1

#studied area of Hong Kong
north = 22.344567
east = 114.222713
south = 22.273698
west = 114.117974
#Map latitude,longitude
latitude = (north+south)/2
longitude = (east+west)/2
res_map = folium.Map(location=[latitude, longitude], zoom_start=13)

colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'pink', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray', 'beige']

for i in range(0,len(df_res['group_c1cd'].unique())):
    df_res_group = df_res[df_res['group_c1cd']==i]
    color_i = colors[i]
    for lat,lng in zip(df_res_group['latitude'],df_res_group['longitude']):
        folium.features.CircleMarker([lat, lng],radius=3,color=color_i,fill = True,fill_color=color_i,fill_opacity=0.6).add_to(res_map)
        
res_map.save("C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/Report/Visulization_selected/Maps/map_c1_counts.html")
webbrowser.open_new_tab("C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/Report/Visulization_selected/Maps/map_c1_counts.html")

#%%
#Distribution of the c1 categories

#Get the distribution
df_count_c1_r = df_count_c1.iloc[:,0:13]
for i in range(len(df_count_c1_r)):
    row = df_count_c1_r.iloc[i,3:13]
    rowSum = sum(row)
    if rowSum>0:
        df_count_c1_r.iloc[i,3:13] = row/rowSum
    else:
        df_count_c1_r.iloc[i,3:13] = row

#%%
#PCA of c1 distribution ratios
X = np.array(df_count_c1_r.iloc[:,3:13])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#PCA
pca = PCA(n_components = 3)
#calculate transformed PCs
pca_count_r_c1 = pca.fit_transform(X_scaled)
df_PC_count_r = pd.DataFrame(data = pca_count_r_c1, columns = ['pc1','pc2','pc3'])
pc1 = df_PC_count_r['pc1']
pc2 = df_PC_count_r['pc2']
pc3 = df_PC_count_r['pc3']
#check the explained variance
pca.explained_variance_ratio_

#%%
#K-Means classification of the distribution

X = np.array(df_count_c1_r.iloc[:,3:13])

K_ClassNumber_r = 10
K_cluster_r = KMeans(n_clusters = K_ClassNumber_r, n_init = 20, max_iter = 500, random_state = 0).fit(X)
df_PC_count_r['K_class'] = K_cluster_r.labels_

#%%
#Check details of the classes！

df_c1_class_r = df_count_c1_r
df_c1_class_r['K_class']=K_cluster_r.labels_
df_c1_class_r['likes']=df_res['likes']

#%% group by DBSCAN classes
df_c1_K_grouped_r = df_c1_class_r.groupby(by='K_class').mean()
df_c1_K_grouped_r['ResCounts'] = list(df_c1_class_r.groupby(by='K_class').count().iloc[:,0])

#%%
df_c1_K_grouped_r.to_csv('C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/Report/c1_count_distributionRatio_K_groups.csv',encoding='utf-8-sig')

#%% Map the results

#add the group to the restaurant dataframe
df_res['group_c1rk'] = df_c1_class_r['K_class']

#studied area of Hong Kong
north = 22.344567
east = 114.222713
south = 22.273698
west = 114.117974
#Map latitude,longitude
latitude = (north+south)/2
longitude = (east+west)/2
res_map = folium.Map(location=[latitude, longitude], zoom_start=13)

colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'pink', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray', 'beige']

for i in range(0,len(df_res['group_c1rk'].unique())):
    df_res_group = df_res[df_res['group_c1rk']==i]
    color_i = colors[i]
    for lat,lng in zip(df_res_group['latitude'],df_res_group['longitude']):
        folium.features.CircleMarker([lat, lng],radius=3,color=color_i,fill = True,fill_color=color_i,fill_opacity=0.6).add_to(res_map)
        
res_map.save("C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/Report/Visulization_selected/Maps/map_c1_DisRatio.html")
webbrowser.open_new_tab("C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/Report/Visulization_selected/Maps/map_c1_DisRatio.html")

#%% ---------------------------------2000m Venues C1 Group Selected C2 detailed Category analysis---------------------------------2000m Venues C1 Group Selected C2 detailed Category analysis
df_X = df_count_c2.iloc[:,3:401]
df_X['groups']=df_c1_class['D_class']
df_0counts = pd.DataFrame(index = df_c1_D_grouped.index,columns = df_X.columns)

#%%
#Logic matrics deciding the frequency of the venues of each group in each c2 category
for i in range(len(df_0counts.index)):
    df = df_X[df_X['groups']==df_0counts.index[i]]
    for j in range(len(df.columns)-1):
        Zcount = df[df.iloc[:,j]==0].count()[0]
        Zcount0 = df[df.iloc[:,j]>=0].count()[0]
        if Zcount >= 0.5*Zcount0:
            df_0counts.iloc[i,j] = 0
        else:
            df_0counts.iloc[i,j] = 1

#%%
df_0counts = df_0counts.iloc[:,0:398].transpose()

#%%
df_0counts['rowSum'] = df_0counts.sum(axis=1)
df_0counts2 = df_0counts[df_0counts['rowSum']>0]
#%%
selected_col = list(df_count_c2.columns[0:3]) + list(df_0counts2.index) + ['likes']
len(selected_col)

#%%
#find the hierarchical category of the selected
df_c2_cat_s = pd.DataFrame(columns=df_cat_hk.columns)
for c2 in selected_col:
    df_c2_cat_rows = df_cat_hk[df_cat_hk['c2']==c2]
    df_c2_cat_s = df_c2_cat_s.append(df_c2_cat_rows)
df_c2_cat_s = df_c2_cat_s[['c2','c1']]
df_c2_cat_s.drop_duplicates(inplace=True)
#%%
df_c2_cat_s.to_csv('C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/Report/c2_selectedCat_for_c1_D_groups.csv',encoding='utf-8-sig')

#%%
df_count_c2s = df_count_c2[selected_col]

#%%
df_count_c2s_values = df_count_c2s.iloc[:,3:len(df_count_c2s.columns)]
df_count_c2s_values['groups'] = df_c1_class['D_class']
df_c1_D_grouped_c2 = df_count_c2s_values.groupby('groups').mean()

#%%
df_c1_D_grouped_c2.to_csv('C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/Report/c2_for_c1_D_groups.csv',encoding='utf-8-sig')

#%%
#c2 selected categories correlation
#rank correlation between nearby venues count of the selected c2 categories and likes
list_dist_c2s_corr = []
for i in range(3,63):
    var = df_count_c2s.columns[i]
    corr = spearmanr(df_count_c2s.iloc[:,i],df_count_c2s.iloc[:,-1])
    r = corr[0]
    p = corr[1]
    list_dist_c2s_corr.append([var,r,p])
df_sCorr_c2s = pd.DataFrame(list_dist_c2s_corr,columns = ['Category','rank_r','test_p'])

#%%
df_count_c2s_values = df_count_c2s.iloc[:,3:63]
df_count_c2s_catMeans = df_count_c2s_values.mean(axis = 0)
df_count_c2s_catStd = df_count_c2s_values.std(axis = 0)

df_sCorr_c2s['catNumberMeans'] = list(df_count_c2s_catMeans)
df_sCorr_c2s['catNumberStd'] = list(df_count_c2s_catStd)
df_sCorr_c2s['Num_Std/Avg'] = df_sCorr_c2s['catNumberStd']/df_sCorr_c2s['catNumberMeans']

#%%

df_sCorr_c2s.to_csv('C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/Report/c2s_count_rank_correlation.csv',encoding='utf-8-sig')

#%%
#rank correlation between nearby venues count ratio of the selected c2 categories and likes

#Get the ratios of the c2 categories
df_count_c2s_r = df_count_c2s.iloc[:,0:64]

#calculate ratios of the counts
for i in range(len(df_count_c2s_r)):
    row = df_count_c2s_r.iloc[i,3:63]
    rowSum = sum(row)
    if rowSum>0:
        df_count_c2s_r.iloc[i,3:63] = row/rowSum
    else:
        df_count_c2s_r.iloc[i,3:63] = row
    if i%100 == 0:
        print(datetime.now().strftime("%H:%M:%S"),'transform loop:',i)
        
#%%
#Rank correlation
list_dist_c2s_r_corr = []
for i in range(3,63):
    var = df_count_c2s_r.columns[i]
    corr = spearmanr(df_count_c2s_r.iloc[:,i],df_count_c2s_r.iloc[:,-1])
    r = corr[0]
    p = corr[1]
    list_dist_c2s_r_corr.append([var,r,p])
df_sCorr_c2s_r = pd.DataFrame(list_dist_c2s_r_corr,columns = ['Category','rank_r','test_p'])

#%%
df_count_c2s_r_values = df_count_c2s_r.iloc[:,3:63]
df_count_c2s_r_catMeans = df_count_c2s_r_values.mean(axis = 0)
df_count_c2s_r_catStd = df_count_c2s_r_values.std(axis = 0)

df_sCorr_c2s_r['catRatioMeans'] = list(df_count_c2s_r_catMeans)
df_sCorr_c2s_r['catRatioStd'] = list(df_count_c2s_r_catStd)
df_sCorr_c2s_r['Ratio_Std/Avg'] = df_sCorr_c2s_r['catRatioStd']/df_sCorr_c2s_r['catRatioMeans']

#%%
df_sCorr_c2s.to_csv('C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/Report/c2s_countRatio_rank_correlation.csv',encoding='utf-8-sig')

#%%

#add c1 DBSCAN group to the restaurant dataframe
df_res['group'] = df_c1_class['D_class']

#%%
#studied area of Hong Kong
north = 22.344567
east = 114.222713
south = 22.273698
west = 114.117974
#Map latitude,longitude
latitude = (north+south)/2
longitude = (east+west)/2
res_map = folium.Map(location=[latitude, longitude], zoom_start=13)

colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'pink', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray', 'beige']

for i in range(0,7):
    df_res_group = df_res[df_res['group']==i]
    color_i = colors[i]
    for lat,lng in zip(df_res_group['latitude'],df_res_group['longitude']):
        folium.features.CircleMarker([lat, lng],radius=3,color=color_i,fill = True,fill_color=color_i,fill_opacity=0.6).add_to(res_map)
        
res_map.save("C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/Report/Visulization_selected/Maps/map.html")
webbrowser.open_new_tab("C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/Report/Visulization_selected/Maps/map.html")
#%%
#--------------------------------------------------------------------------------------------------------
"""
Fourth Part: Clustering analysis with nearby venues within 200 meters and correlations with number of likes
"""
#--------------------------------------------------------------------------------------------------------
#%% ---------------------------------200m Venues C1 Basic Category analysis---------------------------------2000m Venues C1 Basic Category analysis---------------------------------2000m Venues C1 Basic Category analysis
#Distribution of the c1 categories

#Get the distribution
df_count_c1_r = df_count_c1.iloc[:,0:13]
for i in range(len(df_count_c1_r)):
    row = df_count_c1_r.iloc[i,3:13]
    rowSum = sum(row)
    if rowSum>0:
        df_count_c1_r.iloc[i,3:13] = row/rowSum
    else:
        df_count_c1_r.iloc[i,3:13] = row

#%%
#Try the hyperparameter radius from 1 to 100

X = np.array(df_count_c1_r.iloc[:,3:13])
ClassNum = pd.DataFrame(columns=['radius','ClassNumber'])
for i in range(1,21):
    D_cluster = DBSCAN(eps = 0.005*i, min_samples = 50).fit(X)
    D_ClassNumber = len(np.unique(D_cluster.labels_))-1
    ClassNum = ClassNum.append(pd.DataFrame([[i,D_ClassNumber]],columns=['radius','ClassNumber']))
#%%
plt.plot(0.005*ClassNum['radius'], ClassNum['ClassNumber'])
plt.show()

#%%
#set the parameter as 
X = np.array(df_count_c1_r.iloc[:,3:13])
D_cluster = DBSCAN(eps = 0.03, min_samples = 50).fit(X)
D_ClassNumber = len(np.unique(D_cluster.labels_))-1
D_ClassNumber

#%%
#add class to df_PC_count
df_PC_count['D_class'] = D_cluster.labels_

#Check details of the classes！

df_c1_class_r = df_count_c1_r
df_c1_class_r['D_class']=D_cluster.labels_
df_c1_class_r['likes']=df_res['likes']

#Group by DBSCAN classes
df_c1_r_D_grouped = df_c1_class_r.groupby(by='D_class').mean()

df_c1_r_D_grouped['res_counts'] = df_c1_class_r.groupby(by='D_class').count()['ID_res']

#%%
df_c1_r_D_grouped.to_csv('C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/Report/c1_count_r_D_groups_200.csv',encoding='utf-8-sig')

#%% Map the results

#add the group to the restaurant dataframe
df_res['group_c1cd'] = df_c1_class['D_class']+1

#studied area of Hong Kong
north = 22.344567
east = 114.222713
south = 22.273698
west = 114.117974
#Map latitude,longitude
latitude = (north+south)/2
longitude = (east+west)/2
res_map = folium.Map(location=[latitude, longitude], zoom_start=13)

colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'pink', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray', 'beige']

for i in range(0,1):
    df_res_group = df_res[df_res['group_c1cd']==6]
    color_i = colors[i]
    for lat,lng in zip(df_res_group['latitude'],df_res_group['longitude']):
        folium.features.CircleMarker([lat, lng],radius=3,color=color_i,fill = True,fill_color=color_i,fill_opacity=0.6).add_to(res_map)
        
res_map.save("C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/Report/Visulization_selected/Maps/map_selected.html")
webbrowser.open_new_tab("C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/Report/Visulization_selected/Maps/map_selected.html")

#%% ---------------------------------Correlation---------------------------------Correlation---------------------------------Correlation
#rank correlation of likes(if correlated) with count ratios of nearby categories
list_dist_c1_corr = []
for i in range(3,len(df_count_c1.columns)-2):
    var = df_count_c1.columns[i]
    corr = spearmanr(df_count_c1.iloc[:,i],df_count_c1.iloc[:,-1])
    r = corr[0]
    p = corr[1]
    list_dist_c1_corr.append([var,r,p])
df_sCorr_c1_dist = pd.DataFrame(list_dist_c1_corr,columns = ['Category','rank_r','test_p'])

#%%
#counts of all nearby venues correlation
counts_AllNB = df_count_c1.iloc[:,3:13].sum(axis=1)
var = 'NumberOfNearbyVenues'
corr = spearmanr(counts_AllNB,df_count_c1.iloc[:,-1])
r = corr[0]
p = corr[1]
df_sCorr_c1_dist = df_sCorr_c1_dist.append(pd.DataFrame([[var,r,p]],columns = ['Category','rank_r','test_p']))

#%%
df_sCorr_c1_dist.to_csv('C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/Report/c1_count_rank_correlation.csv',encoding='utf-8-sig')

#%%

#rank correlation of likes(if correlated) with count ratios of nearby categories
list_dist_c1_r_corr = []
for i in range(3,len(df_count_c1_r.columns)-2):
    var = df_count_c1_r.columns[i]
    corr = spearmanr(df_count_c1_r.iloc[:,i],df_count_c1_r.iloc[:,-1])
    r = corr[0]
    p = corr[1]
    list_dist_c1_r_corr.append([var,r,p])
df_sCorr_c1_r_dist = pd.DataFrame(list_dist_c1_r_corr,columns = ['Category','rank_r','test_p'])
#%%
df_sCorr_c1_r_dist.to_csv('C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/Report/c1_count_ratio_rank_correlation.csv',encoding='utf-8-sig')

#%%
#Select the features

df_X = df_count_c2.iloc[:,3:401]
df_X['groups']=df_c1_class['D_class']
df_0counts = pd.DataFrame(index = df_c1_D_grouped.index,columns = df_X.columns)

#%%
#Logic matrics deciding the frequency of the venues of each group in each c2 category
for i in range(len(df_0counts.index)):
    df = df_X[df_X['groups']==df_0counts.index[i]]
    for j in range(len(df.columns)-1):
        Zcount = df[df.iloc[:,j]==0].count()[0]
        Zcount0 = df[df.iloc[:,j]>=0].count()[0]
        if Zcount >= 0.5*Zcount0:
            df_0counts.iloc[i,j] = 0
        else:
            df_0counts.iloc[i,j] = 1

#%%
df_0counts = df_0counts.iloc[:,0:398].transpose()

#%%
df_0counts['rowSum'] = df_0counts.sum(axis=1)
#%%
df_0counts2 = df_0counts[df_0counts['rowSum']>=5]
#%%
selected_col = list(df_count_c2.columns[0:3]) + list(df_0counts2.index) + ['likes']
len(selected_col)

#%%
#find the hierarchical category of the selected
df_c2_cat_s = pd.DataFrame(columns=df_cat_hk.columns)
for c2 in selected_col:
    df_c2_cat_rows = df_cat_hk[df_cat_hk['c2']==c2]
    df_c2_cat_s = df_c2_cat_s.append(df_c2_cat_rows)
df_c2_cat_s = df_c2_cat_s[['c2','c1']]
df_c2_cat_s.drop_duplicates(inplace=True)

#%%
df_count_c2['likes']=df_res['likes']
df_count_c2s = df_count_c2[selected_col]

#%%
df_count_c2s_values = df_count_c2s.iloc[:,3:len(df_count_c2s.columns)]
df_count_c2s_values['groups'] = df_c1_class['D_class']
df_c1_D_grouped_c2 = df_count_c2s_values.groupby('groups').mean()

#%%
#c2 selected categories correlation
#rank correlation between nearby venues count of the selected c2 categories and likes
list_dist_c2s_corr = []
for i in range(3,len(df_count_c2s.columns)-1):
    var = df_count_c2s.columns[i]
    corr = spearmanr(df_count_c2s.iloc[:,i],df_count_c2s.iloc[:,-1])
    r = corr[0]
    p = corr[1]
    list_dist_c2s_corr.append([var,r,p])
df_sCorr_c2s = pd.DataFrame(list_dist_c2s_corr,columns = ['Category','rank_r','test_p'])

#%%
df_count_c2s_values = df_count_c2s.iloc[:,3:len(df_count_c2s.columns)-1]
df_count_c2s_catMeans = df_count_c2s_values.mean(axis = 0)
df_count_c2s_catStd = df_count_c2s_values.std(axis = 0)

df_sCorr_c2s['catNumberMeans'] = list(df_count_c2s_catMeans)
df_sCorr_c2s['catNumberStd'] = list(df_count_c2s_catStd)
df_sCorr_c2s['Num_Std/Avg'] = df_sCorr_c2s['catNumberStd']/df_sCorr_c2s['catNumberMeans']

#%%

df_sCorr_c2s.to_csv('C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/Report/c2s_count_rank_correlation.csv',encoding='utf-8-sig')

#%%
#rank correlation between nearby venues count ratio of the selected c2 categories and likes

#Get the ratios of the c2 categories
df_count_c2s_r = df_count_c2s.iloc[:,0:len(df_count_c2s.columns)]

#calculate ratios of the counts
for i in range(len(df_count_c2s_r)):
    row = df_count_c2s_r.iloc[i,3:len(df_count_c2s.columns)-1]
    rowSum = sum(row)
    if rowSum>0:
        df_count_c2s_r.iloc[i,3:len(df_count_c2s.columns)-1] = row/rowSum
    else:
        df_count_c2s_r.iloc[i,3:len(df_count_c2s.columns)-1] = row
    if i%100 == 0:
        print(datetime.now().strftime("%H:%M:%S"),'transform loop:',i)
        
#%%
#Rank correlation
list_dist_c2s_r_corr = []
for i in range(3,len(df_count_c2s.columns)-1):
    var = df_count_c2s_r.columns[i]
    corr = spearmanr(df_count_c2s_r.iloc[:,i],df_count_c2s_r.iloc[:,-1])
    r = corr[0]
    p = corr[1]
    list_dist_c2s_r_corr.append([var,r,p])
df_sCorr_c2s_r = pd.DataFrame(list_dist_c2s_r_corr,columns = ['Category','rank_r','test_p'])

#%%
df_count_c2s_r_values = df_count_c2s_r.iloc[:,3:len(df_count_c2s.columns)-1]
df_count_c2s_r_catMeans = df_count_c2s_r_values.mean(axis = 0)
df_count_c2s_r_catStd = df_count_c2s_r_values.std(axis = 0)

df_sCorr_c2s_r['catRatioMeans'] = list(df_count_c2s_r_catMeans)
df_sCorr_c2s_r['catRatioStd'] = list(df_count_c2s_r_catStd)
df_sCorr_c2s_r['Ratio_Std/Avg'] = df_sCorr_c2s_r['catRatioStd']/df_sCorr_c2s_r['catRatioMeans']

#%%
df_sCorr_c2s.to_csv('C:/Users/User1/Desktop/projects/StatLearning/IBM_DS_FinalProject/Report/c2s_countRatio_rank_correlation.csv',encoding='utf-8-sig')
