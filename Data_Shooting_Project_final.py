#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from geopy import distance
import folium
from folium.plugins import FastMarkerCluster
import plotly.express as px
import json


# In[2]:


df = pd.read_csv("C:/Users/13474/Downloads/NYPD_Shooting(Cleaned_final).xlsb.csv")
#df = pd.read_csv("./NYPD_Shooting(Cleaned_final).xlsb.csv")
df.rename(columns={df.columns[1]: 'Occurance Date'}, inplace=True)
df['Occurance Date'] = pd.to_datetime(df['Occurance Date'])
df['Occurance Time'] = pd.to_datetime(df['Occurance Time'])
print(f'Shape: {df.shape}')
print(df.dtypes)
df.head(3)


# In[7]:


# Nominatim is too slow to get all the zip codes
# Nominatim takes about 1 second per location to get zip code.
# from geopy.geocoders import Nominatim, ArcGIS
# geolocator = Nominatim(user_agent='df')
# example_address = geolocator.reverse((df.Latitude[3], df.Longitude[3]))
# print(example_address)
# example_address.raw #.get('postcode')


# In[10]:


# DO NOT RE-RUN THIS unless you want to wait for 20 minutes 
# and have an API key for ArgGIS
# RUN ONLY ONCE to get zip codes for each latitude and longitude
# Zip codes are saved in a new csv file with the other data. 
# Must sign up for ArcGIS account at https://developers.arcgis.com/ and create API Key

# Use ArcGIS to get zip codes from latitude/longitude
from arcgis.geocoding import reverse_geocode
from arcgis.geocoding import geocode
from arcgis.geometry import Geometry
from arcgis.gis import GIS
import pandas as pd

API_KEY = 'AAPKae6dd3da862141fbb24021699a69cc2fHI53r0GpT9YTjjitDRzHMtv4Axf7S0B2JUmaGg5c6khX0NgQ9nxhTNJtRTVW7UTa'
gis = GIS(api_key=API_KEY)

i = 0
def get_zip(row, lon_field, lat_field, verbosity=100):
    global i
    location = reverse_geocode((Geometry({"x":float(row[lon_field]), "y":float(row[lat_field]), "spatialReference":{"wkid": 4326}})))
    address = location.get('address', {})
    zip = str(address.get('Postal', ''))
    if i%verbosity == 0:
        print(f"{i:8d}  {address.get('Match_addr', 'Missing Address')}")
    i += 1
    return zip

df['zip'] = df.apply(get_zip, axis=1, lat_field='Latitude', lon_field='Longitude')
df.to_csv('./NYPD_Shooting_with_zip.csv')
df.head()


# In[3]:


df = pd.read_csv('./Downloads/NYPD_Shooting_with_zip.csv')
df['Occurance Date'] = pd.to_datetime(df['Occurance Date'])
df['Occurance Time'] = pd.to_datetime(df['Occurance Time'])
print(f"Number of missing zip:  {df['zip'].isna().sum()}")
print("Remove rows with missing zip code")
df = df[~df['zip'].isna()]
df['zip'] = df['zip'].astype(int).astype(str)
df.head(3)


# In[4]:


df2 = df.groupby([df['Occurance Date'].dt.year, 'zip']).size()
df2 = df2.to_frame()
df2.index.names = ['Year', 'postalCode']
df2 = df2.reset_index()
df2.rename(columns={0:'Shootings'}, inplace=True)
df2.head(10)


# In[5]:


nycmap = json.load(open("./Downloads/nyc_zip.geojson"))


# In[6]:


# create choropleth maps for each year, showing shootings in each zip code in NYC
years = df2['Year'].unique()
for year in years:
    fig = px.choropleth_mapbox(
        df2[df2['Year']==year],
        geojson=nycmap,
        locations="postalCode",
        featureidkey="properties.postalCode",
        color="Shootings",
        color_continuous_scale="bluered", #"viridis",
        mapbox_style="carto-positron",
        zoom=9, center={"lat": 40.7, "lon": -73.9},
        opacity=0.7,
        hover_name="postalCode", 
        title=f'New York City Shootings in {year}',
        width=700,
        height=700, 
        range_color=(0, 100)
        )
    #fig.set_title(f'Shootings in {year}') #, fontsize=16)

    fig.show()


# In[ ]:



    


# In[7]:


folium_map= folium.Map(
    location=[40.869058,-73.879632],zoom_start=10,tiles='CartoDB dark_matter')
FastMarkerCluster(df[['Latitude', 'Longitude']].values.tolist()).add_to(folium_map)
folium.LayerControl().add_to(folium_map) 
for row in df.iterrows():
    row=row[1]
    folium.CircleMarker(location=(row["Latitude"],
                                  row["Longitude"]),
                        radius= .5,
                        color="#D22529",
                        popup=row['Location Description'],
                        fill=False).add_to(folium_map)
folium_map


# In[ ]:





# In[8]:


victim_pct = df['Victim Age Group'].value_counts()
victim_pct


# In[9]:


perpetrator_pct = df['Perpetrator Age Group'].value_counts()
perpetrator_pct


# In[10]:


murder_flag = df['Murder Flag'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%'
murder_flag
# False = Victim survived the shooting
# True = Victim died as a result of shooting


# In[11]:


#murder_plot = df['Murder Flag'].value_counts().plot(kind='pie', autopct='%.2f%%')
#plt.show()

labels = ["True", "False"]
colors = ['#Ee149f', '#3f17e8']
sizes = [19.1, 80.9]
explode = (0, 0.1) # explode true
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode,labels=labels, shadow=True, colors=colors,autopct='%1.1f%%')
ax1.axis('equal')
plt.show


# In[12]:


sns.countplot(df['Murder Flag'])
plt.show()


# In[13]:


df.groupby('Occurance Date')['Occurance Date'].value_counts()


# In[14]:


df[['Boro', 'Murder Flag']].describe()


# In[15]:


shooting_inc = df['Boro'].value_counts()
shooting_inc


# In[16]:


Brooklyn_inc = df.query('Boro=="Brooklyn"')
Brooklyn_inc['Murder Flag'].value_counts()


# In[17]:


# group by month; count number of shooting incidents by month
df2 = df.groupby([df['Occurance Date'].dt.year, df['Occurance Date'].dt.month, 'Boro']).size()
df2 = df2.to_frame()
df2.index.names = ['Year', 'Month', 'Boro']
df2 = df2.reset_index()


# In[18]:


df2['Date'] = pd.to_datetime(df2[['Year', 'Month']].assign(DAY=1))
df2.drop(columns=['Year', 'Month'], inplace=True)
df2.rename(columns={0:'Incidents'}, inplace=True)
df2 = df2[['Date', 'Boro', 'Incidents']]
df2


# In[19]:


# convert 'Date' column from datetime64[ns] data type to datetime datatype 
print(df2.dtypes)
#df2['Date'] = pd.to_datetime(df2['Date'])
#df2['just_date'] = df2['Date'].dt.date
#print(df2.dtypes)
#df2['just_date'][0]


# In[20]:


# Only plot for 1 year
for year in range(2006, 2021):
    df_1y = df2[(df2['Date']<datetime(year+1, 1, 1)) & (df2['Date']>=datetime(year, 1, 1))]
    df_1y['Date'] = df_1y['Date'].dt.date
    pivot = pd.pivot_table(data=df_1y, index=['Date'], columns=['Boro'])
    ax = pivot.plot.bar(stacked=True, figsize=(10,7), title=str(year))
    ax.plot()


# In[26]:


import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from pylab import rcParams

import sklearn
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm


# In[27]:


np.set_printoptions(precision=4, suppress=True)
plt.figure(figsize=(10,3))
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-whitegrid')


# In[28]:


df.head(3)


# In[29]:


df.columns['Occurance Date', 'Occurance Time', 'Boro', 'Location Description', 'Perpetrator Age Group',
          'Perpetrator Sex', 'Perpetrator Race', 'Murder Flag', 'Victim Age Group', 'Victim Sex', 
          'Victim_Race', 'Latitude', 'Longitude', 'zip']

X = df.ix[:, (5, 8, 9, 10, 11, 13, 14, 15)].values
Y = df.ix[:, (12)].values


# In[ ]:




