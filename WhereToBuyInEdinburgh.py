#!/usr/bin/env python
# coding: utf-8

# # Neighbourhood insights for buying a home in Edinburgh

# ## Table of Contents  
# - [Introduction](#introduction)
# - [Business problem](#introduction.business_problem)
#     - [Static ranking system](#introduction.static_ranking_system)
#     - [Dynamic ranking system](#introduction.dynamic_ranking_system)
# - [Data](#data)
#     - [Overview](#data.overview)
#     - [Postcode data](#data.postcode_data)
#     - [Define neighbourhoods](#data.define_neighbourhoods)
#     - [Foursquare venue data](#data.foursquare_venue_data)
#     - [Rightmove property sale price data](#data.rightmove_property_sale_price_data)

# ## Introduction <a name="introduction"/>
# 
# ### Business problem <a name="introduction.business_problem"/>
# 
# A major estate agent in **Edinburgh** would like to provide more information about neighbourhoods to their clients. Local amenities are very important in deciding to buy a home, in addition to the property itself. This information, however, is not provided by the estate agent to the same quality as details about the property itself. **Presenting valuable insights about local amenities to potential buyers could attract more customers,** particularly those new to the city.
# 
# The insights about local amenities should be provided to the home buyer in a format that can be directly used in making their decision. Firstly, the information should be easy and quick to understand. Secondly, it should allow intuitive comparison between available properties. Thirdly, it should be objective truth, based on statistics, and not a biased opinion.
# 
# **This project aims to provide a solution for informing home buyers about the neighbourhoods in Edinburgh.** We will achieve this by generating two ranking systems. Firstly, **a static ranking system** for common preference types, such as favouring nightlife venues over parks or grocery stores over restaurants. Secondly, **a dynamic ranking system** that provides a ranking of neighbourhoods based on the client's personal preferences and purchase price range.
# 
# ### Static ranking system <a name="introduction.static_ranking_system"/>
# 
# The static ranking system will be created using **k-means clustering** of neighbourhoods based on the local amenities and identifying preference categories in the resulting clusters.
# 
# ### Dynamic ranking system <a name="introduction.dynamic_ranking_system"/>
# 
# The dynamic ranking system will be ranking how well each neighbourhood matches the ideal neighbourhood based on user preferences. User input will be quantified relative to the **distribution of each feature across all neighbourhoods**. 

# ## Data <a name="data"/>
# 
# ### Overview <a name="data.overview"/>
# 
# For this project we will need data on venues and amenities across Edinburgh and property sale price data. We will acquire the data on venues and amenities using Foursqare API. The sale price data will be acquired using web scraping on Rightmove website.
# 
# The neighbourhoods will be overlapping circular grid fields positioned uniformly across Edinburgh. All venues will be assigned to and mean property sale prices will be computed for these artifical neighbourhoods. These neighbourhoods with the resulting features will be the subject of the statistical and machine learning methods.

# ### Postcode data <a name="data.postcode_data"/>
# 
# Acquire Edinburgh postcode data from doogal.co.uk

# Download and load Edinburgh Postcode table that contains latitude and longitude

# In[1]:


import pandas as pd
import os

if not os.path.isfile('EdinburghPostcodes.csv'):
    # If the file is not available on disk, download it
    urlretrieve ('https://www.doogal.co.uk/AdministrativeAreasCSV.ashx?district=S12000036', 'EdinburghPostcodes.csv')
df_postcodes = pd.read_csv('EdinburghPostcodes.csv', usecols=['Postcode', 'Latitude', 'Longitude', 'In Use?'])
# Only keep postcodes that are in use
df_postcodes.drop(df_postcodes[df_postcodes['In Use?'] == 'No'].index, inplace=True)
df_postcodes.drop(columns=['In Use?'], inplace=True)
df_postcodes.reset_index(drop=True, inplace=True)
# Make column names lower caps
df_postcodes.columns = map(str.lower, df_postcodes.columns)
# Display DataFrame
df_postcodes.head()


# ### Define neighbourhoods <a name="data.define_neighbourhoods"/>
# 
# We will define uniformly distributed points across Edinburgh to serve as centers for neighbourhoods. These neighbourhood center coordinates and postcodes are then used to collect data specific to each location. This will allow later analysis of spatial distributions of venues and home prices.

# Define points on a rectangular grid with 250 m spacing in 3 km radius of Edinburgh Castle.

# In[2]:


import geopy.distance
import numpy as np
# import folium

neighbourhood_spacing = 250 # in meters
max_distance = 4000 # in meters

# Define central latitude and longitude as the position of Edinburgh Castle
edinburgh_castle_postcode = 'EH1 2NG'
central_latitude = df_postcodes[df_postcodes.postcode == edinburgh_castle_postcode]['latitude'].values
central_longitude = df_postcodes[df_postcodes.postcode == edinburgh_castle_postcode]['longitude'].values

# Calculate the distance in meters between two points 0.1 latitude apart in Edinburgh
lat01_in_m = geopy.distance.distance((central_latitude, central_longitude), 
                                      (central_latitude + 0.1, central_longitude)).km * 1000
lon01_in_m = geopy.distance.distance((central_latitude, central_longitude), 
                                      (central_latitude, central_longitude + 0.1)).km * 1000
# Calculate distance in latitude and longitude for neighbourhood spacing
neigh_spacing_lat = (neighbourhood_spacing * 0.1) / lat01_in_m
neigh_spacing_lon = (neighbourhood_spacing * 0.1) / lon01_in_m
# Calculate maximum and minimum coordinates that will be in range of central coordinates
max_lat = central_latitude + ((max_distance * 0.1) / lat01_in_m)
min_lat = central_latitude - ((max_distance * 0.1) / lat01_in_m)
max_lon = central_longitude + ((max_distance * 0.1) / lon01_in_m)
min_lon = central_longitude - ((max_distance * 0.1) / lon01_in_m)

# Calculate possible latitude positions for all neighbourhoods
neigh_lats = np.arange(min_lat, max_lat, neigh_spacing_lat)
# Calculate possible longitude positions for all neighbourhoods
neigh_lons = np.arange(min_lon, max_lon, neigh_spacing_lon)

# Assign latitude and longitude values for each neighbourhood
# Only keep use positions within specified radius of central coordinates
neighbourhood_lat_lon = []
for lat in neigh_lats:
    for lon in neigh_lons:
        distance_from_center = geopy.distance.distance((central_latitude, central_longitude), 
                                                       (lat, lon)).km * 1000
        if distance_from_center <= max_distance:
            neighbourhood_lat_lon.append((lat, lon))
            
# Arrange neighbourhoods in a DataFrame
lat, lon = list(zip(*neighbourhood_lat_lon))
df_neighbourhoods = pd.DataFrame({'latitude': lat, 'longitude': lon})
# Display neighbourhood DataFrame
df_neighbourhoods.head()


# Assign a postcode closest to the center to each of the neighbourhoods.

# In[3]:


if os.path.isfile('NeighbourhoodsWithPostcodes.p'):
    # If the script has already been run, load the result from disk
    df_neighbourhoods = pd.read_pickle('NeighbourhoodsWithPostcodes.p')
else:
    # Find closest postcode to the center of each neighbourhood
    postcodes = []
    for coords in zip(df_neighbourhoods['latitude'], df_neighbourhoods['longitude']):
        distances = []
        for pc_coords in zip(df_postcodes['latitude'], df_postcodes['longitude']):
            distances.append(geopy.distance.distance(coords, pc_coords).km)
        postcodes.append(df_postcodes.loc[np.argmin(distances), 'postcode'])
    # Append postcode list to the neighbourhoods DataFrame
    df_neighbourhoods['postcode'] = postcodes
    # Drop neighbourhoods where assigned postcode more than half the neighourhood spacing away from center
    distances = []
    for lat, lon, postcode in zip(df_neighbourhoods['latitude'], df_neighbourhoods['longitude'], df_neighbourhoods['postcode']):
        # Find the coordinates for the postcode
        pc_lat = float(df_postcodes.loc[df_postcodes['postcode'] == postcode, 'latitude'])
        pc_lon = float(df_postcodes.loc[df_postcodes['postcode'] == postcode, 'longitude'])
        distances.append(geopy.distance.distance((lat, lon), (pc_lat, pc_lon)).km * 1000)
    df_neighbourhoods.drop(df_neighbourhoods[np.array(distances) > neighbourhood_spacing / 2.0].index, inplace=True)
    # Reset index
    df_neighbourhoods.reset_index(drop=True, inplace=True)
    # Store resulting DataFrame on disk
    df_neighbourhoods.to_pickle('NeighbourhoodsWithPostcodes.p')


# Display neighbourhood positions with their central postcodes on a map.

# In[4]:


# m = folium.Map(location=[float(central_latitude), float(central_longitude)], zoom_start=12)
# for lat, lon, postcode in zip(df_neighbourhoods['latitude'], df_neighbourhoods['longitude'], df_neighbourhoods['postcode']):
#     # Draw a circle around the center of a neighbourhood with radius of neighbourhood spacing
#     folium.Circle(
#        location=(lat, lon),
#        radius=neighbourhood_spacing,
#        color='crimson', 
#        weight=1, 
#     ).add_to(m)
#     # Find the coordinates for the postcode
#     pc_lat = float(df_postcodes.loc[df_postcodes['postcode'] == postcode, 'latitude'])
#     pc_lon = float(df_postcodes.loc[df_postcodes['postcode'] == postcode, 'longitude'])
#     # Draw small blue circles at postcode locations
#     folium.Circle(
#        location=(pc_lat, pc_lon),
#        radius=10,
#        color='blue', 
#        weight=2, 
#     ).add_to(m)
# #     # Add a popup to get Postcode
# #     folium.Marker([pc_lat, pc_lon], popup='<i>{}</i>'.format(postcode)).add_to(m)
# m


# ### Foursquare venue data <a name="data.foursquare_venue_data"/>
# 
# Foursquare only returns 50 venues per request, therefore, to get detailed data on venues and amenities each neighbourhood, we need to use multiple requests.
# 
# We will compute uniformly distributed positions across Edinburgh that will be the center points of our Foursquare API calls to collect locations of all the venues in categories of interest (e.g. cafe, turkish restaurant, night club, grocery store, gym).
# 
# We will arrange these examples into a `pandas.DataFrame` that will contain venue category, rating, latitude and longitude.
# 
# The interface with the Foursqare API will be re-using the code in [my Assessment 3 submission](https://nbviewer.jupyter.org/github/sanntann/Coursera_Capstone/blob/master/Assessment_3.ipynb)

# ### Rightmove property sale price data <a name="data.rightmove_property_sale_price_data"/>
# 
# Rightmove is a major UK property website. They provide a list of sale price data going back several years. 
# 
# For some of the properties on the list there is a link to a post on Rightmove website and information on the type of the property, including number of bedrooms. As the latter information is a major determinant of sale price, we will only use data on properties where this information is available. This will allow more client preference specific estimation of mean property prices in neighbourhoods.
# 
# Rightmove provides the address for each property, including the postcode. We will use a Edinburgh postcode latitude and longitude dataset to approximate the latitude and longitude of each property. This will allow assigning each property to one of the artificial neighbourhoods.

# In[5]:


from urllib.request import urlretrieve
from requests import get
from bs4 import BeautifulSoup
import sys


# Define function the extract data for properties from a BeautifulSoup of a html webpage

# In[6]:


def get_property_type_from_sold_property_page(url):
    soup = BeautifulSoup(get(url).text, 'html.parser')
    return soup.find(id='propertydetails').find_all('h2')[1].text


def get_property_data_from_soup(soup):
    # Extract data from the http soup
    date = []
    address = []
    bedrooms = []
    price = []
    property_type = []
    for soup_property in soup.find_all(class_='soldDetails'):
        # Skip properties for which there is no link to post on RightMove website
        if not soup_property.find(class_='soldAddress').has_attr('href'):
            continue
        else:
            url = soup_property.find(class_='soldAddress')['href']
        # Skip properties for which there is no number of bedrooms information
        if len(soup_property.find(class_='noBed').text) == 0:
            continue
        # Collect data for the property
        date.append(soup_property.find(class_='soldDate').text)
        address.append(soup_property.find(class_='soldAddress').text)
        bedrooms.append(soup_property.find(class_='noBed').text)
        price.append(soup_property.find(class_='soldPrice').text)
        # Attempt to collect property type
        try:
            property_type.append(get_property_type_from_sold_property_page(url))        
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            property_type.append('')
            print('Error when collecting property type.')
            print(sys.exc_info()[0])
    # Format data into pandas.DataFrame
    df = pd.DataFrame({'date': date, 
                       'address': address, 
                       'bedrooms': bedrooms, 
                       'property_type': property_type, 
                       'price': price}, 
                      columns=['date', 'address', 'bedrooms', 'property_type', 'price'])
    # Sort the DataFrame by date as well as address
    df.sort_values(['date', 'address'], ascending=[False, True], inplace=True)
    
    return df


# Create a class to manage web scraping rate

# In[7]:


from time import time, sleep

class RateManager(object):
    
    def __init__(self, min_interval, max_interval):
        """
        min_interval - float - minimum delay between calls (in seconds)
        max_interval - float - maximum delay between calls before notification (in seconds)
        """
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.checkpoint = None
        
    def continue_when_ready(self, sleep_interval=0.1, print_interval=False):
        # This is in case of first call to continue_when_ready
        if self.checkpoint is None:
            self.checkpoint = time()
            return None
        # Check if max_interval has been surpassed
        if time() - self.checkpoint > self.max_interval:
            if print_interval:
                print('Interval duration: {}'.format(time() - self.checkpoint))
            self.checkpoint = time()
            return 'timeout'
        # If not over max_interval, wait until min_interval is reached
        if print_interval:
            print('Interval duration: {}'.format(time() - self.checkpoint))
        while time() - self.checkpoint < self.min_interval:
            sleep(sleep_interval)
        self.checkpoint = time()
        return 'intime'


# Create a function to construct rightmove.co.uk http address for specific house price search

# In[8]:


from selenium import webdriver

def rightmove_houseprice_bypostcode_url(radius=250, years=2):
    # Ensure input arguments are allowed
    if radius == 250:
        radius = '0.25'
    elif radius == 500:
        radius = '0.5'
    else:
        raise ValueError('No http address key set for radius {}'.format(radius))
    if not isinstance(years, int) or years > 6 or years < 1:
        raise ValueError('years argument must be int in range 1 to 6.')
    url = 'https://www.rightmove.co.uk/house-prices/detail.html?' +           'country=scotland&locationIdentifier=POSTCODE%5E1071308' +           '&searchLocation=EH1+2NG&propertyType=3&radius={}&year={}'.format(radius, years) +           '&referrer=listChangeCriteria'
    
    return url


def append_specify_page_index_to_houseprice_url(url, page_nr):
    # Ensure input arguments are allowed
    if page_nr > 40:
        raise ValueError('page_nr argument not allowed over 40.')
    # Append page index key value
    url = url + '&index={}'.format(page_nr * 25)
    
    return url


def rightmove_houseprice_url(postcode, radius=250, years=2):
    url = rightmove_houseprice_bypostcode_url(radius=radius, years=years)
    options = webdriver.chrome.options.Options()
    options.add_argument('headless')
    options.add_argument('window-size=1200x600')
    chrome_prefs = {}
    options.experimental_options["prefs"] = chrome_prefs
    chrome_prefs["profile.default_content_settings"] = {"images": 2}
    chrome_prefs["profile.managed_default_content_settings"] = {"images": 2}
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    driver.find_element_by_id('searchLocation').clear()
    driver.find_element_by_id('searchLocation').send_keys(postcode)
    driver.find_element_by_id('housePrices').click()
    url = driver.current_url
    driver.quit()
    
    return url


# Acquire residential property sales prices from RightMove.
# 
# There are likely duplicates in the resulting DataFrame. These will be dealt with later.

# In[10]:


if os.path.isfile('EdinburghPropertiesRaw.p'):
    # If the script has already been run, load the result from disk
    df_property = pd.read_pickle('EdinburghPropertiesRaw.p')
else:
    # Create empty pandas.DataFrame to append new data to
    df_property = pd.DataFrame({'date': [], 
                                'address': [], 
                                'bedrooms': [], 
                                'property_type': [], 
                                'price': [], 
                                'search_postcode': []}, 
                               columns=['date', 
                                        'address', 
                                        'bedrooms', 
                                        'property_type', 
                                        'price', 
                                        'search_postcode'])

    # Use RateManager to avoid overwhelming the website
    rate_manager = RateManager(min_interval=5, max_interval=30)
    max_timeouts = 10
    timeout_count = 0

    # Loop through all postcodes and all possible page indices
    df_prev_property_list = pd.DataFrame({})
    for i, postcode in enumerate(df_neighbourhoods['postcode']):
        print('--------Getting price data for postcode {} --- {} of {} ---'.format(postcode, i + 1, len(df_neighbourhoods['postcode'])))
        url = None
        while url is None:
            try:
                url = rightmove_houseprice_url(postcode, radius=(neighbourhood_spacing), years=2)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                print('Failed to get url. Trying again in 1 minute.')
        for page_nr in range(40):
            full_url = append_specify_page_index_to_houseprice_url(url, page_nr)
            print('Visiting webpage:\n' + full_url)
            # Make sure webpage is not visited too often and that it is not blocking
            if rate_manager.continue_when_ready(print_interval=True) == 'timeout':
                timeout_count += 1
                if timeout_count > max_timeouts:
                    raise RuntimeError('Too many timeouts.')
            # Get website html as BeautifulSoup
            soup = BeautifulSoup(get(full_url).text, 'html.parser')
            # Check if there is a property price data list on this page
            if len(soup.find_all(class_='soldDetails')) == 0:
                print('No properties listed on this page. Stopping page index iteration.')
                break
            df_next_property_list = get_property_data_from_soup(soup)
            # If the new DataFrame is equal to the previous one, stop checking further indices
            if df_prev_property_list.equals(df_next_property_list):
                print('Property list repeated. Stopping page index iteration.')
                break
            else:
                # Append the new property list to main property list and store to check against next one
                df_prev_property_list = df_next_property_list
                print('Got {} properties.'.format(df_next_property_list.shape[0]))
                df_next_property_list['search_postcode'] = postcode
                df_property = df_property.append(df_next_property_list)
        # Save collected property data to disk
        df_property.to_pickle('EdinburghPropertiesRaw.p')
# Report number of properties in raw web scarping result
print('Collected total of {} properties from RightMove.'.format(df_property.shape[0]))


# Format `df_property` DataFrame and keep only the essential information

# In[ ]:


# Define function for extracting number of bedrooms and
# general property type from raw property type data
def get_bedrooms_and_type(raw_property_type):
    pos = raw_property_type.find(' bedroom ')
    raw_property_type = raw_property_type.replace(' bedroom ', ' ')
    bedrooms = raw_property_type[:pos].strip()
    property_type = raw_property_type[pos:].strip()
    
    return bedrooms, property_type

# Reindex DataFrame
df_property.reset_index(drop=True, inplace=True)

# Remove duplicates
df_property.drop_duplicates(subset=['date', 'address', 'price'], keep='first', inplace=True)
# Reindex DataFrame
df_property.reset_index(drop=True, inplace=True)

# Find property_type 'Studio flat' and set it to 1 bedroom flat
idx_studio_flat = df_property['property_type'] == 'Studio flat'
df_property['property_type'].loc[idx_studio_flat] = '1 bedroom flat'

# Remove properties for which property_type is not in correct format
indices = [i for i, x in enumerate(df_property['property_type']) if not (' bedroom ' in x)]
df_property.drop(indices, axis=0, inplace=True)
# Reindex DataFrame
df_property.reset_index(drop=True, inplace=True)

# Extract number of bedrooms and general property type from property_type values
bedrooms, property_type = zip(*[get_bedrooms_and_type(x) for x in df_property['property_type']])
df_property['bedrooms'] = list(map(int, bedrooms))
df_property['property_type'] = property_type

# Rename all flat-like property_types to flats
func = lambda x: 'flat' if 'flat' in x or 'apartment' in x or 'penthouse' in x else x
df_property['property_type'] = df_property['property_type'].apply(func)
# Rename all house-like property_types to house
func = lambda x: 'house' if 'house' in x or 'villa' in x or 'duplex' in x or 'bungalow' in x or 'cottage' in x else x
df_property['property_type'] = df_property['property_type'].apply(func)

# Remove all other property_types than flat or house
idx = (df_property['property_type'] != 'flat') & (df_property['property_type'] != 'house')
df_property.drop(df_property[idx].index, axis=0, inplace=True)
# Reindex DataFrame
df_property.reset_index(drop=True, inplace=True)

# Only keep postcode from address
func = lambda x: ' '.join(x.split()[-2:])
df_property['address'] = df_property['address'].apply(func)
# Rename address column to postcode
df_property.rename(columns={'address': 'postcode'}, inplace=True)

# Print remaining number of properties
print('Property sale price samples remaining after filtering the data: {}'.format(df_property.shape[0]))


# Add longitude and latitude data into `df_property` based on postcode

# In[ ]:


# Merge on latitude and longitude values
df_property = df_property.merge(df_postcodes, how='left', on='postcode')
# Drop rows where latitude and longitude were not available for postcode
df_property.dropna(inplace=True)
# Reindex DataFrame
df_property.reset_index(drop=True, inplace=True)
# # Drop postcode column
# df_property.drop('postcode', axis='columns', inplace=True)
print('Property sale price samples remaining that have \n' + 
      'latitude and logitude values: {}'.format(df_property.shape[0]))
df_property.head()


# In[ ]:




