'''
merch_lat: latitude location of merchant
merch_long: longitude location of merchant
new features:
1. number of eateries (restaurants, cafe, bars) - higher number suggests a more densely populated area,
   may have impact on probability of fraud cases
2. average ratings of these eateries - higher rated areas may suggest a wealthier/ more established region which also
   have an impact on probability of fraud cases
3. number of transportation facilities - connectivity of the region, may give a proxy on how secluded/ crowded the area is
   (airport, bus_station, light_rail_station, subway_station, taxi_stand, train_station, transit_station)
'''

import csv
import googlemaps
import pandas as pd
import time

# Replace with your own API key
api_key = ''

# Create a Google Maps client
gmaps = googlemaps.Client(key=api_key)

# Set radius (in terms of meters)
radius = 3000

# Read the DataFrame containing latitude and longitude
locations_df = pd.read_csv('data/location_data/lat_long_train.csv', header=0)

# Rename the first column to 'index'
locations_df.rename(columns={locations_df.columns[0]: 'index'}, inplace=True)

# Set 'index' as the index column
locations_df.set_index('index', inplace=True)

# Create a Pandas Excel writer
excel_writer = pd.ExcelWriter('google_api_features.xlsx', engine='xlsxwriter')

start_index = 0
processed_locations = 0

# Loop through the DataFrame starting from the specified index
for index, row in locations_df.iloc[start_index:].iterrows():
    
    # Specify the location (latitude and longitude) from the DataFrame
    location = (row['merch_lat'], row['merch_long'])
    
    # Define place types for eateries and transportation facilities
    eatery_types = ['cafe', 'bar', 'restaurant']
    transportation_types = [
        'airport', 'bus_station', 'light_rail_station', 'subway_station',
        'taxi_stand', 'train_station', 'transit_station'
    ]

    # Initialize variables for storing ratings and counts
    location_ratings = []
    num_restaurants = 0
    num_transport_facilities = 0

    # Function to perform nearby search for a specific type of places
    def perform_search(place_types, count_variable, rating_variable=None):
        next_page_token = None
        while True:
            places = gmaps.places_nearby(location=location, radius=radius, type=place_types, page_token=next_page_token)
            for place in places['results']:
                if rating_variable is not None:
                    rating = place.get('rating', 'N/A')
                    if rating != 'N/A':
                        rating_variable.append(rating)
                count_variable += 1
            next_page_token = places.get('next_page_token')
            if not next_page_token:
                break
            time.sleep(2.5)

        return count_variable

    # Perform search for eateries
    num_restaurants = perform_search(eatery_types, num_restaurants, location_ratings)

    # Calculate the mean rating for eateries
    mean_rating = sum(location_ratings) / len(location_ratings) if location_ratings else 0

    # Perform search for transportation facilities
    num_transport_facilities = perform_search(transportation_types, num_transport_facilities)

    # Append the results to the DataFrame
    locations_df.at[index, 'Avg_Rating_Restaurant'] = mean_rating
    locations_df.at[index, 'Num_Restaurants'] = num_restaurants
    locations_df.at[index, 'Num_Transport_Facilities'] = num_transport_facilities

    # Increment the processed locations counter
    processed_locations += 1

    print(f'Processed {processed_locations} locations')

# Reset the index to include it as a column
locations_df.reset_index(inplace=True)

# Save the DataFrame to an Excel file, including the index
locations_df.to_excel(excel_writer, sheet_name='NewFeatures', index=False)

# Save the Excel file
excel_writer.save()