import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, OPTICS
import hdbscan
import requests
import re
import time
import os
import tempfile
from datetime import datetime, timedelta, date
from io import StringIO, BytesIO
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import folium
from folium.plugins import MarkerCluster, HeatMap, Fullscreen
from branca.colormap import LinearColormap
import streamlit.components.v1 as components
import altair as alt
import json

# Check if geospatial dependencies are available
try:
    import geopandas as gpd
    from shapely.geometry import Point, shape
    from geopy.distance import geodesic
    HAVE_GEO_DEPS = True
except ImportError:
    HAVE_GEO_DEPS = False
    st.warning("Geospatial dependencies not available. Spatial join functionality will be limited.")

# Set page config
st.set_page_config(
    page_title="Fire Investigation Tool",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Fire Investigation Tool")
st.markdown("---")

# Initialize session state for results and selected cluster
if 'results' not in st.session_state:
    st.session_state.results = None
if 'selected_cluster' not in st.session_state:
    st.session_state.selected_cluster = None
# Add new session state variables for playback functionality
if 'playback_mode' not in st.session_state:
    st.session_state.playback_mode = False
if 'playback_dates' not in st.session_state:
    st.session_state.playback_dates = []
if 'playback_index' not in st.session_state:
    st.session_state.playback_index = 0

class OSMHandler:
    """Class for handling OpenStreetMap queries"""
    
    def __init__(self, verbose=False):
        self.overpass_url = "https://overpass-api.de/api/interpreter"
        self.timeout = 60  # seconds
        self.max_retries = 3
        self.verbose = verbose  # Add a flag to control logging
        
    def query_osm_features(self, bbox, tags, radius_km=10):
        """
        Query OSM features within a bounding box and with specified tags.
        
        Args:
            bbox (tuple): (min_lon, min_lat, max_lon, max_lat)
            tags (list): List of tag dictionaries to query for
            radius_km (float): Radius in kilometers for spatial join
            
        Returns:
            list: List of OSM features
        """
        if not HAVE_GEO_DEPS:
            # Silent handling instead of st.warning
            return []
            
        # Convert the radius to degrees (approximate, good enough for most uses)
        # 1 degree of latitude = ~111km, 1 degree of longitude varies with latitude
        radius_deg = radius_km / 111.0
        
        # Expand the bbox by the radius
        expanded_bbox = (
            bbox[0] - radius_deg,
            bbox[1] - radius_deg,
            bbox[2] + radius_deg,
            bbox[3] + radius_deg
        )
        
        # Build the Overpass query for all tag combinations
        tag_queries = []
        # Don't show this info
        for tag_dict in tags:
            tag_query = ""
            for k, v in tag_dict.items():
                tag_query += f'["{k}"="{v}"]'
            tag_queries.append(tag_query)
        
        # Combine all tag queries with OR operator
        if tag_queries:
            tag_query_combined = ' nwr ' + ' nwr '.join(tag_queries) + '; '
        else:
            # If no tags provided, match any node, way, or relation
            tag_query_combined = ' nwr; '
        
        # Build query with explicit bbox
        bbox_str = f"{expanded_bbox[0]},{expanded_bbox[1]},{expanded_bbox[2]},{expanded_bbox[3]}"
        overpass_query = f"""
        [out:json][timeout:{self.timeout}][bbox:{bbox_str}];
        (
          {tag_query_combined}
        );
        out center;
        """
        
        # Try to query with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    self.overpass_url,
                    params={"data": overpass_query},
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                # Don't show this success message
                return data['elements']
            except Exception as e:
                if attempt < self.max_retries - 1:
                    # Wait and retry
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    # Don't show this warning
                    return []
    
    def spatial_join(self, df, category, bbox):
        """
        Perform spatial join with OSM features based on category.
        
        Args:
            df (pandas.DataFrame): DataFrame with latitude and longitude columns
            category (str): Category to join ('flares', 'volcanoes', etc.)
            bbox (str): Bounding box string "min_lon,min_lat,max_lon,max_lat"
            
        Returns:
            pandas.DataFrame: DataFrame with additional OSM feature columns
        """
        # Skip if category doesn't need spatial join or we don't have GeoSpatial dependencies
        if not HAVE_GEO_DEPS or category not in ['flares', 'volcanoes']:
            return df
        
        # Parse the bbox string
        bbox_coords = [float(coord) for coord in bbox.split(',')]
        
        # Define tags for different categories - each category has its own exclusive tags
        tags = []
        if category == 'flares':
            # Only use flare/oil & gas related tags for 'flares' category
            tags = [
                {"man_made": "flare"},
                {"usage": "flare_header"},
                {"landmark": "flare_stack"},
                {"industrial": "oil"}
            ]
        elif category == 'volcanoes':
            # Only use volcano related tags for 'volcanoes' category
            tags = [
                {"natural": "volcano"},
                {"geological": "volcanic_vent"},
                {"volcano:type": "stratovolcano"},
                {"volcano:type": "scoria"},
                {"volcano:type": "shield"},
                {"volcano:type": "dirt"},
                {"volcano:type": "lava_dome"},
                {"volcano:type": "caldera"}
            ]
        
        # Query OSM features using the category-specific tags
        osm_features = self.query_osm_features(bbox_coords, tags)
        
        # Create GeoDataFrame from DataFrame
        gdf = gpd.GeoDataFrame(
            df,
            geometry=[Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])],
            crs="EPSG:4326"
        )
        
        # Create buffer of 10km (approximate, for quick calculation)
        # 0.1 degrees is approximately 11km at the equator
        buffer_degrees = 10 / 111  # Approximate conversion from km to degrees
        
        # Add columns for OSM matches
        gdf['osm_match'] = False
        gdf['osm_feature_id'] = None
        gdf['osm_feature_type'] = None
        gdf['osm_distance_km'] = None
        
        # Process OSM features
        if osm_features:
            # Create a list of Points for OSM features
            osm_points = []
            for feature in osm_features:
                # Get center coordinates
                if 'center' in feature:
                    lat, lon = feature['center']['lat'], feature['center']['lon']
                elif 'lat' in feature and 'lon' in feature:
                    lat, lon = feature['lat'], feature['lon']
                else:
                    continue
                
                osm_points.append({
                    'id': feature['id'],
                    'type': feature['type'],
                    'geometry': Point(lon, lat)
                })
            
            # Create GeoDataFrame for OSM features
            osm_gdf = gpd.GeoDataFrame(osm_points, crs="EPSG:4326")
            
            if not osm_gdf.empty:
                # Buffer the DataFrame points by 10km
                gdf_buffered = gdf.copy()
                gdf_buffered['geometry'] = gdf_buffered['geometry'].buffer(buffer_degrees)
                
                # Perform spatial join
                joined = gpd.sjoin(gdf_buffered, osm_gdf, how="left", predicate="intersects")
                
                # Update the original GDF with match information
                for idx, row in joined.iterrows():
                    if pd.notna(row['id']):
                        # Calculate distance (approximately)
                        orig_point = Point(df.loc[idx, 'longitude'], df.loc[idx, 'latitude'])
                        osm_point = row['geometry_right']
                        # Simple Euclidean distance (degrees) * 111 km/degree for approximate km
                        distance = orig_point.distance(osm_point) * 111
                        
                        # Update the dataframe
                        gdf.loc[idx, 'osm_match'] = True
                        gdf.loc[idx, 'osm_feature_id'] = row['id']
                        gdf.loc[idx, 'osm_feature_type'] = row['type']
                        gdf.loc[idx, 'osm_distance_km'] = distance
        
        # Return the updated DataFrame with OSM join information
        result_df = pd.DataFrame(gdf.drop(columns='geometry'))
        matched_count = result_df['osm_match'].sum()
        
        # This one is actually useful to know
        if self.verbose or matched_count > 0:
            st.success(f"Found {matched_count} points within 10km of relevant {category} features.")
        
        # For non-fire categories, only include points that match OSM features
        if category in ['flares', 'volcanoes'] and matched_count > 0:
            filtered_df = result_df[result_df['osm_match'] == True].copy()
            return filtered_df
        else:
            return result_df

class FIRMSHandler:
    def __init__(self, username, password, api_key):
        self.username = username
        self.password = password
        self.api_key = api_key
        self.base_url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        self.session = requests.Session()
        self.session.auth = (username, password)
        self.osm_handler = OSMHandler(verbose=False)

    def get_country_bbox(self, country):
        bboxes = {
            'Afghanistan': '60.52,29.31,75.15,38.48',
            'United States': '-125.0,24.0,-66.0,50.0',
            'Brazil': '-73.0,-33.0,-35.0,5.0',
            'Australia': '113.0,-44.0,154.0,-10.0',
            'India': '68.0,7.0,97.0,37.0',
            'China': '73.0,18.0,135.0,53.0',
            'Canada': '-141.0,41.7,-52.6,83.0',
            'Russia': '19.25,41.151,180.0,81.2',
            'Indonesia': '95.0,-11.0,141.0,6.0',
            'Mongolia': '87.76,41.59,119.93,52.15',
            'Kazakhstan': '46.46,40.57,87.36,55.45',
            'Mexico': '-118.4,14.5,-86.4,32.7',
            'Argentina': '-73.6,-55.1,-53.6,-21.8',
            'Chile': '-75.6,-55.9,-66.9,-17.5',
            'South Africa': '16.3,-34.8,32.9,-22.1',
            'New Zealand': '166.3,-47.3,178.6,-34.4',
            'Thailand': '97.3,5.6,105.6,20.5',
            'Vietnam': '102.1,8.4,109.5,23.4',
            'Malaysia': '99.6,0.8,119.3,7.4',
            'Myanmar': '92.2,9.8,101.2,28.5',
            'Philippines': '116.9,4.6,126.6,19.6',
            'Papua New Guinea': '140.8,-11.7,155.6,-1.3',
            'Greece': '19.4,34.8,28.3,41.8',
            'Turkey': '26.0,36.0,45.0,42.0',
            'Spain': '-9.3,36.0,4.3,43.8',
            'Portugal': '-9.5,37.0,-6.2,42.2',
            'Italy': '6.6,35.5,18.5,47.1',
            'France': '-5.1,41.3,9.6,51.1',
            'Germany': '5.9,47.3,15.0,55.1',
            'Ukraine': '22.1,44.4,40.2,52.4',
            'Sweden': '11.1,55.3,24.2,69.1',
            'Norway': '4.5,58.0,31.1,71.2',
            'Finland': '20.6,59.8,31.6,70.1',
            'Japan': '129.5,31.4,145.8,45.5',
            'South Korea': '126.1,33.1,129.6,38.6',
            'North Korea': '124.2,37.7,130.7,43.0',
            'Iran': '44.0,25.1,63.3,39.8',
            'Iraq': '38.8,29.1,48.8,37.4',
            'Saudi Arabia': '34.6,16.3,55.7,32.2',
            'Egypt': '24.7,22.0,36.9,31.7',
            'Libya': '9.3,19.5,25.2,33.2',
            'Algeria': '-8.7,19.1,12.0,37.1',
            'Morocco': '-13.2,27.7,-1.0,35.9',
            'Sudan': '21.8,8.7,38.6,22.2',
            'South Sudan': '23.4,3.5,35.9,12.2',
            'Ethiopia': '33.0,3.4,47.9,14.8',
            'Kenya': '33.9,-4.7,41.9,5.0',
            'Tanzania': '29.3,-11.7,40.4,-1.0',
            'Uganda': '29.5,-1.4,35.0,4.2',
            'Nigeria': '2.7,4.3,14.7,13.9',
            'Ghana': '-3.3,4.7,1.2,11.2',
            'Ivory Coast': '-8.6,4.4,-2.5,10.7',
            'Guinea': '-15.1,7.2,-7.6,12.7',
            'Somalia': '40.9,-1.7,51.4,11.9',
            'Democratic Republic of the Congo': '12.2,-13.5,31.3,5.3',
            'Angola': '11.7,-18.0,24.1,-4.4',
            'Namibia': '11.7,-28.9,25.3,-16.9',
            'Zambia': '22.0,-18.0,33.7,-8.2',
            'Zimbabwe': '25.2,-22.4,33.1,-15.6',
            'Mozambique': '30.2,-26.9,40.9,-10.5',
            'Madagascar': '43.2,-25.6,50.5,-11.9',
            'Colombia': '-79.0,-4.2,-66.9,12.5',
            'Venezuela': '-73.4,0.6,-59.8,12.2',
            'Peru': '-81.3,-18.4,-68.7,-0.0',
            'Bolivia': '-69.6,-22.9,-57.5,-9.7',
            'Paraguay': '-62.6,-27.6,-54.3,-19.3',
            'Uruguay': '-58.4,-34.9,-53.1,-30.1',
            'Ecuador': '-81.0,-5.0,-75.2,1.4',
            'French Guiana': '-54.6,2.1,-51.6,5.8',
            'Suriname': '-58.1,1.8,-54.0,6.0',
            'Guyana': '-61.4,1.2,-56.5,8.6',
            'Panama': '-83.0,7.2,-77.1,9.7',
            'Costa Rica': '-85.9,8.0,-82.5,11.2',
            'Nicaragua': '-87.7,10.7,-83.1,15.0',
            'Honduras': '-89.4,12.9,-83.1,16.5',
            'El Salvador': '-90.1,13.1,-87.7,14.5',
            'Guatemala': '-92.2,13.7,-88.2,17.8',
            'Belize': '-89.2,15.9,-87.8,18.5',
            'Cuba': '-85.0,19.8,-74.1,23.2',
            'Haiti': '-74.5,18.0,-71.6,20.1',
            'Dominican Republic': '-72.0,17.5,-68.3,20.0',
            'Jamaica': '-78.4,17.7,-76.2,18.5',
            'Puerto Rico': '-67.3,17.9,-65.6,18.5',
            'Bahamas': '-79.0,20.9,-72.7,27.3',
            'Trinidad and Tobago': '-61.9,10.0,-60.5,11.3',
            'Bangladesh': '88.0,20.6,92.7,26.6',
            'Nepal': '80.0,26.3,88.2,30.4',
            'Bhutan': '88.7,26.7,92.1,28.3',
            'Sri Lanka': '79.6,5.9,81.9,9.8',
            'Maldives': '72.7,-0.7,73.8,7.1',
            'Pakistan': '61.0,23.5,77.8,37.1',
            'Afghanistan': '60.5,29.4,74.9,38.5',
            'Uzbekistan': '56.0,37.2,73.1,45.6',
            'Turkmenistan': '52.5,35.1,66.7,42.8',
            'Tajikistan': '67.3,36.7,75.2,41.0',
            'Kyrgyzstan': '69.3,39.2,80.3,43.3',
            'Cambodia': '102.3,10.4,107.6,14.7',
            'Laos': '100.1,13.9,107.7,22.5',
            'Taiwan': '120.0,21.9,122.0,25.3',
            'United Arab Emirates': '51.5,22.6,56.4,26.1',
            'Oman': '52.0,16.6,59.8,26.4',
            'Yemen': '42.5,12.5,54.0,19.0',
            'Kuwait': '46.5,28.5,48.4,30.1',
            'Qatar': '50.7,24.5,51.6,26.2',
            'Bahrain': '50.4,25.8,50.8,26.3',
            'Jordan': '34.9,29.2,39.3,33.4',
            'Lebanon': '35.1,33.0,36.6,34.7',
            'Syria': '35.7,32.3,42.4,37.3',
            'Israel': '34.2,29.5,35.9,33.3',
            'Palestine': '34.9,31.2,35.6,32.6',
            'Cyprus': '32.0,34.6,34.6,35.7',
            'Iceland': '-24.5,63.3,-13.5,66.6',
            'Ireland': '-10.5,51.4,-6.0,55.4',
            'United Kingdom': '-8.2,49.9,1.8,58.7',
            'Belgium': '2.5,49.5,6.4,51.5',
            'Netherlands': '3.3,50.8,7.2,53.5',
            'Luxembourg': '5.7,49.4,6.5,50.2',
            'Switzerland': '5.9,45.8,10.5,47.8',
            'Austria': '9.5,46.4,17.2,49.0',
            'Hungary': '16.1,45.7,22.9,48.6',
            'Slovakia': '16.8,47.7,22.6,49.6',
            'Czech Republic': '12.1,48.5,18.9,51.1',
            'Poland': '14.1,49.0,24.2,54.8',
            'Denmark': '8.0,54.5,15.2,57.8',
            'Estonia': '23.3,57.5,28.2,59.7',
            'Latvia': '20.8,55.7,28.2,58.1',
            'Lithuania': '20.9,53.9,26.8,56.5',
            'Belarus': '23.2,51.3,32.8,56.2',
            'Moldova': '26.6,45.5,30.2,48.5',
            'Romania': '20.3,43.6,29.7,48.3',
            'Bulgaria': '22.4,41.2,28.6,44.2',
            'Serbia': '18.8,42.2,23.0,46.2',
            'Croatia': '13.5,42.4,19.4,46.6',
            'Bosnia and Herzegovina': '15.7,42.6,19.6,45.3',
            'Slovenia': '13.4,45.4,16.6,46.9',
            'Albania': '19.3,39.6,21.1,42.7',
            'North Macedonia': '20.4,40.8,23.0,42.4',
            'Montenegro': '18.4,41.9,20.4,43.6',
            'New Caledonia': '164.0,-22.7,167.0,-20.0',
            'Fiji': '177.0,-19.2,180.0,-16.0',
            'Vanuatu': '166.0,-20.3,170.0,-13.0',
            'Solomon Islands': '155.0,-11.0,170.0,-5.0',
            'Timor-Leste': '124.0,-9.5,127.3,-8.1',
            'Palau': '131.1,2.8,134.7,8.1',
            'Micronesia': '138.0,1.0,163.0,10.0',
            'Marshall Islands': '160.0,4.0,172.0,15.0',
            'Kiribati': '-175.0,-5.0,177.0,5.0',
            'Tuvalu': '176.0,-10.0,180.0,-5.0',
            'Samoa': '-172.8,-14.1,-171.4,-13.4',
            'Tonga': '-175.4,-22.4,-173.7,-15.5',
            'Cook Islands': '-166.0,-22.0,-157.0,-8.0',
        }
        return bboxes.get(country, None)

    def _apply_dbscan(self, df, eps=0.01, min_samples=5, bbox=None, max_time_diff_days=5):
        
        def _apply_dbscan(self, df, eps=0.01, min_samples=5, bbox=None, max_time_diff_days=5):
            # Debug column names
            st.write(f"Input columns: {df.columns.tolist()}")
            
            # Case-insensitive column name mapping
            column_map = {col.lower(): col for col in df.columns}
            
            # Check for date column with various spellings/cases
            date_col = None
            for possible_name in ['acq_date', 'date', 'Date', 'ACQ_DATE']:
                if possible_name in df.columns:
                    date_col = possible_name
                    break
                elif possible_name.lower() in column_map:
                    date_col = column_map[possible_name.lower()]
                    break
            
            if date_col:
                st.write(f"Found date column: '{date_col}'")
                # Standardize the date column name
                df['acq_date'] = df[date_col]
            else:
                st.warning("No date column found! Available columns: " + ", ".join(df.columns.tolist()))
                
        
        """Apply DBSCAN clustering with bbox filtering and time-based constraints
        
        Args:
            df (pandas.DataFrame): DataFrame with latitude and longitude columns
            eps (float): DBSCAN eps parameter - spatial proximity threshold
            min_samples (int): Minimum points to form a cluster
            bbox (str): Bounding box string "min_lon,min_lat,max_lon,max_lat"
            max_time_diff_days (int): Maximum days between events to consider as same cluster
                                     Higher values will group events over longer time periods
        """
        
        if 'Date' in df.columns and 'acq_date' not in df.columns:
                df['acq_date'] = df['Date']
        elif 'date' in df.columns and 'acq_date' not in df.columns:
                df['acq_date'] = df['date']
        
        if len(df) < min_samples:
            st.warning(f"Too few points ({len(df)}) for clustering. Minimum required: {min_samples}")
            return df
        
        # First filter the data by bounding box if provided
        if bbox:
            # Parse the bbox string to get coordinates
            bbox_coords = [float(coord) for coord in bbox.split(',')]
            if len(bbox_coords) == 4:  # min_lon, min_lat, max_lon, max_lat
                min_lon, min_lat, max_lon, max_lat = bbox_coords
                
                # Filter dataframe to only include points within the bounding box
                bbox_mask = (
                    (df['longitude'] >= min_lon) & 
                    (df['longitude'] <= max_lon) & 
                    (df['latitude'] >= min_lat) & 
                    (df['latitude'] <= max_lat)
                )
                
                filtered_df = df[bbox_mask].copy()
                st.info(f"Filtered data to {len(filtered_df)} points within the selected country boundaries.")
                
                # If filtering resulted in too few points, return the filtered df without clustering
                if len(filtered_df) < min_samples:
                    st.warning(f"Too few points within country boundaries ({len(filtered_df)}) for clustering. Minimum required: {min_samples}")
                    # Mark all as noise
                    filtered_df['cluster'] = -1
                    return filtered_df
                
                df = filtered_df
        
        # Time-based clustering - only if acq_date column exists
        if 'acq_date' in df.columns:
            try:
                # Convert acquisition date to datetime 
                df['acq_date_dt'] = pd.to_datetime(df['acq_date'])
                
                cluster_days = df[df['cluster'] >= 0].groupby('cluster')['acq_date'].nunique()
                multi_day_clusters = cluster_days[cluster_days > 1].index.tolist()
                
                if multi_day_clusters:
                    st.success(f"Found {len(multi_day_clusters)} multi-day clusters during clustering: {multi_day_clusters}")
                
                # Create feature matrix with spatial and temporal components
                coords = df[['latitude', 'longitude']].values
                
                # Calculate days from earliest date for temporal component
                earliest_date = df['acq_date_dt'].min()
                df['days_from_earliest'] = (df['acq_date_dt'] - earliest_date).dt.total_seconds() / (24 * 3600)
                
                # Scale the time component - higher weight = stricter time constraint
                time_scaling = 0.1 / max_time_diff_days # Inverse of max days difference
                
                # Create feature matrix with scaled time component
                feature_matrix = np.column_stack([
                    coords,  # Latitude and longitude
                    df['days_from_earliest'].values * time_scaling  # Scaled time component
                ])
                
                # Apply DBSCAN to the combined spatial-temporal features
                clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(feature_matrix)
                df['cluster'] = clustering.labels_
                
                feature_ranges = np.ptp(feature_matrix, axis=0)  # Peak-to-peak range of each dimension
                st.write(f"Feature matrix dimension ranges: Lat/Lon: {feature_ranges[0]:.4f}, {feature_ranges[1]:.4f}, Time: {feature_ranges[2]:.4f}")
                st.write(f"Suggested eps (for reference): {np.mean(feature_ranges):.4f}")
                
                if len(df) > 10:  # Check if we have enough data points
                    sample_points = df.sample(min(10, len(df))).copy()
                    # Calculate time differences between sample points (in days)
                    sample_points['time_diff'] = (sample_points['acq_date_dt'] - sample_points['acq_date_dt'].min()).dt.total_seconds() / (24 * 3600)
                    st.write("Sample time differences (days):", sample_points['time_diff'].tolist())
                    st.write("Time scaling factor:", time_scaling)
                    # Show example feature matrix values
                    st.write("Example feature matrix values (lat, lon, scaled_time):", feature_matrix[:3].tolist())
                
                # Clean up temporary columns
                df = df.drop(columns=['days_from_earliest'])
                
                # Regular clustering stats
                n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
                n_noise = list(clustering.labels_).count(-1)
                
                st.write(f"Number of clusters found: {n_clusters}")
                st.write(f"Number of noise points: {n_noise}")
                st.write(f"Points in clusters: {len(df) - n_noise}")
                
                return df
                
            except Exception as e:
                st.warning(f"Error in time-based clustering: {str(e)}. Falling back to spatial-only clustering.")
                # Fall through to standard clustering
        
        # Standard spatial-only clustering as fallback
        coords = df[['latitude', 'longitude']].values
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        df['cluster'] = clustering.labels_
        
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        n_noise = list(clustering.labels_).count(-1)
        
        st.write(f"Number of clusters found: {n_clusters}")
        st.write(f"Number of noise points: {n_noise}")
        st.write(f"Points in clusters: {len(df) - n_noise}")
        
        return df

    def fetch_fire_data(
        self, 
        country=None, 
        bbox=None, 
        dataset='VIIRS_NOAA20_NRT', 
        start_date=None, 
        end_date=None,
        category='fires',
        use_clustering=True,
        eps=0.01,
        min_samples=5,
        chunk_days=7,  # Default chunk size
        max_time_diff_days=5  # Maximum days gap to consider as same fire (default: 5 days)
    ):
        """Fetch and process fire data from FIRMS API with support for historical data"""
        
        # Determine if we need historical data
        today = datetime.now().date()
        
        # Convert dates to proper format for comparison
        if isinstance(start_date, date):
            start_date_date = start_date
        elif isinstance(start_date, datetime):
            start_date_date = start_date.date()
        else:
            # If it's a string, parse it
            try:
                start_date_date = datetime.strptime(str(start_date), "%Y-%m-%d").date()
            except:
                # Default to 7 days ago if parsing fails
                start_date_date = today - timedelta(days=7)
        
        # Check if we need historical data (more than 10 days ago)
        need_historical = (today - start_date_date).days > 10
        
        # If we need historical data, switch to Standard Processing dataset
        original_dataset = dataset
        if need_historical and "_NRT" in dataset:
            # Switch to Standard Processing version
            dataset = dataset.replace("_NRT", "_SP")
            st.info(f"Fetching historical data using {dataset} dataset")
        
        # Dataset availability dates
        dataset_availability = {
            'MODIS_NRT': {'min_date': '2024-12-01', 'max_date': '2025-03-17'},
            'MODIS_SP': {'min_date': '2000-11-01', 'max_date': '2024-11-30'},
            'VIIRS_NOAA20_NRT': {'min_date': '2024-12-01', 'max_date': '2025-03-17'},
            'VIIRS_NOAA20_SP': {'min_date': '2018-04-01', 'max_date': '2024-11-30'},
            'VIIRS_NOAA21_NRT': {'min_date': '2024-01-17', 'max_date': '2025-03-17'},
            'VIIRS_SNPP_NRT': {'min_date': '2025-01-01', 'max_date': '2025-03-17'},
            'VIIRS_SNPP_SP': {'min_date': '2012-01-20', 'max_date': '2024-12-31'},
            'LANDSAT_NRT': {'min_date': '2022-06-20', 'max_date': '2025-03-17'}
        }
        
        if dataset not in dataset_availability:
            st.error(f"Invalid dataset: {dataset}. Please select a valid dataset.")
            return None
        
        # Check if the requested date range is available for this dataset
        if dataset in dataset_availability:
            min_date = datetime.strptime(dataset_availability[dataset]['min_date'], '%Y-%m-%d').date()
            max_date = datetime.strptime(dataset_availability[dataset]['max_date'], '%Y-%m-%d').date()
            
            if start_date_date < min_date:
                st.warning(f"Start date {start_date_date} is before the earliest available date ({min_date}) for {dataset}. Using earliest available date.")
                start_date_date = min_date
        
        if not bbox and country:
            bbox = self.get_country_bbox(country)
        
        if not bbox:
            st.error("Provide a country or bounding box")
            return None
        
        # Convert dates to strings
        start_date_str = start_date_date.strftime('%Y-%m-%d')
        
        if isinstance(end_date, date):
            end_date_date = end_date
        elif isinstance(end_date, datetime):
            end_date_date = end_date.date()
        else:
            # If it's a string, parse it
            try:
                end_date_date = datetime.strptime(str(end_date), "%Y-%m-%d").date()
            except:
                # Default to today if parsing fails
                end_date_date = today
        
        # Ensure end date doesn't exceed dataset's max date
        if dataset in dataset_availability:
            max_date = datetime.strptime(dataset_availability[dataset]['max_date'], '%Y-%m-%d').date()
            if end_date_date > max_date:
                st.warning(f"End date {end_date_date} is after the latest available date ({max_date}) for {dataset}. Using latest available date.")
                end_date_date = max_date
        
        end_date_str = end_date_date.strftime('%Y-%m-%d')
        
        # Now we need to fetch data in chunks, respecting the 10-day limit
        st.write(f"Fetching fire data from {start_date_str} to {end_date_str} for {country}...")
        
        # Create date chunks of 10 days or less
        date_chunks = []
        current_date = start_date_date
        while current_date <= end_date_date:
            chunk_end = min(current_date + timedelta(days=min(10, chunk_days)-1), end_date_date)
            date_chunks.append((current_date, chunk_end))
            current_date = chunk_end + timedelta(days=1)
        
        # Set up progress tracking
        st.write(f"Processing data in {len(date_chunks)} chunks...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize combined results
        all_results = pd.DataFrame()
        
        # Special handling for large countries
        large_countries = ['United States', 'China', 'Russia', 'Canada', 'Brazil', 'Australia', 'India']
        if country in large_countries:
            st.warning(f"⚠️ You selected a {(end_date_date - start_date_date).days} day period for {country}, which is a large country. This may take a long time to process.")
            
            # Special handling for Russia which is particularly large
            if country == 'Russia':
                st.info("Russia is very large. Dividing into smaller regions for better performance...")
                
                # Process western Russia
                west_bbox = '19.25,41.151,60.0,81.2'
                st.write("Processing Western Russia...")
                for i, (chunk_start, chunk_end) in enumerate(date_chunks):
                    chunk_start_str = chunk_start.strftime('%Y-%m-%d')
                    chunk_end_str = chunk_end.strftime('%Y-%m-%d')
                    status_text.write(f"Western Region - Chunk {i+1}/{len(date_chunks)}: {chunk_start_str} to {chunk_end_str}")
                    progress_bar.progress((i) / (len(date_chunks) * 3))  # 3 regions
                    
                    days_in_chunk = (chunk_end - chunk_start).days + 1
                    if need_historical:
                        url = f"{self.base_url}{self.api_key}/{dataset}/{west_bbox}/{days_in_chunk}/{chunk_start_str}"
                    else:
                        url = f"{self.base_url}{self.api_key}/{dataset}/{west_bbox}/{days_in_chunk}/{chunk_start_str}"
                    
                    try:
                        response = self.session.get(url, timeout=45)  # Shorter timeout
                        response.raise_for_status()
                        if response.text.strip() and "Invalid" not in response.text:
                            chunk_df = pd.read_csv(StringIO(response.text))
                            if not chunk_df.empty:
                                date_mask = (chunk_df['acq_date'] >= chunk_start_str) & (chunk_df['acq_date'] <= chunk_end_str)
                                filtered_chunk = chunk_df[date_mask].copy()
                                if not filtered_chunk.empty:
                                    all_results = pd.concat([all_results, filtered_chunk], ignore_index=True)
                    except Exception as e:
                        st.warning(f"Error processing Western Russia chunk {i+1}: {str(e)}")
                
                # Process central Russia
                central_bbox = '60.0,41.151,120.0,81.2'
                st.write("Processing Central Russia...")
                for i, (chunk_start, chunk_end) in enumerate(date_chunks):
                    chunk_start_str = chunk_start.strftime('%Y-%m-%d')
                    chunk_end_str = chunk_end.strftime('%Y-%m-%d')
                    status_text.write(f"Central Region - Chunk {i+1}/{len(date_chunks)}: {chunk_start_str} to {chunk_end_str}")
                    progress_bar.progress((len(date_chunks) + i) / (len(date_chunks) * 3))
                    
                    days_in_chunk = (chunk_end - chunk_start).days + 1
                    if need_historical:
                        url = f"{self.base_url}{self.api_key}/{dataset}/{central_bbox}/{days_in_chunk}/{chunk_start_str}"
                    else:
                        url = f"{self.base_url}{self.api_key}/{dataset}/{central_bbox}/{days_in_chunk}/{chunk_start_str}"
                    
                    try:
                        response = self.session.get(url, timeout=45)  # Shorter timeout
                        response.raise_for_status()
                        if response.text.strip() and "Invalid" not in response.text:
                            chunk_df = pd.read_csv(StringIO(response.text))
                            if not chunk_df.empty:
                                date_mask = (chunk_df['acq_date'] >= chunk_start_str) & (chunk_df['acq_date'] <= chunk_end_str)
                                filtered_chunk = chunk_df[date_mask].copy()
                                if not filtered_chunk.empty:
                                    all_results = pd.concat([all_results, filtered_chunk], ignore_index=True)
                    except Exception as e:
                        st.warning(f"Error processing Central Russia chunk {i+1}: {str(e)}")
                
                # Process eastern Russia
                east_bbox = '120.0,41.151,180.0,81.2'
                st.write("Processing Eastern Russia...")
                for i, (chunk_start, chunk_end) in enumerate(date_chunks):
                    chunk_start_str = chunk_start.strftime('%Y-%m-%d')
                    chunk_end_str = chunk_end.strftime('%Y-%m-%d')
                    status_text.write(f"Eastern Region - Chunk {i+1}/{len(date_chunks)}: {chunk_start_str} to {chunk_end_str}")
                    progress_bar.progress((2 * len(date_chunks) + i) / (len(date_chunks) * 3))
                    
                    days_in_chunk = (chunk_end - chunk_start).days + 1
                    if need_historical:
                        url = f"{self.base_url}{self.api_key}/{dataset}/{east_bbox}/{days_in_chunk}/{chunk_start_str}"
                    else:
                        url = f"{self.base_url}{self.api_key}/{dataset}/{east_bbox}/{days_in_chunk}/{chunk_start_str}"
                    
                    try:
                        response = self.session.get(url, timeout=45)  # Shorter timeout
                        response.raise_for_status()
                        if response.text.strip() and "Invalid" not in response.text:
                            chunk_df = pd.read_csv(StringIO(response.text))
                            if not chunk_df.empty:
                                date_mask = (chunk_df['acq_date'] >= chunk_start_str) & (chunk_df['acq_date'] <= chunk_end_str)
                                filtered_chunk = chunk_df[date_mask].copy()
                                if not filtered_chunk.empty:
                                    all_results = pd.concat([all_results, filtered_chunk], ignore_index=True)
                    except Exception as e:
                        st.warning(f"Error processing Eastern Russia chunk {i+1}: {str(e)}")
            else:
                # For other large countries, use standard approach with longer timeout
                self.session.timeout = 120  # Increase timeout to 2 minutes
                
                # Standard chunked processing for other countries
                for i, (chunk_start, chunk_end) in enumerate(date_chunks):
                    chunk_start_str = chunk_start.strftime('%Y-%m-%d')
                    chunk_end_str = chunk_end.strftime('%Y-%m-%d')
                    
                    # Update progress
                    status_text.write(f"Fetching chunk {i+1}/{len(date_chunks)}: {chunk_start_str} to {chunk_end_str}")
                    progress_bar.progress((i) / len(date_chunks))
                    
                    # Get the number of days in this chunk
                    days_in_chunk = (chunk_end - chunk_start).days + 1
                    
                    # Format API URL based on historical data approach
                    if need_historical:
                        url = f"{self.base_url}{self.api_key}/{dataset}/{bbox}/{days_in_chunk}/{chunk_start_str}"
                    else:
                        if i == 0 and len(date_chunks) == 1 and days_in_chunk <= 7:
                            url = f"{self.base_url}{self.api_key}/{original_dataset}/{bbox}/7"
                        else:
                            url = f"{self.base_url}{self.api_key}/{dataset}/{bbox}/{days_in_chunk}/{chunk_start_str}"
                    
                    try:
                        # Fetch data for this chunk
                        response = self.session.get(url, timeout=120)  # Longer timeout for large countries
                        response.raise_for_status()
                        
                        # Parse CSV data if valid
                        if response.text.strip() and "Invalid" not in response.text and "Error" not in response.text:
                            chunk_df = pd.read_csv(StringIO(response.text))
                            
                            # Only process non-empty results
                            if not chunk_df.empty:
                                # Filter to ensure records are within the requested date range
                                if 'acq_date' in chunk_df.columns:
                                    date_mask = (chunk_df['acq_date'] >= chunk_start_str) & (chunk_df['acq_date'] <= chunk_end_str)
                                    filtered_chunk = chunk_df[date_mask].copy()
                                    if not filtered_chunk.empty:
                                        all_results = pd.concat([all_results, filtered_chunk], ignore_index=True)
                                else:
                                    all_results = pd.concat([all_results, chunk_df], ignore_index=True)
                    except Exception as e:
                        st.warning(f"Error processing chunk {i+1}: {str(e)}")
        else:
            # Standard chunked processing for normal countries
            for i, (chunk_start, chunk_end) in enumerate(date_chunks):
                chunk_start_str = chunk_start.strftime('%Y-%m-%d')
                chunk_end_str = chunk_end.strftime('%Y-%m-%d')
                
                # Update progress
                status_text.write(f"Fetching chunk {i+1}/{len(date_chunks)}: {chunk_start_str} to {chunk_end_str}")
                progress_bar.progress((i) / len(date_chunks))
                
                # Get the number of days in this chunk
                days_in_chunk = (chunk_end - chunk_start).days + 1
                
                # Format API URL based on historical data approach
                if need_historical:
                    url = f"{self.base_url}{self.api_key}/{dataset}/{bbox}/{days_in_chunk}/{chunk_start_str}"
                else:
                    if i == 0 and len(date_chunks) == 1 and days_in_chunk <= 7:
                        url = f"{self.base_url}{self.api_key}/{original_dataset}/{bbox}/7"
                    else:
                        url = f"{self.base_url}{self.api_key}/{dataset}/{bbox}/{days_in_chunk}/{chunk_start_str}"
                
                try:
                    # Fetch data for this chunk
                    response = self.session.get(url, timeout=60)
                    response.raise_for_status()
                    
                    # Parse CSV data if valid
                    if response.text.strip() and "Invalid" not in response.text and "Error" not in response.text:
                        chunk_df = pd.read_csv(StringIO(response.text))
                        
                        # Only process non-empty results
                        if not chunk_df.empty:
                            # Filter to ensure records are within the requested date range
                            if 'acq_date' in chunk_df.columns:
                                date_mask = (chunk_df['acq_date'] >= chunk_start_str) & (chunk_df['acq_date'] <= chunk_end_str)
                                filtered_chunk = chunk_df[date_mask].copy()
                                if not filtered_chunk.empty:
                                    all_results = pd.concat([all_results, filtered_chunk], ignore_index=True)
                            else:
                                all_results = pd.concat([all_results, chunk_df], ignore_index=True)
                except Exception as e:
                    st.warning(f"Error processing chunk {i+1}: {str(e)}")
        
        # Clean up progress indicators
        progress_bar.progress(1.0)
        status_text.empty()
        
        # Check if we got any data
        if all_results.empty:
            st.warning(f"No records found for {category} in {country} for the selected date range")
            return None
        
        st.success(f"Successfully fetched {len(all_results)} records from FIRMS API")
        
        # Apply bbox filtering to make sure points are within country boundaries
        if bbox and not all_results.empty:
            # Parse the bbox string to get coordinates
            bbox_coords = [float(coord) for coord in bbox.split(',')]
            if len(bbox_coords) == 4:  # min_lon, min_lat, max_lon, max_lat
                min_lon, min_lat, max_lon, max_lat = bbox_coords
                
                # Filter dataframe to only include points within the bounding box
                bbox_mask = (
                    (all_results['longitude'] >= min_lon) & 
                    (all_results['longitude'] <= max_lon) & 
                    (all_results['latitude'] >= min_lat) & 
                    (all_results['latitude'] <= max_lat)
                )
                
                filtered_df = all_results[bbox_mask].copy()
                st.info(f"Filtered data to {len(filtered_df)} points within the selected country boundaries.")
                
                if len(filtered_df) == 0:
                    st.warning(f"No points found within the specified bounding box for {country}.")
                    return None
                
                all_results = filtered_df
        
        # Apply clustering to the results if needed
        if use_clustering and not all_results.empty:
            all_results = self._apply_dbscan(all_results, eps=eps, min_samples=min_samples, bbox=bbox, max_time_diff_days=max_time_diff_days)
        
        # Apply spatial joins for specific categories
        if category in ['flares', 'volcanoes'] and HAVE_GEO_DEPS and not all_results.empty:
            with st.spinner(f'Performing spatial join with OSM {category} data...'):
                original_count = len(all_results)
                all_results = self.osm_handler.spatial_join(all_results, category, bbox)
                
                # If spatial join found no matches
                if all_results.empty:
                    # Create a container for the message
                    message_container = st.empty()
                    message_container.warning(f"No {category} found within the selected area and date range. Try a different location or category.")
                    # Return None to prevent map creation
                    return None
                    
        st.write("Raw Data Information:")
        st.write(f"Total records: {len(all_results)}")
        
        return all_results

def get_temp_column(df):
    """Determine which temperature column to use based on available data"""
    if 'bright_ti4' in df.columns:
        return 'bright_ti4'
    elif 'brightness' in df.columns:
        return 'brightness'
    else:
        return None

def get_category_display_name(category):
    """Return the display name for a category"""
    if category == "fires":
        return "Fire"
    elif category == "flares":
        return "Flare"
    elif category == "volcanoes":
        return "Volcano"
    else:
        return "Cluster"  # Default for raw data

def get_category_singular(category):
    """Return singular form of category name for UI purposes"""
    if category == "fires":
        return "fire"
    elif category == "flares":
        return "flare"
    elif category == "volcanoes":
        return "volcano"
    else:
        return "cluster"  # Default for raw data

def create_export_map(data, title, basemap_tiles, basemap, dot_color='#ff3300', border_color='white', border_width=1.5, fixed_zoom=7):
    """Create a simplified map for export with static zoom and custom colors"""
    if data.empty:
        return None
    
    # Find coordinate columns
    lat_col = next((col for col in ['latitude', 'Latitude', 'lat', 'Lat'] if col in data.columns), None)
    lon_col = next((col for col in ['longitude', 'Longitude', 'lon', 'Lon'] if col in data.columns), None)
    
    if not lat_col or not lon_col:
        st.error(f"Cannot find coordinate columns in {data.columns.tolist()}")
        return None
    
    # Calculate center point of data
    center_lat = (data[lat_col].min() + data[lat_col].max()) / 2
    center_lon = (data[lon_col].min() + data[lon_col].max()) / 2
    
    # Get the appropriate tile URL based on basemap selection
    tile_url = basemap_tiles.get(basemap, basemap_tiles['Dark'])
    
    # Create a map with fixed zoom and the selected basemap
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=fixed_zoom,  # Fixed zoom level
        tiles=tile_url
    )
    
    # Add title
    title_html = f'<h3 align="center" style="font-size:16px; color: white;"><b>{title}</b></h3>'
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add each point with custom colors
    for idx, row in data.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=6,
            color=border_color,         # Border color
            weight=border_width,        # Border width
            fill=True,
            fill_color=dot_color,       # Fill color
            fill_opacity=0.9
        ).add_to(m)
    
    # Return the HTML
    return m._repr_html_()

def export_timeline(df, cluster_id=None, category="fires", playback_dates=None, basemap_tiles=None, basemap="Dark"):
    """Create a timeline export as GIF or MP4
    
    Args:
        df (pandas.DataFrame): DataFrame with fire data
        cluster_id (int, optional): Specific cluster ID to export. If None, exports all clusters.
        category (str): Category name (fires, flares, etc.)
        playback_dates (list): List of dates to include in playback
        basemap_tiles (dict): Dictionary mapping of basemap names to tile URLs
        basemap (str): Selected basemap name
    """
    # Initialize basemap_tiles if not provided
    if basemap_tiles is None:
        basemap_tiles = {
            'Dark': 'cartodbdark_matter',
            'Light': 'cartodbpositron',
            'Satellite': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            'Terrain': 'stamenterrain'
        }
    
    # Filter for the selected cluster(s)
    if cluster_id is not None:
        # Export a single cluster
        export_single_cluster_timeline(df, cluster_id, category, playback_dates, basemap_tiles, basemap)
    else:
        # Export all clusters (multi-cluster visualization)
        export_all_clusters_timeline(df, category, playback_dates, basemap_tiles, basemap)

def create_gif_from_frames(frames, fps=2):
    """Create a GIF from HTML frames - without individual frame downloads"""
    try:
        # Import required libraries
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        from PIL import Image
        import tempfile
        import os
        
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        screenshot_paths = []
        
        # Set up Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1200,800")
        
        st.info("Setting up browser for image capture...")
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        
        # Process each frame
        progress_bar = st.progress(0)
        for i, html_content in enumerate(frames):
            st.write(f"Processing frame {i+1}/{len(frames)}")
            progress_bar.progress((i+1)/len(frames))
            
            # Create simplified HTML
            simple_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ margin: 0; padding: 0; background-color: #000; }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """
            
            # Save to file
            html_path = os.path.join(temp_dir, f"frame_{i}.html")
            with open(html_path, "w") as f:
                f.write(simple_html)
            
            # Load in browser
            driver.get(f"file://{html_path}")
            time.sleep(2)  # Wait for map to render
            
            # Take screenshot
            screenshot_path = os.path.join(temp_dir, f"frame_{i}.png")
            driver.save_screenshot(screenshot_path)
            screenshot_paths.append(screenshot_path)
        
        # Close driver
        driver.quit()
        
        # Create GIF
        images = [Image.open(path) for path in screenshot_paths if os.path.exists(path)]
        if not images:
            raise Exception("No valid screenshots captured")
            
        # Output GIF
        gif_buffer = BytesIO()
        images[0].save(
            gif_buffer,
            format='GIF',
            append_images=images[1:],
            save_all=True,
            duration=1000//fps,
            loop=0
        )
        gif_buffer.seek(0)
        
        # Clean up
        for path in screenshot_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass
        
        st.success("GIF created successfully!")
        
        # Removed the individual frame download buttons
        
        return gif_buffer.getvalue()
        
    except Exception as e:
        st.error(f"Error creating GIF: {str(e)}")
        return None

def export_single_cluster_timeline(df, cluster_id, category, playback_dates, basemap_tiles, basemap):
    """Export timeline for a single cluster"""
    # Get data for the selected cluster
    cluster_data = df[df['cluster'] == cluster_id]
    
    # Group by date and count points
    date_counts = cluster_data.groupby('acq_date').size()
    dates_with_data = list(date_counts.index)
    
    # Check if we have at least 2 dates with data
    if len(dates_with_data) <= 1:
        st.warning(f"This {get_category_singular(category)} only has data for one date. Timeline export requires data on multiple dates.")
        return
    
    # Set up progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Capture frames for each date
    frames = []
    total_dates = len(dates_with_data)
    
    # Custom colors for fire visualization
    dot_color = '#ff3300'  # Red-orange fill
    border_color = 'white'
    
    # Find the bounds of all data to determine a fixed zoom level
    all_lat = []
    all_lon = []
    
    # Find coordinate columns
    lat_col = next((col for col in ['latitude', 'Latitude', 'lat', 'Lat'] if col in cluster_data.columns), None)
    lon_col = next((col for col in ['longitude', 'Longitude', 'lon', 'Lon'] if col in cluster_data.columns), None)
    
    if lat_col and lon_col:
        all_lat = cluster_data[lat_col].tolist()
        all_lon = cluster_data[lon_col].tolist()
    
    for i, date in enumerate(sorted(dates_with_data)):
        status_text.write(f"Processing frame {i+1}/{total_dates}: {date}")
        progress_bar.progress((i+1)/total_dates)
        
        # Create map for this date
        playback_title = f"{get_category_display_name(category)} {cluster_id} - {date}"
        
        # Filter data for this date and cluster
        date_data = df[(df['cluster'] == cluster_id) & (df['acq_date'] == date)].copy()
        
        if not date_data.empty:
            # Create a simplified map for export
            folium_map = create_export_map(
                date_data, 
                playback_title, 
                basemap_tiles, 
                basemap,
                dot_color=dot_color,
                border_color=border_color
            )
            frames.append(folium_map)
    
    status_text.write("Processing complete. Preparing download...")
    
    # Store frames in session state
    st.session_state.frames = frames
    
    # Provide download option
    if frames:
        # Create download buffer
        st.info(f"Timeline export ready for cluster {cluster_id}")
        st.download_button(
            label="Download as GIF",
            data=create_gif_from_frames(frames),
            file_name=f"{category}_{cluster_id}_timeline.gif",
            mime="image/gif",
            key="download_gif_single_btn",
            use_container_width=True
        )
        progress_bar.empty()
        status_text.empty()
    else:
        st.error("Failed to create timeline export - no frames were generated")
        progress_bar.empty()
        status_text.empty()

def export_all_clusters_timeline(df, category, playback_dates, basemap_tiles, basemap):
    """Export timeline showing all clusters over time, using the same approach as single cluster export"""
    # Filter out noise points
    valid_data = df[df['cluster'] >= 0].copy()
    
    if valid_data.empty:
        st.warning("No valid clusters found to export.")
        return
    
    # Identify the correct date column 
    date_col = None
    for possible_name in ['acq_date', 'date', 'Date', 'ACQ_DATE']:
        if possible_name in valid_data.columns:
            date_col = possible_name
            break
    
    if not date_col:
        st.error("❌ No date column found in data")
        return
        
    # Get all unique dates
    dates_with_data = sorted(valid_data[date_col].unique())
    
    # Check if we have at least 2 dates with data
    if len(dates_with_data) <= 1:
        st.warning(f"Data only spans one date. Timeline export requires data on multiple dates.")
        return
    
    # Set up progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Capture frames for each date
    frames = []
    total_dates = len(dates_with_data)
    
    # Custom colors for fire visualization
    dot_color = '#ff3300'  # Red-orange fill
    border_color = 'white'
    
    for i, date in enumerate(dates_with_data):
        status_text.write(f"Processing frame {i+1}/{total_dates}: {date}")
        progress_bar.progress((i+1)/total_dates)
        
        # Create map for this date
        playback_title = f"All {get_category_display_name(category)}s - {date}"
        
        # Filter data for this date across all clusters
        date_data = valid_data[valid_data[date_col] == date].copy()
        
        if not date_data.empty:
            # Create a simplified map for export using the selected basemap
            folium_map = create_export_map(
                date_data, 
                playback_title, 
                basemap_tiles, 
                basemap,
                dot_color=dot_color,
                border_color=border_color
            )
            frames.append(folium_map)
    
    status_text.write("Processing complete. Preparing download...")
    
    # Store frames in session state
    st.session_state.frames = frames
    
    # Provide download option
    if frames:
        # Create download buffer
        st.info(f"Timeline export ready for all clusters")
        st.download_button(
            label="Download as GIF",
            data=create_gif_from_frames(frames),
            file_name=f"{category}_all_clusters_timeline.gif",
            mime="image/gif",
            key="download_gif_all_btn",
            use_container_width=True
        )
        progress_bar.empty()
        status_text.empty()
    else:
        st.error("Failed to create timeline export - no frames were generated")
        progress_bar.empty()
        status_text.empty()

def plot_fire_detections_folium(df, title="Fire Detections", selected_cluster=None, playback_mode=False, playback_date=None, dot_size_multiplier=1.0, color_palette='inferno', category="fires"):
    """Plot fire detections on a folium map with color palette based on temperature"""
    
    # Create a working copy of the dataframe
    plot_df = df.copy()
    
    # Filter out noise points (-1) if category is not raw data
    if category != "raw data":
        plot_df = plot_df[plot_df['cluster'] >= 0].copy()
    
    # Apply cluster selection filter if a cluster is selected
    if selected_cluster is not None and selected_cluster in plot_df['cluster'].values:
        # Filter for just the selected cluster
        selected_data = plot_df[plot_df['cluster'] == selected_cluster].copy()
        other_data = plot_df[plot_df['cluster'] != selected_cluster].copy()
        
        # Update title if filtering is applied
        category_display = get_category_display_name(category)
        title = f"{title} - {category_display} {selected_cluster}"
        
        # If in playback mode, further filter by date
        if playback_mode and playback_date is not None:
            selected_data = selected_data[selected_data['acq_date'] == playback_date].copy()
            # Add date to title
            title = f"{title} - {playback_date}"
            # No need to show other data in playback mode
            other_data = pd.DataFrame()
            
        # Replace plot_df with filtered version
        plot_df = pd.concat([selected_data, other_data])
    elif playback_mode and playback_date is not None:
        # Filter by date only if no cluster is selected
        plot_df = plot_df[plot_df['acq_date'] == playback_date].copy()
        title = f"{title} - {playback_date}"
    
    # Check if there is any data to plot
    if plot_df.empty:
        st.warning("No data to plot for the selected filters.")
        # Create an empty map with default center if no data
        m = folium.Map(location=[34.0, 65.0], zoom_start=4, control_scale=True, 
                      tiles='cartodbdark_matter')
        
        # Add information about why map is empty
        empty_info = """
        <div style="position: absolute; 
                    top: 50%; 
                    left: 50%; 
                    transform: translate(-50%, -50%);
                    padding: 20px; 
                    background-color: rgba(0,0,0,0.8); 
                    color: white;
                    z-index: 9999; 
                    border-radius: 5px;
                    text-align: center;">
            <h3>No data points to display</h3>
            <p>The selected filters returned no results.</p>
            <p>Try changing your selection criteria.</p>
        </div>
        """
        
        m.get_root().html.add_child(folium.Element(empty_info))
        return m
    
    # Calculate the bounding box for auto-zoom
    min_lat = plot_df['latitude'].min()
    max_lat = plot_df['latitude'].max()
    min_lon = plot_df['longitude'].min()
    max_lon = plot_df['longitude'].max()
    
    # Create a map centered on the mean coordinates with appropriate zoom
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    
    # Determine which temperature column to use
    temp_col = get_temp_column(plot_df)
    
    # Set the initial tiles based on basemap parameter (defaulting to dark if not specified)
    basemap_tiles = {
        'Dark': 'cartodbdark_matter',
        'Light': 'cartodbpositron',
        'Satellite': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        'Terrain': 'stamenterrain'
    }
    
    initial_tiles = 'cartodbdark_matter'
    basemap = st.session_state.get('basemap', 'Dark')
    if basemap in basemap_tiles:
        initial_tiles = basemap_tiles[basemap]
    
    m = folium.Map(location=[center_lat, center_lon], control_scale=True, 
                  tiles=initial_tiles)  # Set the base map from user selection
    
    Fullscreen().add_to(m)
    
    # Automatically zoom to fit all points
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]], padding=(50, 50))
    
    # Add a title to the map
    title_html = f'''
             <h3 align="center" style="font-size:16px; color: white;"><b>{title}</b></h3>
             '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Create feature groups for different sets of points
    fg_all = folium.FeatureGroup(name="All Points")
    fg_selected = folium.FeatureGroup(name="Selected Points")
    
    # Define different color palettes
    color_palettes = {
        'inferno': ['#FCFFA4', '#F8DF3A', '#FB9E3A', '#ED6925', '#D94E11', '#B62A07', '#8B0F07', '#5D0C0C', '#420A68'],  # Reversed to make hottest white/yellow
        'viridis': ['#FDE725', '#BBDF27', '#6DCE59', '#35B779', '#1F9E89', '#26828E', '#31688E', '#3E4989', '#482878'],  # Reversed for visibility
        'plasma': ['#F0F921', '#FCCE25', '#FCA636', '#F1844B', '#E16462', '#CC4778', '#B12A90', '#8F0DA4', '#6A00A8'],   # Reversed
        'magma': ['#FCFDBF', '#FECA8D', '#FD9668', '#F1605D', '#CD4071', '#9E2F7F', '#721F81', '#440F76', '#180F3D'],    # Reversed
        'cividis': ['#FEE838', '#E1CC55', '#C3B369', '#A59C74', '#8A8678', '#707173', '#575D6D', '#3B496C', '#123570']   # Reversed
    }
    
    # Create colormap for temperature
    if temp_col:
        # Get selected color palette
        selected_palette = color_palettes.get(color_palette, color_palettes['inferno'])
        
        vmin = plot_df[temp_col].min()
        vmax = plot_df[temp_col].max()
        colormap = LinearColormap(
            selected_palette,
            vmin=vmin, 
            vmax=vmax,
            caption=f'Temperature (K)'
        )
    
    # Base dot sizes, will be multiplied by the dot_size_multiplier
    base_small_dot = 5 * dot_size_multiplier
    base_medium_dot = 6 * dot_size_multiplier
    base_large_dot = 8 * dot_size_multiplier
    
    # Process data based on selection state
    if selected_cluster is not None and selected_cluster in plot_df['cluster'].values:
        # Split data into selected and unselected
        selected_data = plot_df[plot_df['cluster'] == selected_cluster]
        other_data = plot_df[plot_df['cluster'] != selected_cluster]
        
        # Add unselected clusters if not in playback mode
        if not other_data.empty and not playback_mode:
            for idx, point in other_data.iterrows():
                if temp_col and not pd.isna(point[temp_col]):
                    color = colormap(point[temp_col])
                
                popup_text = f"""
                <b>Cluster:</b> {point['cluster']}<br>
                <b>Date:</b> {point['acq_date']}<br>
                <b>Time:</b> {point['acq_time']}<br>
                <b>FRP:</b> {point['frp']:.2f}<br>
                <b>Coordinates:</b> {point['latitude']:.4f}, {point['longitude']:.4f}<br>
                """
                if temp_col and not pd.isna(point[temp_col]):
                    popup_text += f"<b>Temperature:</b> {point[temp_col]:.2f}K<br>"
                
                circle = folium.CircleMarker(
                    location=[point['latitude'], point['longitude']],
                    radius=base_small_dot,
                    color='white',  # Use white border for visibility on dark background
                    weight=0.5,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.9,  # Increased opacity for better visibility
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"Cluster {point['cluster']} - ({point['latitude']:.4f}, {point['longitude']:.4f})"
                )
                
                circle.add_to(fg_all)
        
        # Add selected cluster with different style
        if not selected_data.empty:
            for idx, point in selected_data.iterrows():
                if temp_col and not pd.isna(point[temp_col]):
                    color = colormap(point[temp_col])
                else:
                    color = '#ff3300'  # Default red
                
                popup_text = f"""
                <b>Cluster:</b> {point['cluster']}<br>
                <b>Date:</b> {point['acq_date']}<br>
                <b>Time:</b> {point['acq_time']}<br>
                <b>FRP:</b> {point['frp']:.2f}<br>
                <b>Coordinates:</b> {point['latitude']:.4f}, {point['longitude']:.4f}<br>
                """
                if temp_col and not pd.isna(point[temp_col]):
                    popup_text += f"<b>Temperature:</b> {point[temp_col]:.2f}K<br>"
                
                folium.CircleMarker(
                    location=[point['latitude'], point['longitude']],
                    radius=base_large_dot,
                    color='white',  # White border for visibility
                    weight=1.5,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.9,
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"Cluster {point['cluster']} - Selected - ({point['latitude']:.4f}, {point['longitude']:.4f})"
                ).add_to(fg_selected)
    else:
        # Add all points with default style
        for idx, point in plot_df.iterrows():
            if temp_col and not pd.isna(point[temp_col]):
                color = colormap(point[temp_col])
            else:
                color = '#3186cc'  # Default blue

            # Create a popup with a URL parameter-based cluster selection
            popup_html = f"""
            <div style="text-align: center;">
                <p><b>Cluster:</b> {point['cluster']}</p>
                <p><b>Date:</b> {point['acq_date']}</p>
                <p><b>Time:</b> {point['acq_time']}</p>
                <p><b>FRP:</b> {point['frp']:.2f}</p>
                <p><b>Coordinates:</b> {point['latitude']:.4f}, {point['longitude']:.4f}</p>
                <p style="color: #4CAF50; font-weight: bold;">
                    To select this cluster, use the dropdown menu below the map
                </p>
            </div>
            """
            
            popup = folium.Popup(popup_html, max_width=300)

            folium.CircleMarker(
                location=[point['latitude'], point['longitude']],
                radius=base_medium_dot,
                color='white',  # White border for visibility
                weight=0.5,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,  # Increased opacity
                popup=popup,
                tooltip=f"Cluster {point['cluster']} - ({point['latitude']:.4f}, {point['longitude']:.4f})"
            ).add_to(fg_all)

    fg_all.add_to(m)
    fg_selected.add_to(m)

    # Add base layers with explicit names
    folium.TileLayer(
        'cartodbpositron', 
        name='Light Map',
        attr='© <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, © <a href="https://carto.com/attribution">CARTO</a>'
    ).add_to(m)

    folium.TileLayer(
        'cartodbdark_matter', 
        name='Dark Map',
        attr='© <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, © <a href="https://carto.com/attribution">CARTO</a>'
    ).add_to(m)

    folium.TileLayer(
        'stamenterrain', 
        name='Terrain Map',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.'
    ).add_to(m)

    folium.TileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        name='Satellite',
        attr='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
    ).add_to(m)

    # Now add the layer control after all layers are added
    folium.LayerControl(position='topright').add_to(m)

    # If you have a temperature colormap, add it after the layer control
    if temp_col:
        colormap.add_to(m)
    
    # Add an interaction explanation with instructions to use UI for selection
    info_text = """
    <div style="position: fixed; 
                bottom: 20px; 
                left: 10px; 
                padding: 10px; 
                background-color: rgba(0,0,0,0.7); 
                color: white;
                z-index: 9999; 
                border-radius: 5px;
                max-width: 300px;
                font-size: 12px;">
        <b>Interaction:</b><br>
        • Hover over points to see details<br>
        • Click points to view full information<br>
        • Use the dropdown menu below the map to select clusters<br>
        • Zoom with +/- or mouse wheel<br>
        • Change base maps with layer control (top right)
    </div>
    """
    m.get_root().html.add_child(folium.Element(info_text))
    
    return m

def create_cluster_summary(df, category="fires"):
    """Create summary statistics for each cluster"""
    if df is None or df.empty:
        return None
    
    # Filter out noise points (-1) if category is not raw data
    if category != "raw data":
        summary_df = df[df['cluster'] >= 0].copy()
    else:
        summary_df = df.copy()
        
    if summary_df.empty:
        st.warning("No valid clusters found after filtering.")
        return None
        
    cluster_summary = (summary_df
                      .groupby('cluster')
                      .agg({
                          'latitude': ['count', 'mean'],
                          'longitude': 'mean',
                          'frp': ['mean', 'sum'],
                          'acq_date': ['min', 'max']
                      })
                      .round(3))
    
    cluster_summary.columns = [
        'Number of Points', 'Mean Latitude', 'Mean Longitude',
        'Mean FRP', 'Total FRP', 'First Detection', 'Last Detection'
    ]
    
    # Add temperature statistics based on dataset type
    temp_col = get_temp_column(df)
    if temp_col:
        temp_stats = summary_df.groupby('cluster')[temp_col].agg(['mean', 'max']).round(2)
        cluster_summary['Mean Temperature'] = temp_stats['mean']
        cluster_summary['Max Temperature'] = temp_stats['max']
    
    return cluster_summary.reset_index()

def plot_feature_time_series(df, cluster_id, features):
    """Generate time series plots for selected features of a cluster with robust error handling"""
    if df is None or df.empty or cluster_id is None:
        return None
    
    # Filter for the selected cluster
    cluster_data = df[df['cluster'] == cluster_id].copy()
    
    if cluster_data.empty:
        return None
    
    # Create a daily summary with mean, max for each feature
    daily_data = []
    
    for date in sorted(cluster_data['acq_date'].unique()):
        day_data = {'date': date}
        day_df = cluster_data[cluster_data['acq_date'] == date]
        
        # Calculate stats for requested features
        for feature in features:
            if feature in cluster_data.columns:
                # Calculate values with safety checks
                mean_val = day_df[feature].mean()
                max_val = day_df[feature].max()
                min_val = day_df[feature].min()
                count_val = day_df[feature].count()
                
                # Check for invalid values
                if not (pd.isna(mean_val) or np.isinf(mean_val)):
                    day_data[f'{feature}_mean'] = mean_val
                else:
                    day_data[f'{feature}_mean'] = None
                    
                if not (pd.isna(max_val) or np.isinf(max_val)):
                    day_data[f'{feature}_max'] = max_val
                else:
                    day_data[f'{feature}_max'] = None
                    
                if not (pd.isna(min_val) or np.isinf(min_val)):
                    day_data[f'{feature}_min'] = min_val
                else:
                    day_data[f'{feature}_min'] = None
                    
                day_data[f'{feature}_count'] = count_val
                
        # Only add days with valid data
        has_valid_data = False
        for key in day_data:
            if key != 'date' and day_data[key] is not None:
                has_valid_data = True
                break
                
        if has_valid_data:
            daily_data.append(day_data)
    
    # Create dataframe from daily summaries
    if not daily_data:
        return None
        
    daily_df = pd.DataFrame(daily_data)
    
    # Convert date column to datetime - with error handling
    try:
        daily_df['date'] = pd.to_datetime(daily_df['date'])
    except Exception as e:
        st.warning(f"Error converting dates: {str(e)}")
        # Use index as fallback
        daily_df['date'] = range(len(daily_df))
    
    # Feature display names and descriptions
    feature_info = {
        'frp': {
            'display_name': 'Fire Radiative Power (MW)',
            'description': 'Fire Radiative Power (FRP) measures the rate of emitted energy from a fire in megawatts (MW). Higher values indicate more intense burning.'
        },
        'bright_ti4': {
            'display_name': 'Brightness (K)',
            'description': 'Brightness Temperature is measured in Kelvin (K) and indicates how hot the fire is. Higher values indicate hotter fires.'
        }
    }
    
    # Create chart data for combined visualization - with proper filtering
    chart_data = pd.DataFrame({'date': daily_df['date']})
    
    # Add data for each selected feature, filtering out invalid values
    for feature in features:
        feature_key = f'{feature}_mean'
        if feature_key in daily_df.columns:
            feature_display = feature_info.get(feature, {}).get('display_name', feature)
            
            # First replace inf with NaN
            if feature_key in daily_df:
                daily_df[feature_key].replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Only add columns with valid data
            if feature_key in daily_df and not daily_df[feature_key].isna().all():
                chart_data[feature_display] = daily_df[feature_key]
    
    # If no valid feature data, return None
    if len(chart_data.columns) <= 1:  # Only has 'date' column
        return None
        
    # Remove any rows with NaN values to prevent chart errors
    chart_data = chart_data.dropna()
    if chart_data.empty:
        return None
        
    # Make sure date column has at least 2 unique values
    if len(chart_data['date'].unique()) < 2:
        st.info("Not enough time points to create a useful chart.")
        return None
    
    # Melt the dataframe for Altair
    try:
        melted_data = pd.melt(
            chart_data, 
            id_vars=['date'], 
            var_name='Feature', 
            value_name='Value'
        )
        
        # Final check for infinite values
        melted_data = melted_data[~np.isinf(melted_data['Value'])]
        
        # Create chart with robust error handling
        combined_chart = alt.Chart(melted_data).mark_line(point=True).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('Value:Q', title='Value', scale=alt.Scale(zero=False)),
            color=alt.Color('Feature:N', legend=alt.Legend(title='Feature')),
            tooltip=['date:T', 'Value:Q', 'Feature:N']
        ).properties(
            title='Fire Evolution Over Time',
            width=600,
            height=300
        ).interactive()
        
        return combined_chart, feature_info
    except Exception as e:
        st.warning(f"Error creating chart: {str(e)}")
        return None

def display_feature_exploration(df, cluster_id, category, current_date=None, caller_id=None):
    """Display feature exploration interface for the selected cluster"""
    if df is None or df.empty or cluster_id is None:
        return
    
    # Create a unique key suffix based on caller and date if needed
    key_suffix = f"_{caller_id}" if caller_id else ""
    if current_date:
        key_suffix += f"_{current_date}"
    
    # Filter data for the selected cluster
    cluster_data = df[df['cluster'] == cluster_id].copy()
    
    # If in playback mode and a date is provided, filter for that date
    if current_date is not None:
        cluster_data = cluster_data[cluster_data['acq_date'] == current_date].copy()
    
    if cluster_data.empty:
        st.warning(f"No data available for selected {get_category_display_name(category).lower()}.")
        return
    
    # Limit to only 'frp' and 'bright_ti4' features
    available_features = []
    
    if 'frp' in cluster_data.columns:
        available_features.append('frp')
        
    temp_col = get_temp_column(df)
    if temp_col and temp_col in cluster_data.columns:
        available_features.append(temp_col)
    
    # Fixed features with better names
    feature_display_names = {
        'frp': 'Fire Radiative Power',
        'bright_ti4': 'Brightness'
    }
    
    # Feature selection checkboxes - horizontal arrangement
    category_display = get_category_display_name(category)
    
    if current_date is not None:
        st.write(f"### {category_display} {cluster_id} Data for {current_date}")
    else:
        st.write(f"### {category_display} {cluster_id} Evolution Over Time")
    
    cols = st.columns([1, 1, 3])
    
    selected_features = []
    
    frp_key = f"show_frp_{cluster_id}{key_suffix}"
    temp_key = f"show_temp_{cluster_id}{key_suffix}"
    
    with cols[0]:
        if 'frp' in available_features:
            show_frp = st.checkbox("Fire Radiative Power", value=True, key=frp_key)
            if show_frp:
                selected_features.append('frp')
    
    with cols[1]:
        if temp_col in available_features:
            show_temp = st.checkbox("Brightness", value=False, key=temp_key)
            if show_temp:
                selected_features.append(temp_col)
    
    # Generate and display a single combined chart for selected features
    if selected_features:
        # If we're in playback mode, we can just show a simple summary for the current date
        if current_date is not None:
            # Display a simple summary table for this date
            if not cluster_data.empty:
                summary = {}
                
                for feature in selected_features:
                    if feature in cluster_data.columns:
                        mean_val = cluster_data[feature].mean()
                        max_val = cluster_data[feature].max()
                        if not (np.isnan(mean_val) or np.isinf(mean_val) or np.isnan(max_val) or np.isinf(max_val)):
                            feature_name = feature_display_names.get(feature, feature)
                            summary[f"Average {feature_name}"] = f"{mean_val:.2f}"
                            summary[f"Maximum {feature_name}"] = f"{max_val:.2f}"
                
                # Display the summary in metrics
                metric_cols = st.columns(len(summary))
                for i, (key, value) in enumerate(summary.items()):
                    with metric_cols[i]:
                        st.metric(key, value)
            else:
                st.info("No data available for this date.")
        else:
            # We're in normal mode, show the time series chart
            # Unpack the tuple correctly - we expect (chart, feature_info)
            result = plot_feature_time_series(df, cluster_id, selected_features)
            
            if result and isinstance(result, tuple) and len(result) == 2:
                chart, feature_info = result
                
                # Display chart
                st.altair_chart(chart, use_container_width=True)
                
                # Add hover explanations
                with st.expander("What do these metrics mean?"):
                    for feature in selected_features:
                        if feature in feature_info:
                            st.write(f"**{feature_info[feature]['display_name']}**: {feature_info[feature]['description']}")
            else:
                st.warning("Not enough time-series data to generate chart.")
    else:
        st.info("Please select at least one feature to visualize.")
        


def display_coordinate_view(df, playback_date=None):
    """Display a table with coordinates and details for the selected cluster"""
    if df is None or df.empty:
        st.info("No data available to display coordinates.")
        return
    
    if st.session_state.selected_cluster is not None:
        # Filter for the selected cluster
        cluster_points = df[df['cluster'] == st.session_state.selected_cluster]
        
        # If in playback mode, further filter by date
        if playback_date is not None:
            cluster_points = cluster_points[cluster_points['acq_date'] == playback_date]
        
        if not cluster_points.empty:
            if playback_date is not None:
                st.subheader(f"Points in Cluster {st.session_state.selected_cluster} on {playback_date}")
            else:
                if not st.session_state.playback_mode:
                    st.subheader(f"Point Details in Cluster {st.session_state.selected_cluster}")
                else:
                    st.subheader(f"Point Details in Cluster {st.session_state.selected_cluster} on {st.session_state.playback_dates[st.session_state.playback_index]}")
            
            # Create a display version of the dataframe with formatted columns
            display_df = cluster_points[['latitude', 'longitude', 'frp', 'acq_date', 'acq_time']].copy()
            
            # Add temperature column if available
            temp_col = get_temp_column(df)
            if temp_col:
                display_df['temperature'] = cluster_points[temp_col]
            
            # Add a formatted coordinate column
            display_df['Coordinates'] = display_df.apply(
                lambda row: f"{row['latitude']:.4f}, {row['longitude']:.4f}", 
                axis=1
            )
            
            # Display columns
            display_columns = ['Coordinates', 'frp', 'acq_date', 'acq_time']
            if temp_col:
                display_columns.append('temperature')
                
            # Display the dataframe
            st.dataframe(
                display_df[display_columns],
                column_config={
                    "Coordinates": "Lat, Long",
                    "frp": st.column_config.NumberColumn("FRP", format="%.2f"),
                    "acq_date": "Date",
                    "acq_time": "Time",
                    "temperature": st.column_config.NumberColumn("Temp (K)", format="%.2f")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning(f"No points found for Cluster {st.session_state.selected_cluster}" + 
                      (f" on {playback_date}" if playback_date is not None else ""))
    else:
        st.info("Select a cluster to view detailed point information.")

def create_arrow_navigation(key_suffix=""):
    """Create arrow navigation buttons and JavaScript for keyboard navigation"""
    # Create button columns for navigation
    col1, col2, col3 = st.columns([1, 4, 1])
    
    with col1:
        prev_key = f"prev_btn_{key_suffix}" if key_suffix else "prev_btn"
        prev_clicked = st.button("◀", key=prev_key, help="Previous Date (Left Arrow)", on_click=None)
        if prev_clicked and st.session_state.get('playback_index', 0) > 0:
            st.session_state.playback_index -= 1
            st.rerun()
    
    with col2:
        playback_dates = st.session_state.get('playback_dates', [])
        playback_index = st.session_state.get('playback_index', 0)
        
        if playback_dates and st.session_state.get('playback_mode', False):
            if playback_index < len(playback_dates):
                current_date = playback_dates[playback_index]
                total_dates = len(playback_dates)
                
                # Date slider
                slider_key = f"date_slider_{key_suffix}" if key_suffix else "date_slider_direct"
                date_index = st.slider(
                    "Select Date", 
                    0, 
                    total_dates - 1, 
                    playback_index,
                    key=slider_key,
                    help="Use slider or arrow buttons to change the date"
                )
                
                # Update index if slider changed
                if date_index != playback_index:
                    st.session_state.playback_index = date_index
                    st.rerun()
                    
                st.write(f"**Current Date: {current_date}** (Day {playback_index + 1} of {total_dates})")
    
    with col3:
        next_key = f"next_btn_{key_suffix}" if key_suffix else "next_btn"
        next_clicked = st.button("▶", key=next_key, help="Next Date (Right Arrow)", on_click=None)
        playback_dates = st.session_state.get('playback_dates', [])
        playback_index = st.session_state.get('playback_index', 0)
        
        if next_clicked and playback_index < len(playback_dates) - 1:
            st.session_state.playback_index += 1
            st.rerun()

def handle_url_parameters():
    """Handle URL parameters for cluster selection"""
    # Check if selected_cluster parameter exists
    if 'selected_cluster' in st.query_params:
        try:
            # Get cluster ID from URL parameters
            cluster_id = int(st.query_params['selected_cluster'])
            
            # Clear the URL parameter immediately to prevent reprocessing
            del st.query_params['selected_cluster']
            
            # Skip further processing if there's no valid results
            if 'results' not in st.session_state or st.session_state.results is None:
                return
                
            # Check if it's a new selection
            if st.session_state.get('selected_cluster') != cluster_id:
                # Update the selected cluster in session state
                st.session_state.selected_cluster = cluster_id
                
                # Get unique dates for the selected cluster
                cluster_points = st.session_state.results[st.session_state.results['cluster'] == cluster_id]
                unique_dates = sorted(cluster_points['acq_date'].unique())
                
                # Store the dates and initialize to the first one
                st.session_state.playback_dates = unique_dates
                st.session_state.playback_index = 0
                
                # Enable playback mode when selecting from map
                st.session_state.playback_mode = True
                
                # Rerun to update UI
                st.rerun()
        except (ValueError, TypeError):
            # Invalid cluster ID, ignore it
            if 'selected_cluster' in st.query_params:
                del st.query_params['selected_cluster']

def main():
    # Create a two-column layout for the main interface
    main_cols = st.columns([1, 3])
    
    with main_cols[0]:
        # Analysis Settings Section
        st.subheader("Analysis Settings")
        
        # Country selection
        st.write("Please select your location")
        country_options = {
            'Afghanistan': '60.52,29.31,75.15,38.48',
            'United States': '-125.0,24.0,-66.0,50.0',
            'Brazil': '-73.0,-33.0,-35.0,5.0',
            'Australia': '113.0,-44.0,154.0,-10.0',
            'India': '68.0,7.0,97.0,37.0',
            'China': '73.0,18.0,135.0,53.0',
            'Canada': '-141.0,41.7,-52.6,83.0',
            'Russia': '19.25,41.151,180.0,81.2',
            'Indonesia': '95.0,-11.0,141.0,6.0',
            # ... other countries would be here
        }
        country = st.selectbox(
            "Please select",
            list(country_options.keys())
        )
        
        # Dataset selection - keep checkboxes
        st.subheader("Select Datasets")
        
        datasets = {}
        datasets['VIIRS_NOAA20_NRT'] = st.checkbox("VIIRS NOAA-20", value=True)
        datasets['VIIRS_SNPP_NRT'] = st.checkbox("VIIRS SNPP", value=True)
        datasets['MODIS_NRT'] = st.checkbox("MODIS", value=True)
        
        # Determine which datasets are selected
        selected_datasets = [ds for ds, is_selected in datasets.items() if is_selected]
        if selected_datasets:
            dataset = selected_datasets[0]  # Use the first selected dataset
        else:
            st.warning('Please select at least one dataset')
            dataset = None
        
        # Category selection
        st.subheader("Select Category")
        category = st.selectbox(
            "Thermal Detection Type",
            ["fires", "flares", "raw data"],
            key="category_select",
            help="""
            Fires: Temperature > 300K, FRP > 1.0 (VIIRS) or Confidence > 80% (MODIS)
            Gas Flares: Temperature > 1000K, typically industrial sources
            Raw Data: All data points including noise points not assigned to clusters
            """
        )
        
        # Date range selection
        st.subheader("Select Date Range")
        today = datetime.now()
        default_end_date = today
        default_start_date = default_end_date - timedelta(days=7)
        
        date_cols = st.columns(2)
        
        with date_cols[0]:
            start_date = st.date_input(
                "Start Date",
                value=default_start_date,
                max_value=today
            )
        
        with date_cols[1]:
            end_date = st.date_input(
                "End Date",
                value=default_end_date,
                min_value=start_date,
                max_value=today
            )
            
        
        
        # Calculate date range in days
        date_range_days = (end_date - start_date).days
        
        # Define large countries that might be slow with wide date ranges
        large_countries = ['United States', 'China', 'Russia', 'Canada', 'Brazil', 'Australia', 'India']
        
        # Show warning for large countries with wide date ranges
        if country in large_countries and date_range_days > 14:
            st.warning(f"⚠️ You selected a {date_range_days}-day period for {country}, which is a large country. This may take a long time to process. Consider reducing your date range to 14 days or less for faster results.")
        
        with st.expander("Advanced Clustering Settings"):
            # Two-column layout for clustering parameters
            clust_cols = st.columns(2)
            
            with clust_cols[0]:
                eps_val = st.slider("Spatial Proximity (eps)", 0.005, 0.05, value=0.01, step=0.001, 
                                    help="DBSCAN eps parameter. Higher values create larger clusters.")
            
            with clust_cols[1]:
                min_samples_val = st.slider("Minimum Points", 3, 15, value=5, step=1,
                                        help="Minimum points required to form a cluster.")
                
            use_clustering = st.checkbox("Use Clustering", value=True, 
                                    help="Group nearby detections into clusters for easier analysis.")
                                    
            # Add time-based clustering parameter
            max_time_diff = st.slider("Max Days Between Events (Same Cluster)", 1, 10, value=5, step=1,
                                    help="Maximum days between fire events to be considered same cluster. Lower values create more temporally distinct clusters.")
            
            show_multiday_only = st.checkbox("Show only multi-day fires", value=False,
                               help="Filter to show only fires that span multiple days")
                
        # API credentials (hidden in expander)
        with st.expander("API Settings"):
            username = st.text_input("FIRMS Username", value="tombrown4444")
            password = st.text_input("FIRMS Password", value="wft_wxh6phw9URY-pkv", type="password")
            api_key = st.text_input("FIRMS API Key", value="897a9b7869fd5e4ad231573e14e1c8c8")
        
        # Generate button - make it bigger
        st.markdown('<style>.stButton button { font-size: 20px; padding: 15px; }</style>', unsafe_allow_html=True)
        generate_button = st.button("Generate Analysis", key="generate_button", use_container_width=True)
        
        if generate_button:
            with st.spinner("Analyzing fire data..."):
                handler = FIRMSHandler(username, password, api_key)
                results = handler.fetch_fire_data(
                    country=country,
                    dataset=dataset,
                    category=category,
                    start_date=start_date,
                    end_date=end_date,
                    use_clustering=use_clustering,
                    eps=eps_val,
                    min_samples=min_samples_val,
                    chunk_days=7,
                    max_time_diff_days=max_time_diff
                )
                
            # MULTI-DAY FILTERING CODE
            if show_multiday_only and results is not None and not results.empty:
                # Standardize column names for date (add this)
                if 'Date' in results.columns and 'acq_date' not in results.columns:
                    results['acq_date'] = results['Date']
                elif 'date' in results.columns and 'acq_date' not in results.columns:
                    results['acq_date'] = results['date']
                # Debug information - before filtering
                all_clusters = results[results['cluster'] >= 0]['cluster'].unique()
                st.write(f"Found {len(all_clusters)} clusters before filtering")
                
                # Count days per cluster
                cluster_days = results[results['cluster'] >= 0].groupby('cluster')['acq_date'].nunique()
                
                # More detailed information
                for cluster_id, day_count in cluster_days.items():
                    dates = sorted(results[results['cluster'] == cluster_id]['acq_date'].unique())
                    date_range = f"{dates[0]} to {dates[-1]}" if len(dates) > 1 else dates[0]
                    st.write(f"Cluster {cluster_id}: {day_count} days ({date_range})")
                
                # Get multi-day clusters
                multiday_clusters = cluster_days[cluster_days > 1].index.tolist()
                
                # Filter results to keep only multi-day clusters
                if multiday_clusters:
                    multi_day_mask = results['cluster'].isin(multiday_clusters)
                    filtered_results = results[multi_day_mask].copy()
                    st.success(f"✓ Filtered to {len(multiday_clusters)} clusters that span multiple days")
                    results = filtered_results
                else:
                    st.warning("⚠ No multi-day fire clusters found. Try adjusting clustering parameters or date range.")

                if not multiday_clusters:
                    st.warning("⚠ No multi-day fire clusters found. The clustering algorithm didn't find any fires spanning multiple days. Try increasing the 'Max Days Between Events' slider or adjust the 'Spatial Proximity' value.")
            
            # Store results in session state
            st.session_state.results = results
            # Reset selected cluster
            st.session_state.selected_cluster = None
            # Reset playback mode
            st.session_state.playback_mode = False
                
    with main_cols[1]:
        # Handle URL parameters
        handle_url_parameters()
        
        # Main map and visualization section
        if 'results' in st.session_state and st.session_state.results is not None and not st.session_state.results.empty:
            # Get category display name for UI
            category_display = get_category_display_name(category)
            
            st.subheader(f"Detection Map")
            
            # Always show the export button
            export_col1, export_col2 = st.columns(2)
            with export_col1:
                if st.button("Export All Clusters Timeline", key="export_all_btn", use_container_width=True):
                    # Define the basemap_tiles dictionary
                    basemap_tiles = {
                        'Dark': 'cartodbdark_matter',
                        'Light': 'cartodbpositron',
                        'Satellite': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                        'Terrain': 'stamenterrain'
                    }
                    
                    # Get all dates from the dataset for playback
                    all_dates = sorted(st.session_state.results[st.session_state.results['cluster'] >= 0]['acq_date'].unique())
                    
                    # Export timeline for all clusters (passing None as cluster_id)
                    export_timeline(
                        st.session_state.results, 
                        None,  # Export all clusters
                        category,
                        all_dates,
                        basemap_tiles,
                        "Dark"
                    )
            
            # Set up variables for map creation
            map_settings = {
                'color_palette': 'inferno',
                'basemap': 'Dark',
                'dot_size_multiplier': 1.0
            }
            
            # Check if we're in playback mode
            if not st.session_state.get('playback_mode', False):
                # NORMAL MODE - Create the folium visualization
                with st.spinner("Generating map..."):
                    folium_map = plot_fire_detections_folium(
                        st.session_state.results, 
                        f"{category_display} Clusters - {country}", 
                        st.session_state.get('selected_cluster'),
                        category=category,
                        color_palette=map_settings.get('color_palette', 'inferno'),
                        dot_size_multiplier=map_settings.get('dot_size_multiplier', 1.0)
                    )
                
                if folium_map:
                    # Display the folium map
                    html_map = folium_map._repr_html_()
                    components.html(html_map, height=550, width=None)
                else:
                    st.warning("No data to display on the map.")
            else:
                # PLAYBACK MODE - Timeline view
                playback_dates = st.session_state.get('playback_dates', [])
                playback_index = st.session_state.get('playback_index', 0)
                
                if playback_dates and playback_index < len(playback_dates):
                    current_date = playback_dates[playback_index]
                    
                    # Create the playback visualization
                    playback_title = f"{category_display} {st.session_state.get('selected_cluster')} - {current_date}"
                    
                    folium_map = plot_fire_detections_folium(
                        st.session_state.results,
                        playback_title,
                        st.session_state.get('selected_cluster'),
                        True,
                        current_date,
                        category=category,
                        color_palette=map_settings.get('color_palette', 'inferno'),
                        dot_size_multiplier=map_settings.get('dot_size_multiplier', 1.0)
                    )
                    
                    if folium_map:
                        # Save the map to an HTML string and display it
                        html_map = folium_map._repr_html_()
                        components.html(html_map, height=550, width=None)
                    else:
                        st.warning("No data to display for the current date.")
            
            # Display cluster selection dropdown
            st.markdown("---")
            st.subheader("Select a Cluster for Analysis")
            
            if 'results' in st.session_state and st.session_state.results is not None:
                # Create cluster summary for dropdown
                cluster_summary = create_cluster_summary(st.session_state.results, category)
                
                if cluster_summary is not None:
                    # Get all valid clusters
                    clusters = sorted(cluster_summary['cluster'].tolist())
                    
                    # Create dropdown options with additional info
                    options = ["None"]
                    for cluster_id in clusters:
                        cluster_info = cluster_summary[cluster_summary['cluster'] == cluster_id]
                        points = cluster_info['Number of Points'].values[0]
                        frp = cluster_info['Mean FRP'].values[0]
                        options.append(f"Cluster {cluster_id} ({points} points, FRP: {frp:.2f})")
                    
                    # Display the dropdown
                    selected = st.selectbox(
                        "Choose a cluster to view details and timeline",
                        options=options,
                        key="cluster_selectbox"
                    )
                    
                    # Extract cluster ID if selected
                    if selected != "None":
                        cluster_match = re.search(r"Cluster (\d+)", selected)
                        if cluster_match:
                            selected_cluster = int(cluster_match.group(1))
                            
                            # Update session state if different from current selection
                            if selected_cluster != st.session_state.get('selected_cluster'):
                                st.session_state.selected_cluster = selected_cluster
                                
                                # Get dates for this cluster
                                cluster_data = st.session_state.results[
                                    st.session_state.results['cluster'] == selected_cluster
                                ]
                                unique_dates = sorted(cluster_data['acq_date'].unique())
                                
                                # Store dates in session state
                                st.session_state.playback_dates = unique_dates
                                st.session_state.playback_index = 0
                                
                                # Reset playback mode when selecting a new cluster
                                st.session_state.playback_mode = False
                                
                                # Rerun to update UI
                                st.rerun()
                    else:
                        # If "None" selected, clear the selected cluster
                        if st.session_state.get('selected_cluster') is not None:
                            st.session_state.selected_cluster = None
                            st.session_state.playback_mode = False
                            st.rerun()
            
            # If a cluster is selected, show timeline options and analysis
            if st.session_state.get('selected_cluster') is not None:
                cluster_data = st.session_state.results[
                    st.session_state.results['cluster'] == st.session_state.selected_cluster
                ]
                unique_dates = sorted(cluster_data['acq_date'].unique())
                
                # Show timeline options if there are multiple dates
                if len(unique_dates) > 1:
                    st.markdown("---")
                    st.subheader(f"Timeline Options for Cluster {st.session_state.selected_cluster}")
                    
                    # Display timeline controls based on playback mode
                    if st.session_state.get('playback_mode', False):
                        # We're in playback mode - show navigation controls
                        create_arrow_navigation(key_suffix="main_view")
                        
                        # Exit button
                        if st.button("Exit Timeline View", key="exit_timeline_btn", use_container_width=True):
                            st.session_state.playback_mode = False
                            st.rerun()
                    else:
                        # Not in playback mode - show options to enter timeline or export
                        cols = st.columns(2)
                        with cols[0]:
                            if st.button("View Timeline", key="view_timeline_btn", use_container_width=True):
                                # Enable playback mode
                                st.session_state.playback_mode = True
                                st.session_state.playback_dates = unique_dates
                                st.session_state.playback_index = 0
                                st.rerun()
                        
                        with cols[1]:
                            if st.button("Export Timeline", key="export_timeline_btn", use_container_width=True):
                                # Define the basemap_tiles dictionary if not defined globally
                                basemap_tiles = {
                                    'Dark': 'cartodbdark_matter',
                                    'Light': 'cartodbpositron',
                                    'Satellite': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                                    'Terrain': 'stamenterrain'
                                }
                                
                                export_timeline(
                                    st.session_state.results, 
                                    st.session_state.selected_cluster,
                                    category,
                                    unique_dates,
                                    basemap_tiles,                   # This parameter was missing
                                    map_settings.get('basemap', 'Dark')
                                )
                
                # Show feature exploration for the selected cluster
                st.markdown("---")
                
                # If in playback mode, show data for current date
                if st.session_state.get('playback_mode', False):
                    current_date = st.session_state.playback_dates[st.session_state.playback_index]
                    display_feature_exploration(
                        st.session_state.results,
                        st.session_state.selected_cluster,
                        category,
                        current_date,
                        caller_id="playback_view"
                    )
                else:
                    # Show overall data
                    display_feature_exploration(
                        st.session_state.results,
                        st.session_state.selected_cluster,
                        category,
                        caller_id="normal_view"
                    )
                
                # Show coordinate table at the bottom
                st.markdown("---")
                
                # Display coordinates filtered by date if in playback mode
                if st.session_state.get('playback_mode', False):
                    current_date = st.session_state.playback_dates[st.session_state.playback_index]
                    display_coordinate_view(st.session_state.results, current_date)
                else:
                    display_coordinate_view(st.session_state.results)
        else:
            # No results yet
            st.info("Select settings and click 'Generate Analysis' to visualize fire data.")

if __name__ == "__main__":
    main()