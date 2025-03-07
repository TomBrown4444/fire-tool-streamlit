import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, OPTICS
import hdbscan
import requests
from datetime import datetime, timedelta
from io import StringIO
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
# Add folium related imports
import folium
from folium.plugins import MarkerCluster, HeatMap
from branca.colormap import LinearColormap
import streamlit.components.v1 as components

# Set page config
st.set_page_config(
    page_title="Fire Analysis Tool",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Fire Analysis Tool")
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

class FIRMSHandler:
    def __init__(self, username, password, api_key):
        self.username = username
        self.password = password
        self.api_key = api_key
        self.base_url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        self.session = requests.Session()
        self.session.auth = (username, password)

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

    def _apply_dbscan(self, df, eps=0.01, min_samples=5, bbox=None):
        """Apply DBSCAN clustering with bbox filtering"""
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
        min_samples=5
    ):
        """Fetch and process fire data"""
        dataset_start_dates = {
            'MODIS_NRT': '2000-11-01',
            'VIIRS_SNPP_NRT': '2012-01-19',
            'VIIRS_NOAA20_NRT': '2018-01-01',
            'VIIRS_NOAA21_NRT': '2023-01-01'
        }
        
        if dataset not in dataset_start_dates:
            st.error(f"Invalid dataset. Choose from: {list(dataset_start_dates.keys())}")
            return None

        if not bbox and country:
            bbox = self.get_country_bbox(country)
        
        if not bbox:
            st.error("Provide a country or bounding box")
            return None

        # Check if the country is large and show a message
        large_countries = ['United States', 'China', 'Russia', 'Canada', 'Brazil', 'Australia', 'India']
        if country in large_countries:
            st.info(f"Fetching data for {country}, which may take longer due to the size of the country. Please be patient...")
            
        url = f"{self.base_url}{self.api_key}/{dataset}/{bbox}/7"
        
        with st.spinner('Fetching data...'):
            try:
                response = self.session.get(url)
                response.raise_for_status()
                df = pd.read_csv(StringIO(response.text))
                
                st.write("Raw Data Information:")
                st.write(f"Total records: {len(df)}")
                
                if len(df) == 0:
                    st.warning(f"No records found for {category} in {country}")
                    return None
                
                if use_clustering:
                    df = self._apply_dbscan(df, eps=eps, min_samples=min_samples, bbox=bbox)
                
                return df

            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                return None

def get_temp_column(df):
    """Determine which temperature column to use based on available data"""
    if 'bright_ti4' in df.columns:
        return 'bright_ti4'
    elif 'brightness' in df.columns:
        return 'brightness'
    else:
        return None

def plot_fire_detections_folium(df, title="Fire Detections", selected_cluster=None, playback_mode=False, playback_date=None, dot_size_multiplier=1.0, color_palette='inferno'):
    """Plot fire detections on a folium map with color palette based on temperature"""
    # Create a working copy of the dataframe
    plot_df = df.copy()
    
    # Apply cluster selection filter if a cluster is selected
    if selected_cluster is not None and selected_cluster in plot_df['cluster'].values:
        # Filter for just the selected cluster
        plot_df = plot_df[plot_df['cluster'] == selected_cluster].copy()
        title = f"{title} - Cluster {selected_cluster}"
    
    # Then apply playback filter if in playback mode
    if playback_mode and playback_date is not None:
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
    initial_tiles = 'cartodbdark_matter'
    if 'basemap' in globals() and basemap in basemap_tiles:
        initial_tiles = basemap_tiles[basemap]
    
    m = folium.Map(location=[center_lat, center_lon], control_scale=True, 
                  tiles=initial_tiles)  # Set the base map from user selection
    
    # Automatically zoom to fit all points
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]], padding=(50, 50))
    
    # Add a title to the map
    title_html = f'''
             <h3 align="center" style="font-size:16px; color: white;"><b>{title}</b></h3>
             '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Create feature groups for different sets of points
    fg_all = folium.FeatureGroup(name="All Points")
    fg_selected = folium.FeatureGroup(name="Selected Cluster")
    
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
                else:
                    color = '#3186cc'  # Default blue
                
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
                radius=base_medium_dot,
                color='white',  # White border for visibility
                weight=0.5,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,  # Increased opacity
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=f"Cluster {point['cluster']} - ({point['latitude']:.4f}, {point['longitude']:.4f})"
            )
            
            circle.add_to(fg_all)
    
    # Add feature groups to map
    fg_all.add_to(m)
    fg_selected.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add colormap to map if temperature data is available
    if temp_col:
        colormap.add_to(m)
    
    # Add basemap layers with proper attribution
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
        • Use the dropdown menu on the right to select clusters<br>
        • Zoom with +/- or mouse wheel<br>
        • Change base maps with layer control (top right)
    </div>
    """
    m.get_root().html.add_child(folium.Element(info_text))
    
    return m

def create_cluster_summary(df):
    """Create summary statistics for each cluster"""
    if df is None or df.empty:
        return None
        
    cluster_summary = (df[df['cluster'] != -1]
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
        temp_stats = df[df['cluster'] != -1].groupby('cluster')[temp_col].agg(['mean', 'max']).round(2)
        cluster_summary['Mean Temperature'] = temp_stats['mean']
        cluster_summary['Max Temperature'] = temp_stats['max']
    
    return cluster_summary.reset_index()

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
                st.subheader(f"Points in Cluster {st.session_state.selected_cluster}")
            
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

def main():
    # Add custom CSS to make the map fill more space
    st.markdown("""
        <style>
        .main > div {max-width: 100% !important;}
        .stApp {background-color: #0e1117;}
        .element-container {width: 100% !important;}
        .css-1d391kg {width: 100% !important;}
        </style>
        """, unsafe_allow_html=True)
    
    # Sidebar for inputs
    st.sidebar.header("Analysis Settings")
    
    # Country selection
    st.sidebar.subheader("Please select your country")
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
        'Morocco': '-13.2,27.7,-1.0,35.9'
    }
    country = st.sidebar.selectbox(
        "",  # Empty label since we have the subheader
        list(country_options.keys())
    )
    
    # Dataset selection
    st.sidebar.subheader("Select Datasets")
    datasets = {}
    datasets['VIIRS_NOAA20_NRT'] = st.sidebar.checkbox("VIIRS NOAA-20", value=True)
    datasets['VIIRS_SNPP_NRT'] = st.sidebar.checkbox("VIIRS SNPP", value=True)
    datasets['MODIS_NRT'] = st.sidebar.checkbox("MODIS", value=True)
    dataset = st.sidebar.selectbox(
        "Select Dataset",
        ['VIIRS_NOAA20_NRT', 'VIIRS_SNPP_NRT', 'MODIS_NRT']
    )
    
    #Category selection
    st.sidebar.subheader("Select Category")
    category = st.sidebar.selectbox(
      "Thermal Detection Type",
      ["wildfires", "flares", "volcanoes"],
      help="""
        Wildfires: Temperature > 300K, FRP > 1.0 (VIIRS) or Confidence > 80% (MODIS)
        Gas Flares: Temperature > 1000K, typically industrial sources
        Volcanic Activity: Temperature > 1300K, clustered near known volcanic regions
        """
    )
        
    selected_datasets = [dataset for dataset, is_selected in datasets.items() if is_selected]
    if not selected_datasets:
      st.sidebar.warning('Please select at least one dataset')
    
    # Category selection
    st.sidebar.subheader("Select Category")
    category = st.sidebar.selectbox(
      "Thermal Detection Type",
      ["fires", "flares", "volcanoes", "raw data"],
      key="category_select",
      help="""
        Fires: Temperature > 300K, FRP > 1.0 (VIIRS) or Confidence > 80% (MODIS)
        Gas Flares: Temperature > 1000K, typically industrial sources
        Volcanic Activity: Temperature > 1300K, clustered near known volcanic regions
        Raw Data: All data points including noise points not assigned to clusters
        """
    )
    
    # Visualization settings
    st.sidebar.subheader("Visualization Settings")
    
    # Color palette selection
    color_palette = st.sidebar.selectbox(
        "Color Palette",
        ['inferno', 'viridis', 'plasma', 'magma', 'cividis'],
        index=0,
        key="color_palette_select",
        help="Select a color palette for hotspot temperature visualization"
    )
    
    # Base map selection
    basemap = st.sidebar.selectbox(
        "Base Map",
        ['Dark', 'Light', 'Satellite', 'Terrain'],
        index=0,
        key="basemap_select",
        help="Select the base map style"
    )
    
    # Map base map selection to Folium tile names
    basemap_tiles = {
        'Dark': 'cartodbdark_matter',
        'Light': 'cartodbpositron',
        'Satellite': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        'Terrain': 'stamenterrain'
    }
    
    # Dot size adjustment
    dot_size_multiplier = st.sidebar.slider(
        "Dot Size",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Adjust the size of hotspot markers on the map"
    )
    
    # Date range selection
    st.sidebar.subheader("Select Date Range")
    default_end_date = datetime.now()
    default_start_date = default_end_date - timedelta(days=7)
    
    start_date = st.sidebar.date_input(
      "Start Date",
      value = default_start_date,
      max_value=default_end_date
    )
      
    end_date = st.sidebar.date_input(
      "End Date",
      value = default_end_date,
      min_value=start_date,
      max_value=default_end_date
    )
    
    # Calculate date range in days
    date_range_days = (end_date - start_date).days
    
    # Define large countries that might be slow with wide date ranges
    large_countries = ['United States', 'China', 'Russia', 'Canada', 'Brazil', 'Australia', 'India']
    
    # Show warning for large countries with wide date ranges
    if country in large_countries and date_range_days > 14:
        st.sidebar.warning(f"⚠️ You selected a {date_range_days}-day period for {country}, which is a large country. This may take a long time to process. Consider reducing your date range to 14 days or less for faster results.")
    
    # API credentials (hidden in expander)
    with st.sidebar.expander("API Settings"):
        username = st.text_input("FIRMS Username", value="tombrown4444")
        password = st.text_input("FIRMS Password", value="wft_wxh6phw9URY-pkv", type="password")
        api_key = st.text_input("FIRMS API Key", value="897a9b7869fd5e4ad231573e14e1c8c8")
    
    # Generate button
    generate_button = st.sidebar.button("Generate Analysis")
    
    # Add logic to check if we should proceed with analysis
    proceed_with_analysis = True
    
    if generate_button and proceed_with_analysis:
        with st.spinner("Analyzing fire data..."):
            handler = FIRMSHandler(username, password, api_key)
            results = handler.fetch_fire_data(
                country=country,
                dataset=dataset,
                category='wildfires',
                use_clustering=True
            )
            # Store results in session state
            st.session_state.results = results
            # Reset selected cluster
            st.session_state.selected_cluster = None
            # Reset playback mode
            st.session_state.playback_mode = False
    
    # Display results in two columns
    if st.session_state.results is not None and not st.session_state.results.empty:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Fire Detection Map")
            
            # Check if we're in playback mode
            if not st.session_state.playback_mode:
                # Create the folium visualization (normal mode)
                folium_map = plot_fire_detections_folium(
                    st.session_state.results, 
                    f"Fire Clusters - {country}", 
                    st.session_state.selected_cluster
                )
                
                if folium_map:
                    # Display the folium map - maintain aspect ratio
                    html_map = folium_map._repr_html_()
                    components.html(html_map, height=600, width=985)
                    
                    # Add a map info section
                    with st.expander("Map Information"):
                        st.write("""
                        - **Points** represent fire detections from satellite data
                        - **Colors** indicate temperature using the inferno color palette (yellow/white = hottest, purple/black = coolest)
                        - **Highlighted points** (black outline) are from the selected cluster
                        - **Popup information** displays when clicking on a point
                        - **Layer control** in the top right allows toggling different layers
                        - **Basemap options** can be changed using the layer control
                        - **Coordinate data** is displayed in the table below when a cluster is selected
                        """)
                    
                    # Display coordinate table for selected cluster
                    display_coordinate_view(st.session_state.results)
                else:
                    st.warning("No data to display on the map.")
            else:
                # We're in playback mode - get current date
                current_date = st.session_state.playback_dates[st.session_state.playback_index]
                
                # Add exit button for playback mode
                if st.button("Exit Play Back"):
                    st.session_state.playback_mode = False
                    st.rerun()
                
                # Create the playback visualization
                playback_title = f"Cluster {st.session_state.selected_cluster} - {current_date}"
                
                # Get only the data for the selected cluster and date
                playback_data = st.session_state.results[
                    (st.session_state.results['cluster'] == st.session_state.selected_cluster) &
                    (st.session_state.results['acq_date'] == current_date)
                ]
                
                folium_map = plot_fire_detections_folium(
                    playback_data,
                    playback_title,
                    st.session_state.selected_cluster,
                    True,
                    current_date
                )
                
                if folium_map:
                    # Save the map to an HTML string and display it using components
                    html_map = folium_map._repr_html_()
                    components.html(html_map, height=600, width=700)
                    
                    # Add the time slider (only if there are enough dates)
                    if len(st.session_state.playback_dates) > 1:
                        st.write("### Timeline")
                        date_index = st.slider(
                            "Select Date", 
                            0, 
                            len(st.session_state.playback_dates) - 1, 
                            st.session_state.playback_index
                        )
                        
                        # Update index if changed
                        if date_index != st.session_state.playback_index:
                            st.session_state.playback_index = date_index
                            st.rerun()
                    else:
                        st.write("Not enough dates for playback.")
                    
                    # Show current timeline info
                    st.write(f"**Date: {current_date}** (Day {st.session_state.playback_index + 1} of {len(st.session_state.playback_dates)})")
                    
                    # Display statistics for this date
                    st.write("### Daily Statistics")
                    st.write(f"**Detection points:** {len(playback_data)}")
                    if not playback_data.empty:
                        st.write(f"**Mean FRP:** {playback_data['frp'].mean():.2f}")
                        
                        # Display temperature if available
                        temp_col = get_temp_column(playback_data)
                        if temp_col:
                            st.write(f"**Mean Temperature:** {playback_data[temp_col].mean():.2f}K")
                            st.write(f"**Max Temperature:** {playback_data[temp_col].max():.2f}K")
                    
                    # Display coordinate view for just this date
                    display_coordinate_view(st.session_state.results, current_date)
                else:
                    st.warning("No data to display for this date.")
        
        with col2:
            st.subheader("Cluster Summary")
            
            # Create the cluster summary table
            cluster_summary = create_cluster_summary(st.session_state.results)
            
            if cluster_summary is not None:
                # Allow user to select a cluster from the table
                st.write("Select a cluster to highlight on the map:")
                cluster_options = [f"Cluster {c}" for c in cluster_summary['cluster'].tolist()]
                selected_from_table = st.selectbox(
                    "Select cluster",
                    ["None"] + cluster_options,
                    key="cluster_select"
                )
                
                if selected_from_table != "None":
                    cluster_id = int(selected_from_table.split(' ')[1])
                    
                    # Check if this is a new selection (different from current)
                    if st.session_state.selected_cluster != cluster_id:
                        st.session_state.selected_cluster = cluster_id
                        
                        # Get unique dates for the selected cluster
                        cluster_points = st.session_state.results[st.session_state.results['cluster'] == st.session_state.selected_cluster]
                        unique_dates = sorted(cluster_points['acq_date'].unique())
                        
                        # Store the dates and initialize to the first one
                        st.session_state.playback_dates = unique_dates
                        st.session_state.playback_index = 0
                        
                        # Reset playback mode when selecting a new cluster
                        st.session_state.playback_mode = False
                        
                        # Force a rerun to update the map with the new selection
                        st.rerun()
                    
                    # Show playback timeline directly (no button needed)
                    if len(st.session_state.playback_dates) > 1:
                        st.write("### Timeline")
                        
                        # Create columns for playback controls
                        play_cols = st.columns([4, 1])
                        
                        with play_cols[0]:
                            # Date slider
                            date_index = st.slider(
                                "Select Date", 
                                0, 
                                len(st.session_state.playback_dates) - 1, 
                                st.session_state.playback_index,
                                key="date_slider_direct",
                                help="Slide to change the date and see how the fire cluster evolved over time"
                            )
                            
                            # Update playback index and mode if slider changed
                            if date_index != st.session_state.playback_index:
                                st.session_state.playback_index = date_index
                                st.session_state.playback_mode = True
                                st.rerun()
                        
                        with play_cols[1]:
                            # Toggle playback mode
                            if st.session_state.playback_mode:
                                if st.button("Exit Timeline", key="exit_timeline"):
                                    st.session_state.playback_mode = False
                                    st.rerun()
                            else:
                                if st.button("Start Timeline", key="start_timeline"):
                                    st.session_state.playback_mode = True
                                    st.rerun()
                    else:
                        st.info("This cluster only appears on one date. Timeline playback is not available.")
                else:
                    # If "None" is selected, clear the selected cluster
                    if st.session_state.selected_cluster is not None:
                        st.session_state.selected_cluster = None
                        # Force a rerun to update the map with all clusters shown
                        st.rerun()
                
                # Highlight the selected cluster in the table if one is selected
                if st.session_state.selected_cluster is not None:
                    highlight_func = lambda x: ['background-color: rgba(255, 220, 40, 0.6); color: black;' 
                                              if x.name == st.session_state.selected_cluster 
                                              else '' for i in x]
                    styled_summary = cluster_summary.style.apply(highlight_func, axis=1)
                    st.dataframe(
                        styled_summary,
                        column_config={
                            "cluster": "Cluster ID",
                            "Number of Points": st.column_config.NumberColumn(help="Fire detections in cluster"),
                            "Mean FRP": st.column_config.NumberColumn(format="%.2f"),
                            "Total FRP": st.column_config.NumberColumn(format="%.2f"),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    # Display the normal table without highlighting
                    st.dataframe(
                        cluster_summary,
                        column_config={
                            "cluster": "Cluster ID",
                            "Number of Points": st.column_config.NumberColumn(help="Fire detections in cluster"),
                            "Mean FRP": st.column_config.NumberColumn(format="%.2f"),
                            "Total FRP": st.column_config.NumberColumn(format="%.2f"),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                
                # Display detailed info for the selected cluster
                if st.session_state.selected_cluster is not None:
                    cluster_data = cluster_summary[cluster_summary['cluster'] == st.session_state.selected_cluster].iloc[0]
                    
                    st.markdown("---")
                    st.write(f"### Cluster {st.session_state.selected_cluster} Details")
                    
                    # Create two columns for details
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        st.write(f"**Detection Points:** {cluster_data['Number of Points']}")
                        st.write(f"**Mean Location:** {cluster_data['Mean Latitude']}, {cluster_data['Mean Longitude']}")
                        st.write(f"**First Detection:** {cluster_data['First Detection']}")
                        st.write(f"**Last Detection:** {cluster_data['Last Detection']}")
                    
                    with detail_col2:
                        st.write(f"**Mean FRP:** {cluster_data['Mean FRP']:.2f}")
                        st.write(f"**Total FRP:** {cluster_data['Total FRP']:.2f}")
                        if 'Mean Temperature' in cluster_data:
                            st.write(f"**Mean Temperature:** {cluster_data['Mean Temperature']:.2f}K")
                            st.write(f"**Max Temperature:** {cluster_data['Max Temperature']:.2f}K")
                    
                    # Add a help tooltip
                    st.info("""
                    **FRP** (Fire Radiative Power) is measured in megawatts (MW) and indicates the intensity of the fire.
                    Higher values suggest more intense burning.
                    """)
                    
                    # Add temperature explanation if available
                    if 'Mean Temperature' in cluster_data:
                        st.info("""
                        **Temperature coloring**: 
                        - Yellow/White indicates the hottest areas (higher temperature)
                        - Orange/Red shows medium temperature
                        - Purple/Black indicates lower temperature
                        """)

if __name__ == "__main__":
    main()