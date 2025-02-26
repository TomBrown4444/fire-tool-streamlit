import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import requests
from datetime import datetime, timedelta
from io import StringIO
import matplotlib.colors as mcolors

# Set page config
st.set_page_config(
    page_title="Fire Analysis Tool",
    layout="wide"
)

# Title and description
st.title("Fire Analysis Tool")
st.markdown("---")

# Initialize session state for results and selected cluster
if 'results' not in st.session_state:
    st.session_state.results = None
if 'selected_cluster' not in st.session_state:
    st.session_state.selected_cluster = None

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
            'China': '73.0,18.0,135.0,53.0'
        }
        return bboxes.get(country, None)

    def _apply_dbscan(self, df, eps=0.01, min_samples=5):
        """Apply DBSCAN clustering"""
        if len(df) < min_samples:
            st.warning(f"Too few points ({len(df)}) for clustering. Minimum required: {min_samples}")
            return df
        
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
        category='wildfires',
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
                    df = self._apply_dbscan(df, eps=eps, min_samples=min_samples)
                
                return df

            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                return None

def plot_fire_detections(df, title="Fire Detections", selected_cluster=None):
    """Plot fire detections using matplotlib with highlighted clusters"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Separate the data based on the selected cluster
    if selected_cluster is not None and selected_cluster in df['cluster'].values:
        highlighted_points = df[df['cluster'] == selected_cluster]
        other_points = df[df['cluster'] != selected_cluster]
        
        # Plot other points with lower alpha
        scatter1 = ax.scatter(
            other_points['longitude'], 
            other_points['latitude'],
            c=other_points['cluster'],
            cmap='viridis',
            s=50,
            alpha=0.3
        )
        
        # Plot highlighted points with higher alpha and larger size
        scatter2 = ax.scatter(
            highlighted_points['longitude'], 
            highlighted_points['latitude'],
            c=highlighted_points['cluster'],
            cmap='viridis',
            s=80,
            alpha=0.9,
            edgecolors='yellow',
            linewidths=1
        )
    else:
        # Plot all points normally
        scatter1 = ax.scatter(
            df['longitude'], 
            df['latitude'],
            c=df['cluster'],
            cmap='viridis',
            s=50,
            alpha=0.6
        )
    
    # Add a colorbar
    cbar = plt.colorbar(scatter1, label='Cluster ID')
    
    # Set plot title and labels
    ax.set_title(title, pad=20, fontsize=14)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add coordinate annotations
    ax.text(0.01, 0.01, 'Click on a cluster in the table to highlight it on the map',
            transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    # For selected clusters, add an information box
    if selected_cluster is not None and selected_cluster in df['cluster'].values:
        cluster_data = df[df['cluster'] == selected_cluster]
        info_text = (
            f"Cluster: {selected_cluster}\n"
            f"Points: {len(cluster_data)}\n"
            f"Mean Lat: {cluster_data['latitude'].mean():.4f}\n"
            f"Mean Long: {cluster_data['longitude'].mean():.4f}\n"
            f"Mean FRP: {cluster_data['frp'].mean():.2f}"
        )
        ax.text(0.99, 0.99, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.7))
    
    return fig

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
    if 'bright_ti4' in df.columns:
        temp_stats = df[df['cluster'] != -1].groupby('cluster')['bright_ti4'].agg(['mean', 'max']).round(2)
        cluster_summary['Mean Temperature'] = temp_stats['mean']
        cluster_summary['Max Temperature'] = temp_stats['max']
    elif 'brightness' in df.columns:
        temp_stats = df[df['cluster'] != -1].groupby('cluster')['brightness'].agg(['mean', 'max']).round(2)
        cluster_summary['Mean Temperature'] = temp_stats['mean']
        cluster_summary['Max Temperature'] = temp_stats['max']
    
    return cluster_summary.reset_index()

def display_coordinate_view(df):
    """Display a table with coordinates and details for the selected cluster"""
    if df is None or df.empty:
        return
    
    if st.session_state.selected_cluster is not None:
        cluster_points = df[df['cluster'] == st.session_state.selected_cluster]
        
        if not cluster_points.empty:
            st.subheader(f"Points in Cluster {st.session_state.selected_cluster}")
            
            # Create a display version of the dataframe with formatted columns
            display_df = cluster_points[['latitude', 'longitude', 'frp', 'acq_date', 'acq_time']].copy()
            
            # Add a formatted coordinate column
            display_df['Coordinates'] = display_df.apply(
                lambda row: f"{row['latitude']:.4f}, {row['longitude']:.4f}", 
                axis=1
            )
            
            # Display the dataframe
            st.dataframe(
                display_df[['Coordinates', 'frp', 'acq_date', 'acq_time']],
                column_config={
                    "Coordinates": "Lat, Long",
                    "frp": st.column_config.NumberColumn("FRP", format="%.2f"),
                    "acq_date": "Date",
                    "acq_time": "Time"
                },
                hide_index=True,
                use_container_width=True
            )

def main():
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
    
    # API credentials (hidden in expander)
    with st.sidebar.expander("API Settings"):
        username = st.text_input("FIRMS Username", value="tombrown4444")
        password = st.text_input("FIRMS Password", value="wft_wxh6phw9URY-pkv", type="password")
        api_key = st.text_input("FIRMS API Key", value="897a9b7869fd5e4ad231573e14e1c8c8")
    
    # Generate button
    if st.sidebar.button("Generate Analysis"):
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
    
    # Display results in two columns
    if st.session_state.results is not None and not st.session_state.results.empty:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Fire Detection Map")
            
            # Create the matplotlib visualization
            fig = plot_fire_detections(
                st.session_state.results, 
                f"Fire Clusters - {country}", 
                st.session_state.selected_cluster
            )
            
            # Display the matplotlib figure
            st.pyplot(fig)
            
            # Add a map info section
            with st.expander("Map Information"):
                st.write("""
                - **Points** represent fire detections from satellite data
                - **Colors** indicate different fire clusters identified by DBSCAN
                - **Highlighted points** (yellow border) are from the selected cluster
                - **Coordinate data** is displayed in the table below when a cluster is selected
                """)
            
            # Display coordinate table for selected cluster
            display_coordinate_view(st.session_state.results)
            
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
                    ["None"] + cluster_options
                )
                
                if selected_from_table != "None":
                    cluster_id = int(selected_from_table.split(' ')[1])
                    st.session_state.selected_cluster = cluster_id
                else:
                    st.session_state.selected_cluster = None
                
                # Highlight the selected cluster in the table if one is selected
                if st.session_state.selected_cluster is not None:
                    highlight_func = lambda x: ['background-color: #ffff99' 
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

if __name__ == "__main__":
    main()