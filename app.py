# app.py

import streamlit as st
import pandas as pd
from quakes import EarthquakeData, DistanceCalculator, TimeCalculator, EarthquakeGrouper, EarthquakeAnalyzer, EarthquakeVisualizer

# Page configuration
st.set_page_config(page_title="Earthquake Analysis", layout="wide")

# Title
st.title("Earthquake Data Analysis Tool")

# Description
st.markdown("""
Welcome to this Earthquake Clustering Tool!

This webapp allows you to analyze seismic data, group related events, and visualize the results interactively.

**Instructions:**
1. Upload your dataset (CSV or Excel file).
2. Assign columns for spatial (latitude, longitude) and temporal (event time) data.
3. Adjust clustering parameters to analyze and visualize clusters.

**Note:** While this tool was originally designed for earthquake data, it can be used for clustering any type of spatiotemporal data. Just map the relevant fields during the column assignment step.

The tool supports datasets with varying column names, so you can map your dataset columns to the required fields. Let's get started!
""")

# File Upload
uploaded_file = st.file_uploader("Upload an Earthquake Data CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    
    # Load and preprocess data
    eq_data = EarthquakeData(uploaded_file)
    try:
        eq_data.load_data()
        df, N = eq_data.get_data()
        
        st.write("### Sample Data")
        st.dataframe(df.head())
        
        # Distance Calculation
        with st.spinner('Calculating distances...'):
            distance_calculator = DistanceCalculator(df)
            distance_calculator.compute_distances()
            D_matrix = distance_calculator.get_distances()
        
        # Time Calculation
        with st.spinner('Calculating time differences...'):
            time_calculator = TimeCalculator(df)
            time_calculator.compute_times()
            T_matrix = time_calculator.get_times()
        
        # Clustering Parameters
        st.write("### Clustering Parameters")
        set_dist = st.slider("Select Distance Threshold (km)", 1, 100, 50)
        set_time_minutes = st.slider("Select Time Threshold (minutes)", 1, 1440, 60)  # Max 24 hours
        set_time = set_time_minutes * 60  # Convert to seconds
        
        # Grouping
        with st.spinner('Grouping earthquakes...'):
            grouper = EarthquakeGrouper(D_matrix, T_matrix, N)
            groupings = grouper.group_with_bfs(set_dist=set_dist, set_time=set_time)
        
        # Analysis
        with st.spinner('Analyzing groups...'):
            analyzer = EarthquakeAnalyzer(df, groupings)
            analyzer.analyze_groups()
            df["Earthquake category"] = ["core" if idx in analyzer.cores else "other" for idx in df.index]
        
        st.write("### Earthquake Groupings")
        st.write(f"Number of clusters: {len(groupings)}")
        st.write(f"Core Earthquakes: {len(analyzer.cores)}")
        
        # Display Cluster Column in Sample Data
        st.write("### Sample Data with Clusters")
        st.dataframe(df.head())
        
        # Visualization
        st.write("### Visualization")
        try:
            visualizer = EarthquakeVisualizer(df)
            visualizer.plot_clusters()
        except ValueError as e:
            st.error(f"Visualization Error: {e}")
        
        # Download Processed Data
        st.write("### Download Processed Data")
        st.download_button(
            label="Download Processed CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='processed_earthquake_data.csv',
            mime='text/csv'
        )
    
    except ValueError as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload a CSV or Excel file to begin analysis.")