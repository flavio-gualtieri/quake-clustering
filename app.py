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
    
    try:
        # Load raw data
        raw_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.write("### Uploaded Data")
        st.dataframe(raw_data.head())

        # Column Mapping
        st.write("### Assign Columns")
        col_lat = st.selectbox("Select the column for Latitude", raw_data.columns)
        col_lon = st.selectbox("Select the column for Longitude", raw_data.columns)
        col_time = st.selectbox("Select the column for Event Time", raw_data.columns)
        col_mag = st.selectbox("Select the column for Magnitude", raw_data.columns)
        col_id = st.selectbox("Select the column for Earthquake ID", raw_data.columns)

        # Confirm Column Mapping
        if st.button("Confirm Column Mapping"):
            st.success("Column mapping confirmed!")

            # Map columns and preprocess data
            column_mapping = {
                col_lat: "Latitude",
                col_lon: "Longitude",
                col_time: "Event Time",
                col_mag: "Magnitude",
                col_id: "ID"
            }

            eq_data = EarthquakeData(uploaded_file, column_mapping)
            eq_data.load_data()
            df, N = eq_data.get_data()

            st.write("### Sample Data (Mapped)")
            st.dataframe(df.head())

            # Clustering Parameters
            st.write("### Clustering Parameters")
            set_dist = st.slider("Select Distance Threshold (km)", 1, 200, 50)
            set_time_minutes = st.slider("Select Time Threshold (minutes)", 1, 1440, 60)  # 1 day max
            set_time = set_time_minutes * 60  # Convert to seconds

            # Distance and Time Calculation
            with st.spinner('Calculating distances...'):
                distance_calculator = DistanceCalculator(df)
                distance_calculator.compute_distances()
                D_matrix = distance_calculator.get_distances()

            with st.spinner('Calculating time differences...'):
                time_calculator = TimeCalculator(df)
                time_calculator.compute_times()
                T_matrix = time_calculator.get_times()

            # Grouping
            with st.spinner('Grouping earthquakes...'):
                grouper = EarthquakeGrouper(D_matrix, T_matrix, N)
                groupings = grouper.group_with_bfs(set_dist=set_dist, set_time=set_time)

            # Analysis
            with st.spinner('Analyzing groups...'):
                analyzer = EarthquakeAnalyzer(df, groupings)
                analyzer.analyze_groups()

                # Assign unique colors to clusters
                cluster_colors = {f"Cluster {i+1}": color for i, color in enumerate(px.colors.qualitative.Set1)}
                df["Cluster Color"] = df["Cluster"].map(cluster_colors).fillna("gray")

                df["Earthquake category"] = [
                    "core" if idx in analyzer.cores else "other" for idx in df.index
                ]

            st.write("### Earthquake Groupings")
            st.write(f"Number of clusters: {len(groupings)}")
            st.write(f"Core Earthquakes: {len(analyzer.cores)}")

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
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV or Excel file to begin analysis.")
