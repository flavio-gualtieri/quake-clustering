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
    
    # Initialize session state for column mappings and processing steps
    if "column_mapping_confirmed" not in st.session_state:
        st.session_state["column_mapping_confirmed"] = False
    if "parameters_confirmed" not in st.session_state:
        st.session_state["parameters_confirmed"] = False
    if "set_dist" not in st.session_state:
        st.session_state["set_dist"] = 50  # Default distance threshold
    if "set_time_minutes" not in st.session_state:
        st.session_state["set_time_minutes"] = 60  # Default time threshold (in minutes)

    # Load the data into a DataFrame
    try:
        raw_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.write("### Uploaded Data")
        st.dataframe(raw_data.head())

        # Column Selection Menu
        if not st.session_state["column_mapping_confirmed"]:
            st.write("### Assign Columns")
            col_lat = st.selectbox("Select the column for Latitude", raw_data.columns)
            col_lon = st.selectbox("Select the column for Longitude", raw_data.columns)
            col_time = st.selectbox("Select the column for Event Time", raw_data.columns)
            col_mag = st.selectbox("Select the column for Magnitude", raw_data.columns)
            col_id = st.selectbox("Select the column for Earthquake ID", raw_data.columns)

            # Confirm mappings
            if st.button("Confirm Column Mapping"):
                st.session_state["column_mapping_confirmed"] = True
                st.session_state["col_lat"] = col_lat
                st.session_state["col_lon"] = col_lon
                st.session_state["col_time"] = col_time
                st.session_state["col_mag"] = col_mag
                st.session_state["col_id"] = col_id
                st.success("Column mapping confirmed!")

        # Proceed after column mapping confirmation
        if st.session_state["column_mapping_confirmed"]:
            data = raw_data.rename(
                columns={
                    st.session_state["col_lat"]: "Latitude",
                    st.session_state["col_lon"]: "Longitude",
                    st.session_state["col_time"]: "Event Time",
                    st.session_state["col_mag"]: "Magnitude",
                    st.session_state["col_id"]: "ID",
                }
            )
            
            # Convert "Event Time" to datetime
            data["Event Time"] = pd.to_datetime(data["Event Time"], errors="coerce")

            # Check for invalid datetime conversion
            if data["Event Time"].isna().any():
                st.error("Some rows have invalid Event Time values. Please check your data.")
            else:
                st.write("### Sample Data (Mapped)")
                st.dataframe(data.head())
                
                # Clustering Parameters
                st.write("### Clustering Parameters")
                set_dist = st.slider(
                    "Select Distance Threshold (km)", 
                    1, 100, st.session_state["set_dist"], key="set_dist"
                )
                set_time_minutes = st.slider(
                    "Select Time Threshold (minutes)", 
                    1, 1440, st.session_state["set_time_minutes"], key="set_time_minutes"
                )
                set_time = set_time_minutes * 60  # Convert to seconds

                # Confirmation Button for Parameters
                if not st.session_state["parameters_confirmed"]:
                    if st.button("Confirm Parameters"):
                        st.session_state["parameters_confirmed"] = True
                        st.success("Parameters confirmed!")

                # Proceed after parameters confirmation
                if st.session_state["parameters_confirmed"]:
                    # Distance Calculation
                    with st.spinner('Calculating distances...'):
                        distance_calculator = DistanceCalculator(data)
                        distance_calculator.compute_distances()
                        D_matrix = distance_calculator.get_distances()

                    # Time Calculation
                    with st.spinner('Calculating time differences...'):
                        time_calculator = TimeCalculator(data)
                        time_calculator.compute_times()
                        T_matrix = time_calculator.get_times()

                    # Grouping
                    with st.spinner('Grouping earthquakes...'):
                        grouper = EarthquakeGrouper(D_matrix, T_matrix, len(data))
                        groupings = grouper.group_with_bfs(
                            set_dist=st.session_state["set_dist"], 
                            set_time=set_time
                        )

                    # Analysis
                    with st.spinner('Analyzing groups...'):
                        analyzer = EarthquakeAnalyzer(data, groupings)
                        analyzer.analyze_groups()
                        data["Earthquake category"] = ["core" if idx in analyzer.cores else "other" for idx in data.index]

                    # Display Clustering Results
                    st.write("### Earthquake Groupings")
                    st.write(f"Number of clusters: {len(groupings)}")
                    st.write(f"Core Earthquakes: {len(analyzer.cores)}")

                    # Display Cluster Column in Sample Data
                    st.write("### Sample Data with Clusters")
                    st.dataframe(data.head())

                    # Visualization
                    st.write("### Visualization")
                    try:
                        visualizer = EarthquakeVisualizer(data)
                        visualizer.plot_clusters()
                    except ValueError as e:
                        st.error(f"Visualization Error: {e}")

                    # Download Processed Data
                    st.write("### Download Processed Data")
                    st.download_button(
                        label="Download Processed CSV",
                        data=data.to_csv(index=False).encode('utf-8'),
                        file_name='processed_earthquake_data.csv',
                        mime='text/csv'
                    )
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload a CSV or Excel file to begin analysis.")