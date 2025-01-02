import streamlit as st
import pandas as pd
from quakes import EarthquakeData, DistanceCalculator, TimeCalculator, EarthquakeGrouper, EarthquakeAnalyzer, EarthquakeVisualizer

# Page configuration
st.set_page_config(page_title="Earthquake Analysis", layout="wide")

# Title
st.title("🌍 Earthquake Data Analysis Tool")

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
        
        # Grouping
        set_dist = st.slider("Select Distance Threshold (km)", 1, 100, 50)
        set_time = st.slider("Select Time Threshold (seconds)", 3600, 86400, 3600)
        
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
        
        # Visualization
        st.write("### Visualization")
        visualizer = EarthquakeVisualizer(df)
        visualizer.plot_clusters()
        
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
