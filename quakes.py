import numpy as np
import pandas as pd
from datetime import datetime
from geopy.distance import geodesic as gd
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


# ------------------------------
# 1. EarthquakeData Class
# ------------------------------

class EarthquakeData:
    def __init__(self, file):
        self.file = file  # Accept UploadedFile object
        self.df = None
        self.N = 0

    def load_data(self):
        if self.file.name.endswith(".csv"):
            self.df = pd.read_csv(self.file)
            self.df = self.df[["ID", "Event Time", "Latitude", "Longitude", "Magnitude"]].dropna().drop_duplicates(
                subset=["ID"])
            self.df["Event Time"] = pd.to_datetime(self.df["Event Time"])
        elif self.file.name.endswith(".xlsx"):
            self.df = pd.read_excel(self.file)
            self.df = self.df[["id", "Event Time", "Latitude", "Longitude", "Magnitude"]].dropna().drop_duplicates(
                subset=["id"])
            self.df.rename(columns={'id': 'ID'}, inplace=True)
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
        
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.df["Year"] = self.df["Event Time"].dt.year
        self.N = len(self.df)

    def get_data(self):
        return self.df, self.N



# ------------------------------
# 2. DistanceCalculator Class
# ------------------------------
class DistanceCalculator:
    def __init__(self, dataframe):
        self.df = dataframe
        self.D_matrix = None

    def compute_distances(self):
        N = len(self.df)
        self.D_matrix = np.zeros((N, N))
        latitudes = self.df["Latitude"]
        longitudes = self.df["Longitude"]
        
        for i in tqdm(range(N)):
            for j in range(i, N):
                delta = gd((latitudes[i], longitudes[i]), (latitudes[j], longitudes[j])).km
                self.D_matrix[i, j] = delta
                self.D_matrix[j, i] = delta

    def get_distances(self):
        return self.D_matrix


# ------------------------------
# 3. TimeCalculator Class
# ------------------------------
class TimeCalculator:
    def __init__(self, dataframe):
        self.df = dataframe
        self.T_matrix = None

    def compute_times(self):
        N = len(self.df)
        self.T_matrix = np.zeros((N, N))
        timestamps = self.df["Event Time"].apply(lambda x: x.timestamp()).tolist()
        
        for i in tqdm(range(N)):
            for j in range(i, N):
                delta = abs(timestamps[i] - timestamps[j])
                self.T_matrix[i, j] = delta
                self.T_matrix[j, i] = delta

    def get_times(self):
        return self.T_matrix


# ------------------------------
# 4. EarthquakeGrouper Class
# ------------------------------
class EarthquakeGrouper:
    def __init__(self, distance_matrix, time_matrix, N):
        self.D = distance_matrix
        self.T = time_matrix
        self.N = N

    def group_with_bfs(self, set_dist, set_time):
        edges = []
        origins = []
        for i in range(self.N):
            for j in range(i, self.N):
                if i != j and self.D[i, j] <= set_dist and self.T[i, j] <= set_time:
                    origins.append(i)
                    edges.append((i, j, self.D[i, j]))
        
        # BFS-based clustering
        O_copy = copy.deepcopy(list(set(origins)))
        groupings = []
        while O_copy:
            grouping = []
            frontier = [O_copy.pop(0)]
            while frontier:
                current = frontier.pop(0)
                grouping.append(current)
                for edge in edges:
                    if edge[0] == current and edge[1] not in grouping:
                        frontier.append(edge[1])
                        grouping.append(edge[1])
            groupings.append(grouping)
        
        return groupings


# ------------------------------
# 5. EarthquakeAnalyzer Class
# ------------------------------
class EarthquakeAnalyzer:
    def __init__(self, dataframe, groupings):
        self.df = dataframe
        self.groupings = groupings
        self.cores = []
        self.fores = {}
        self.afters = {}

    def analyze_groups(self):
        cluster_col = [""] * len(self.df)  # Create an empty cluster column

        for cluster_idx, group in enumerate(self.groupings):
            mini_df = self.df.loc[group].sort_values(by="Event Time").reset_index(drop=True)
            
            # Get the core earthquake
            core_idx = mini_df["Magnitude"].idxmax()
            core_id = mini_df.at[core_idx, "ID"]

            self.cores.append(core_idx)

            # Identify foreshocks and aftershocks
            fores = mini_df.iloc[:core_idx]["ID"].tolist() if core_idx > 0 else []
            afters = mini_df.iloc[core_idx + 1:]["ID"].tolist() if core_idx < len(mini_df) - 1 else []

            self.fores[str(core_id)] = fores
            self.afters[str(core_id)] = afters

            # Assign cluster label to all earthquakes in the current group
            cluster_label = f"Cluster {cluster_idx + 1}"
            for idx in group:
                cluster_col[idx] = cluster_label

        self.df["Cluster"] = cluster_col  # Add Cluster column to the DataFrame


# ------------------------------
# 6. EarthquakeVisualizer Class
# ------------------------------
class EarthquakeVisualizer:
    def __init__(self, dataframe):
        self.df = dataframe

    def plot_clusters(self):
        # Ensure Cluster column exists
        if "Cluster" not in self.df.columns:
            raise ValueError("The 'Cluster' column is missing from the DataFrame. Ensure clustering was performed correctly.")

        # Define colors for categories
        colors = []
        for cat in self.df["Earthquake category"]:
            if "after" in cat:
                colors.append(px.colors.sequential.Inferno[0])
            elif "fore" in cat:
                colors.append(px.colors.sequential.Inferno[3])
            elif "core" in cat:
                colors.append(px.colors.sequential.Inferno[5])
            else:
                colors.append(px.colors.sequential.Inferno[7])

        fig = go.Figure(data=go.Scattergeo(
            lon=self.df["Longitude"],
            lat=self.df["Latitude"],
            text=self.df[["Cluster", "Magnitude", "Event Time", "Earthquake category"]].astype(str).agg('<br>'.join, axis=1),
            mode='markers',
            marker=dict(color=colors, size=8),
        ))
        fig.update_geos(projection_type="orthographic")
        fig.update_layout(
            title="Earthquake Cluster Visualization",
            geo=dict(
                showland=True,
                landcolor="rgb(243, 243, 243)",
                subunitcolor="rgb(255, 255, 255)"
            )
        )
        fig.show()


# ------------------------------
# Main Program
# ------------------------------
if __name__ == "__main__":
    eq_data = EarthquakeData("earthquakes.csv")
    eq_data.load_data()
    df, N = eq_data.get_data()

    distance_calculator = DistanceCalculator(df)
    distance_calculator.compute_distances()
    D_matrix = distance_calculator.get_distances()

    time_calculator = TimeCalculator(df)
    time_calculator.compute_times()
    T_matrix = time_calculator.get_times()

    grouper = EarthquakeGrouper(D_matrix, T_matrix, N)
    groupings = grouper.group_with_bfs(set_dist=50, set_time=3600)

    analyzer = EarthquakeAnalyzer(df, groupings)
    analyzer.analyze_groups()

    df["Earthquake category"] = ["core" if idx in analyzer.cores else "other" for idx in df.index]

    visualizer = EarthquakeVisualizer(df)
    visualizer.plot_clusters()
