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
    def __init__(self, filename):
        self.filename = filename
        self.df = None
        self.N = 0

    def load_data(self):
        if self.filename.endswith(".csv"):
            self.df = pd.read_csv(self.filename)
            self.df = self.df[["ID", "Event Time", "Latitude", "Longitude", "Magnitude"]].dropna().drop_duplicates(
                subset=["ID"])
            self.df["Event Time"] = pd.to_datetime(self.df["Event Time"])
        else:
            self.df = pd.read_excel(self.filename)
            self.df = self.df[["id", "Event Time", "Latitude", "Longitude", "Magnitude"]].dropna().drop_duplicates(
                subset=["id"])
            self.df.rename(columns={'id': 'ID'}, inplace=True)
        
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
        for group in self.groupings:
            mini_df = self.df.loc[group].sort_values(by="Event Time")
            core = mini_df["Magnitude"].idxmax()
            self.cores.append(core)

            fores = mini_df.loc[:core - 1]["ID"].tolist() if core > 0 else []
            afters = mini_df.loc[core + 1:]["ID"].tolist() if core < len(mini_df) - 1 else []

            self.fores[mini_df["ID"][core]] = fores
            self.afters[mini_df["ID"][core]] = afters


# ------------------------------
# 6. EarthquakeVisualizer Class
# ------------------------------
class EarthquakeVisualizer:
    def __init__(self, dataframe):
        self.df = dataframe

    def plot_clusters(self):
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
            text=self.df[["Cluster", "Magnitude", "Event Time", "Earthquake category"]],
            mode='markers',
            marker=dict(color=colors)
        ))
        fig.update_geos(projection_type="orthographic")
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
