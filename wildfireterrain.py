import numpy as np
import pandas as pd
import random
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import time
import dash_bootstrap_components as dbc

# --------------------------
# Load CSV files with probabilities
fire_feature_analysis_data = pd.read_csv("fire_feature_analysis.csv")
fire_spread_probability_data = pd.read_csv("fire_spread_probability.csv")
fire_terrain_transition_probabilities_data = pd.read_csv("fire_terrain_transition_probabilities.csv")
fire_growth_time_series_data = pd.read_csv("fire_growth_time_series.csv")
fire_spread_direction_probabilities_data = pd.read_csv("fire_spread_direction_probabilities.csv")
# --------------------------

# Define grid size (N x M)
GRID_SIZE = 10

# Terrain mapping (Dataset Terrain -> Real-World Equivalent -> Color)
terrain_data = {
    "C1": ("Spruce-Lichen Woodland", "#556B2F"),
    "C2": ("Boreal Spruce", "#2E8B57"),
    "C3": ("Mature Jack or Lodgepole Pine", "#006400"),
    "C4": ("Immature Jack or Lodgepole Pine", "#228B22"),
    "S1": ("Jack or Lodgepole Pine Slash", "#8B4513"),
    "S2": ("White Spruce-Balsam Slash", "#A0522D"),
    "M1": ("Boreal Mixedwood-Leafless", "#BDB76B"),
    "M2": ("Boreal Mixedwood-Green", "#6B8E23"),
    "D1": ("Leafless Aspen", "#DAA520"),
    "O1a": ("Matted Grass", "#9ACD32"),
    "O1b": ("Standing Grass", "#ADFF2F"),
    "Water": ("Water Bodies", "#4682B4"),
    "Urban": ("Urban/Buildings", "#8B0000"),
    "Barren": ("Desert/Barren Land", "#A9A9A9")
}

# Define environmental variables
WIND_DIRECTIONS = ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]
SLOPE_ANGLES = [0, 5, 10, 15, 20, 25, 30]

# Global grid variables
def initialize_grid():
    global fire_grid, terrain_matrix, fire_status, elevation_matrix
    fire_grid = np.empty((GRID_SIZE, GRID_SIZE), dtype=object)
    terrain_matrix = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    fire_status = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)  # 0: No Fire, 1: On Fire
    elevation_matrix = np.zeros((GRID_SIZE, GRID_SIZE))  # Use slope_angle + noise as elevation
    
    terrain_keys = list(terrain_data.keys())
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            terrain_key = random.choice(terrain_keys)
            real_world_name, color = terrain_data[terrain_key]
            wind_speed = random.randint(0, 45)
            slope_angle = random.choice(SLOPE_ANGLES)
            moisture_level = round(random.uniform(0.1, 0.9), 2)
            
            fire_grid[i, j] = {
                "terrain": terrain_key,
                "real_world": real_world_name,
                "color": color,
                "wind_speed": wind_speed,
                "wind_direction": random.choice(WIND_DIRECTIONS),
                "slope_angle": slope_angle,
                "moisture_level": moisture_level
            }
            terrain_matrix[i, j] = terrain_keys.index(terrain_key)
            # For elevation, use slope_angle plus some noise for realism
            elevation_matrix[i, j] = slope_angle + random.uniform(0, 5)

initialize_grid()

# Create Dash app
# app = dash.Dash(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])  # You can choose another theme like BOOTSTRAP


# Function to generate 2D grid figure (for main view)
def generate_grid_figure():
    fig = px.imshow(terrain_matrix, 
                    color_continuous_scale=[terrain_data[t][1] for t in terrain_data],
                    origin='upper', range_color=[0, len(terrain_data)-1])
    fig.update_layout(title="Wildfire Grid", xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
    
    # Overlay fire icons on burning cells
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if fire_status[i, j] == 1:
                fig.add_trace(go.Scatter(
                    x=[j], y=[i],
                    mode="text",
                    text="🔥",
                    textfont=dict(size=20),
                    showlegend=False
                ))
    return fig

# MDP Fire Spread Simulation function (2D)
def simulate_fire_mdp(steps=10):
    global fire_status
    frames = []
    
    # Define growth rates for key terrain types (default to 1.0 for others)
    terrain_growth_rate = {
        "C6": 2.0,
        "C2": 1.68,
        "C1": 1.48,
        "C3": 1.18,
        "O1a": 0.75,
        "O1b": 0.76,
        "M3": 0.0
    }
    
    # Function to map wind speed to a multiplier
    def get_wind_multiplier(speed):
        if speed <= 9:
            return 0.63
        elif 10 <= speed <= 18:
            return 1.16
        elif 19 <= speed <= 27:
            return 1.64
        else:  # For speed above 27 (up to 45 in our grid)
            return 3.05

    for step in range(steps):
        new_status = fire_status.copy()
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                # Process only burning cells
                if fire_status[i, j] == 1:
                    # Retrieve the burning cell's wind speed and calculate wind multiplier
                    burning_wind_speed = fire_grid[i, j]["wind_speed"]
                    wind_multiplier = get_wind_multiplier(burning_wind_speed)
                    
                    # Retrieve the growth rate for the burning cell's terrain
                    burning_terrain = fire_grid[i, j]["terrain"]
                    gr = terrain_growth_rate.get(burning_terrain, 1.0)
                    
                    # Attempt to ignite the 4-neighbors
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE and fire_status[ni, nj] == 0:
                            # If neighbor terrain is Water, skip ignition
                            if fire_grid[ni, nj]["terrain"] == "Water":
                                continue
                            
                            # Look up transition probability from CSV; default to 0.5 if not found
                            try:
                                p = fire_terrain_transition_probabilities_data.loc[
                                    fire_terrain_transition_probabilities_data["Terrain_Type"] == fire_grid[ni, nj]["terrain"],
                                    "Transition_Probability"
                                ].values[0]
                            except IndexError:
                                p = 0.5
                            
                            # Adjust the probability by growth rate (normalized by max value 2.0) and wind multiplier
                            adjusted_p = p * (gr / 2.0) * wind_multiplier
                            final_probability = min(adjusted_p, 1)  # Ensure probability doesn't exceed 1
                            
                            if random.random() < final_probability:
                                new_status[ni, nj] = 1
        
        # Update fire status and save frame for animation
        fire_status = new_status.copy()
        frames.append(fire_status.copy())
        time.sleep(0.5)  # Optional pause to simulate time progression
    return frames



# Function to generate 2D simulation figure with animation frames
def generate_2d_simulation_figure(frames):
    # Create base figure of the terrain grid
    base_fig = px.imshow(terrain_matrix, 
                         color_continuous_scale=[terrain_data[t][1] for t in terrain_data],
                         origin='upper', range_color=[0, len(terrain_data)-1])
    base_fig.update_layout(title="2D Wildfire Simulation", 
                           xaxis=dict(showticklabels=False), 
                           yaxis=dict(showticklabels=False))
    
    anim_frames = []
    for idx, f_status in enumerate(frames):
        scatter_data = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if f_status[i, j] == 1:
                    scatter_data.append({"x": j, "y": i})
        frame = go.Frame(
            data=[go.Scatter(
                x=[d["x"] for d in scatter_data],
                y=[d["y"] for d in scatter_data],
                mode="markers",
                marker=dict(color="red", size=20),
                showlegend=False
            )],
            name=str(idx)
        )
        anim_frames.append(frame)
    
    # Overlay initial fire markers (if any)
    init_scatter_data = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if frames[0][i, j] == 1:
                init_scatter_data.append({"x": j, "y": i})
    fire_scatter = go.Scatter(
        x=[d["x"] for d in init_scatter_data],
        y=[d["y"] for d in init_scatter_data],
        mode="markers",
        marker=dict(color="red", size=20),
        name="Fire"
    )
    base_fig.add_trace(fire_scatter)
    
    base_fig.frames = anim_frames
    base_fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [{
                "label": "Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}]
            }]
        }]
    )
    return base_fig


app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("🔥 Wildfire Simulation Dashboard", 
                        className="text-center text-warning mb-4 display-5 fw-bold"), width=12)
    ]),

    dbc.Row([
        # Control Panel
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("⚙️ Cell Configuration", className="mb-0 fw-semibold")),
                dbc.CardBody([
                    dbc.Label("🌬️ Wind Speed (km/h)", className="fw-semibold"),
                    dbc.Input(id="wind-speed-input", type="number", placeholder="e.g. 15", min=0, max=45, className="mb-2"),

                    dbc.Label("💧 Moisture Level (0.0 - 1.0)", className="fw-semibold"),
                    dbc.Input(id="moisture-input", type="number", placeholder="e.g. 0.6", step=0.01, min=0, max=1, className="mb-2"),

                    dbc.Label("🧭 Wind Direction", className="fw-semibold"),
                    dcc.Dropdown(WIND_DIRECTIONS, id="wind-direction-dropdown", placeholder="Select Direction", className="mb-2"),

                    dbc.Label("⛰️ Terrain Type", className="fw-semibold"),
                    dcc.Dropdown(list(terrain_data.keys()), id="terrain-dropdown", placeholder="Select Terrain", className="mb-3"),

                    dbc.Row([
                        dbc.Col(dbc.Button("Update Cell", id="update-button", color="primary", className="w-100"), width=4),
                        dbc.Col(dbc.Button("Toggle Fire", id="toggle-fire-button", color="danger", className="w-100"), width=4),
                        dbc.Col(dbc.Button("Regenerate Grid", id="regenerate-grid-button", color="secondary", className="w-100"), width=4),
                    ], className="g-2")
                ])
            ], className="mb-4 shadow-sm rounded-3"),

            dbc.Card([
                dbc.CardHeader(html.H5("📍 Hover Info", className="mb-0 fw-semibold")),
                dbc.CardBody([
                    html.Div(id="hover-data", children="Hover over a cell to see details.", className="text-muted")
                ])
            ], className="mb-4 shadow-sm rounded-3"),

            dbc.Button("▶️ Start Simulation", 
                       id="start-simulation-button", 
                       color="success", 
                       size="lg", 
                       className="w-100 fw-bold shadow-sm")
        ], width=4),

        # Grid and Simulation View
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🗺️ Current Grid", className="fw-semibold")),
                dbc.CardBody([
                    dcc.Graph(id="grid-figure", figure=generate_grid_figure(), config={"displayModeBar": False})
                ])
            ], className="mb-4 shadow-sm rounded-3"),

            html.Div(id="simulation-div", style={"display": "none"}, children=[
                dbc.Card([
                    dbc.CardHeader(html.H5("🔥 Simulation Animation", className="fw-semibold")),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-spinner",
                            type="circle",
                            fullscreen=False,
                            children=dcc.Graph(id="simulation-2d-figure"),
                            color="#28a745"
                        )
                    ])
                ], className="shadow-sm rounded-3")
            ])
        ], width=8)
    ])
], fluid=True, className="pt-4")



@app.callback(
    Output("hover-data", "children"),
    Input("grid-figure", "hoverData")
)
def display_hover_data(hoverData):
    if hoverData:
        x, y = hoverData["points"][0]["x"], hoverData["points"][0]["y"]
        cell = fire_grid[y, x]
        return f"Terrain: {cell['real_world']}, Wind: {cell['wind_speed']} km/h, Moisture: {cell['moisture_level']}"
    return "Hover over a cell to see details."

@app.callback(
    Output("grid-figure", "figure"),
    [Input("update-button", "n_clicks"), Input("toggle-fire-button", "n_clicks"), Input("regenerate-grid-button", "n_clicks")],
    [Input("wind-speed-input", "value"), Input("moisture-input", "value"),
     Input("wind-direction-dropdown", "value"), Input("terrain-dropdown", "value"),
     Input("grid-figure", "clickData")]
)
def update_grid(update_clicks, fire_clicks, regenerate_clicks, wind_speed, moisture_level, wind_direction, terrain_type, clickData):
    ctx = dash.callback_context
    if not ctx.triggered:
        return generate_grid_figure()
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "regenerate-grid-button":
        initialize_grid()
    elif clickData:
        x, y = clickData["points"][0]["x"], clickData["points"][0]["y"]
        if trigger_id == "toggle-fire-button":
            fire_status[y, x] = 1 - fire_status[y, x]  # Toggle fire status
        elif trigger_id == "update-button":
            if terrain_type:
                fire_grid[y, x]["terrain"] = terrain_type
                fire_grid[y, x]["real_world"], fire_grid[y, x]["color"] = terrain_data[terrain_type]
                terrain_matrix[y, x] = list(terrain_data.keys()).index(terrain_type)
            if wind_speed is not None:
                fire_grid[y, x]["wind_speed"] = wind_speed
            if moisture_level is not None:
                fire_grid[y, x]["moisture_level"] = moisture_level
            if wind_direction:
                fire_grid[y, x]["wind_direction"] = wind_direction
    return generate_grid_figure()

# Callback to start simulation (2D animated view)
@app.callback(
    [Output("simulation-div", "style"), Output("simulation-2d-figure", "figure")],
    Input("start-simulation-button", "n_clicks")
)
def trigger_simulation(n_clicks):
    if n_clicks:
        sim_frames = simulate_fire_mdp(steps=10)
        fig2d = generate_2d_simulation_figure(sim_frames)
        return {"display": "block"}, fig2d
    return {"display": "none"}, {}

if __name__ == '__main__':
    app.run(debug=True)

