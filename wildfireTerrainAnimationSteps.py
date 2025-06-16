import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import random
# from theme_config import THEMES, save_theme, load_theme
import os
import sys
from flask import request
from dash_bootstrap_components import themes


#themes
THEMES = {
    "Bootstrap": themes.BOOTSTRAP,
    "Cyborg": themes.CYBORG,
    "Darkly": themes.DARKLY,
    "Flatly": themes.FLATLY,
    "Solar": themes.SOLAR,
    "Sketchy": themes.SKETCHY
}

# Reproducibility
random.seed(42)
np.random.seed(42)

# Constants
GRID_SIZE = 10

# Terrain data
terrain_data = {
    "C6": ("Conifer Dense", "#228B22"),
    "C2": ("Conifer Sparse", "#32CD32"),
    "C1": ("Mixed Forest", "#90EE90"),
    "C3": ("Shrubland", "#ADFF2F"),
    "O1a": ("Grassland A", "#F0E68C"),
    "O1b": ("Grassland B", "#FFFFE0"),
    "M3": ("Moss", "#A9A9A9"),
    "Water": ("Water", "#1E90FF")
}

fire_terrain_transition_probabilities_data = pd.DataFrame({
    "Terrain_Type": ["C6", "C2", "C1", "C3", "O1a", "O1b", "M3"],
    "Transition_Probability": [0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.3]
})

# Grid state
fire_grid = np.empty((GRID_SIZE, GRID_SIZE), dtype=object)
terrain_matrix = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
fire_status = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

# Initialize grid
def initialize_grid():
    global fire_grid, terrain_matrix, fire_status
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            terrain = random.choice(list(terrain_data.keys()))
            fire_grid[i, j] = {
                "terrain": terrain,
                "real_world": terrain_data[terrain][0],
                "color": terrain_data[terrain][1],
                "wind_speed": random.randint(0, 30),
                "moisture_level": random.randint(0, 100),
                "wind_direction": random.choice(["N", "S", "E", "W"])
            }
            terrain_matrix[i, j] = list(terrain_data.keys()).index(terrain)
    fire_status.fill(0)

initialize_grid()



# Dash app
# current_theme_name = load_theme() # if need to save it 
current_theme_name = "Cyborg"
# app = dash.Dash(__name__, external_stylesheets=[THEMES[current_theme_name]])

app = dash.Dash(__name__)

app.layout = dbc.Container([

    # dcc.Location(id="url", refresh=True),
    html.Link(id="theme-link", rel="stylesheet", href=THEMES[current_theme_name]),

    dbc.Row([
        dbc.Col(html.H1("Wildfire Simulation Dashboard",
                        className="text-center text-warning mb-4 display-5 fw-bold"), width=12)
    ]),

    dbc.Row([
        # 🛠️ Control Panel
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🎨 Select Theme", className="mb-0 fw-semibold")),
                dbc.CardBody([
                    dbc.Label("Choose a theme", className="fw-semibold"),
                    dcc.Dropdown(
                        id="theme-selector",
                        options=[{"label": name, "value": name} for name in THEMES.keys()],
                        value=current_theme_name,
                        clearable=False,
                        className="mb-2"
                    )
                ])
            ], className="mb-4 shadow-sm rounded-3"),

            dbc.Card([
                dbc.CardHeader(html.H5("⚙️ Cell Configuration", className="mb-0 fw-semibold")),
                dbc.CardBody([
                    dbc.Label("🌬️ Wind Speed (km/h)", className="fw-semibold"),
                    dbc.Input(id="wind-speed-input", type="number", placeholder="e.g. 15", min=0, max=45, className="mb-2"),

                    dbc.Label("💧 Moisture Level (0.0 - 1.0)", className="fw-semibold"),
                    dbc.Input(id="moisture-input", type="number", placeholder="e.g. 0.6", step=0.01, min=0, max=1, className="mb-2"),

                    dbc.Label("🧭 Wind Direction", className="fw-semibold"),
                    dcc.Dropdown(["N", "S", "E", "W"], id="wind-direction-dropdown", placeholder="Select Direction", className="mb-2"),

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

            dbc.Card([
                dbc.CardHeader(html.H5("▶️ Simulation Setup", className="mb-0 fw-semibold")),
                dbc.CardBody([
                    dbc.Label("⏱️ Number of Simulation Steps", className="fw-semibold"),
                    dbc.Input(id="simulation-steps", type="number", placeholder="e.g. 10", min=1, value=10, className="mb-3"),
                    dbc.Button("Start Simulation", id="start-simulation-button", color="success", size="lg", className="w-100 fw-bold shadow-sm")
                ])
            ], className="mb-4 shadow-sm rounded-3"),

            dbc.Card([
                dbc.CardHeader(html.H5("👥 Contributors", className="mb-0 fw-semibold")),
                dbc.CardBody([
                    html.Ul([
                        html.Li("Kaushal Sambanna"),
                        html.Li("Devansh Makam")
                    ], className="mb-0 fw-normal")
                ])
            ], className="mb-4 shadow-sm rounded-3")

            
        ], width=4),

        # 🗺️ Grid and 🔥 Simulation Panel
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🗺️ Current Grid", className="fw-semibold")),
                dbc.CardBody([
                    dcc.Graph(id="grid-figure", config={"displayModeBar": False})
                ])
            ], className="mb-4 shadow-sm rounded-3"),

            html.Div(id="simulation-div", style={"display": "none"}, children=[
                dbc.Card([
                    dbc.CardHeader(html.H5("Fire Simulation Animation", className="fw-semibold")),
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



# Callbacks
@app.callback(
    Output("theme-link", "href"),
    Input("theme-selector", "value"),
    prevent_initial_call=True
)
def change_theme(theme_name):
    # save_theme(theme_name)
    return THEMES[theme_name]


@app.callback(
    Output("hover-data", "children"),
    Input("grid-figure", "hoverData")
)
def display_hover_data(hoverData):
    if hoverData and "points" in hoverData:
        try:
            x, y = hoverData["points"][0]["x"], hoverData["points"][0]["y"]
            cell = fire_grid[y, x]
            return f"Terrain: {cell['real_world']}, Wind: {cell['wind_speed']} km/h, Moisture: {cell['moisture_level']}"
        except Exception:
            pass
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
    elif clickData and "points" in clickData:
        x, y = clickData["points"][0]["x"], clickData["points"][0]["y"]
        if trigger_id == "toggle-fire-button":
            fire_status[y, x] = 1 - fire_status[y, x]
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

@app.callback(
    [Output("wind-speed-input", "value"),
     Output("moisture-input", "value"),
     Output("wind-direction-dropdown", "value"),
     Output("terrain-dropdown", "value")],
    Input("grid-figure", "clickData")
)
def update_config_on_cell_click(clickData):
    if clickData and "points" in clickData:
        x, y = clickData["points"][0]["x"], clickData["points"][0]["y"]
        cell = fire_grid[y, x]
        return (
            cell.get("wind_speed", 0),
            cell.get("moisture_level", 0.0),
            cell.get("wind_direction", None),
            cell.get("terrain", None)
        )
    # Default values
    return 0, 0.0, None, None


@app.callback(
    [Output("simulation-div", "style"), Output("simulation-2d-figure", "figure")],
    [Input("start-simulation-button", "n_clicks")],
    [State("simulation-steps", "value")]
)
def trigger_simulation(n_clicks, steps):
    if n_clicks:
        sim_frames = simulate_fire_mdp(steps=steps or 10)
        fig2d = generate_2d_simulation_figure(sim_frames)
        return {"display": "block"}, fig2d
    return {"display": "none"}, {}

# Helpers
def generate_grid_figure():
    fig = px.imshow(terrain_matrix,
                    color_continuous_scale=[terrain_data[t][1] for t in terrain_data],
                    origin='upper', range_color=[0, len(terrain_data)-1])
    fig.update_layout(title="Wildfire Grid", xaxis=dict(showticklabels=False),
                      yaxis=dict(showticklabels=False, autorange='reversed'))
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

def simulate_fire_mdp(steps=10):
    global fire_status
    frames = []

    terrain_growth_rate = {
        "C6": 2.0, "C2": 1.68, "C1": 1.48, "C3": 1.18,
        "O1a": 0.75, "O1b": 0.76, "M3": 0.0
    }

    def get_wind_multiplier(speed):
        if speed <= 9: return 0.63
        elif speed <= 18: return 1.16
        elif speed <= 27: return 1.64
        else: return 3.05

    for step in range(steps):
        new_status = fire_status.copy()
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if fire_status[i, j] == 1:
                    wind_speed = fire_grid[i, j]["wind_speed"]
                    wind_multiplier = get_wind_multiplier(wind_speed)
                    terrain = fire_grid[i, j]["terrain"]
                    gr = terrain_growth_rate.get(terrain, 1.0)

                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE and fire_status[ni, nj] == 0:
                            if fire_grid[ni, nj]["terrain"] == "Water":
                                continue
                            try:
                                p = fire_terrain_transition_probabilities_data.loc[
                                    fire_terrain_transition_probabilities_data["Terrain_Type"] == fire_grid[ni, nj]["terrain"],
                                    "Transition_Probability"
                                ].values[0]
                            except IndexError:
                                p = 0.5
                            adjusted_p = p * (gr / 2.0) * wind_multiplier
                            if random.random() < min(adjusted_p, 1):
                                new_status[ni, nj] = 1
        fire_status = new_status
        frames.append(new_status.copy())
    return frames

def generate_2d_simulation_figure(frames):
    base_fig = px.imshow(terrain_matrix,
                         color_continuous_scale=[terrain_data[t][1] for t in terrain_data],
                         origin='upper', range_color=[0, len(terrain_data)-1])
    base_fig.update_layout(title="2D Wildfire Simulation",
                           xaxis=dict(showticklabels=False),
                           yaxis=dict(showticklabels=False, autorange='reversed'))

    anim_frames = []
    for idx, f_status in enumerate(frames):
        scatter_data = [{"x": j, "y": i} for i in range(GRID_SIZE) for j in range(GRID_SIZE) if f_status[i, j] == 1]
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

    init_data = [{"x": j, "y": i} for i in range(GRID_SIZE) for j in range(GRID_SIZE) if frames[0][i, j] == 1]
    fire_scatter = go.Scatter(
        x=[d["x"] for d in init_data],
        y=[d["y"] for d in init_data],
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
                "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
            }]
        }]
    )
    return base_fig

# Run
if __name__ == '__main__':
    app.run(debug=True)
