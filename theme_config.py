import json
from dash_bootstrap_components import themes

THEMES = {
    "Bootstrap": themes.BOOTSTRAP,
    "Cyborg": themes.CYBORG,
    "Darkly": themes.DARKLY,
    "Flatly": themes.FLATLY,
    "Solar": themes.SOLAR,
    "Sketchy": themes.SKETCHY
}

def save_theme(theme_name):
    with open("theme.json", "w") as f:
        json.dump({"theme": theme_name}, f)

def load_theme():
    try:
        with open("theme.json", "r") as f:
            return json.load(f).get("theme", "Cyborg")
    except:
        return "Cyborg"
