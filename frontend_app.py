import streamlit as st
import requests
import pandas as pd
import pydeck as pdk
import datetime
from streamlit_folium import st_folium
import folium
import altair as alt
import time
import geopandas as gpd
import numpy as np
from geopy.distance import geodesic
from sklearn.neighbors import BallTree
import pycountry
import os
import glob

# Helper: ISO3 to country name

def get_country_name(iso3):
    try:
        return pycountry.countries.get(alpha_3=iso3).name
    except Exception:
        return iso3

# --- Load GIS datasets into gis_df ---
gis_files = [
    ("Crude Oil Refineries", "data/GIS/Crude_oil_refinaries.csv"),
    ("Gathering and Processing", "data/GIS/Gathering_and_Processing.csv"),
    ("Nat Gas Flaring Detections", "data/GIS/Nat_gas_flaring_detections.csv"),
    ("Petroleum Terminals", "data/GIS/Petroleum_terminals.csv"),
    ("OGIM", "data/GIS/OGIM.csv"),
    ("Global Oil and Gas Extraction Tracker", "data/GIS/Global-Oil-and-Gas-Extraction-Tracker-Feb-2025 - Main data.csv"),
]
gis_dfs = []
for label, path in gis_files:
    try:
        df = pd.read_csv(path)
        df["_dataset"] = label
        gis_dfs.append(df)
    except Exception as e:
        pass  # Optionally log or print error
if gis_dfs:
    gis_df = pd.concat(gis_dfs, ignore_index=True)
else:
    gis_df = pd.DataFrame()

BACKEND_URL = "http://localhost:8000"

# --- Load both compiled datasets for trade analysis (needed for selectors) ---
gas_file = 'data/GIS/compiled_gas_data.csv'
oil_file = 'data/GIS/compiled_oil_data.csv'
gas_df, oil_df = None, None
try:
    gas_df = pd.read_csv(gas_file, encoding='utf-8')
    gas_df.columns = gas_df.columns.str.strip()
    gas_df = gas_df.loc[:, ~gas_df.columns.str.contains('^Unnamed')]
except Exception as e:
    gas_df = None
try:
    oil_df = pd.read_csv(oil_file, encoding='utf-8')
    oil_df.columns = oil_df.columns.str.strip()
    oil_df = oil_df.loc[:, ~oil_df.columns.str.contains('^Unnamed')]
except Exception as e:
    oil_df = None

# Build unified selectors from both datasets
all_countries = set()
all_flows = set()
if gas_df is not None:
    all_countries.update(gas_df['reporterISO'].dropna().unique())
    all_flows.update(gas_df['flowCode'].dropna().unique())
if oil_df is not None:
    all_countries.update(oil_df['reporterISO'].dropna().unique())
    all_flows.update(oil_df['flowCode'].dropna().unique())
country_options = sorted(all_countries)
flow_options = sorted(all_flows)
commodity_options = ['Gas', 'Oil', 'Both']

# Sidebar selectors (must be defined before any use of selected_country)
selected_country = st.sidebar.selectbox("Select Country (ISO)", country_options, key='unified_country')
selected_flow = st.sidebar.selectbox("Select Import/Export", flow_options, key='unified_flow')
selected_commodity = st.sidebar.selectbox("Select Commodity", commodity_options, key='unified_commodity')

st.set_page_config(page_title="Methane Monitoring Demo", layout="wide")
st.title("Methane Emissions Monitoring")

# Add company logo just below the title
st.image('data/company_logo.jpeg', width=360)

st.markdown(
    '''
    <style>
    .stApp {
        background-color: #f7f9fb;
    }
    .card {
        background: #fff;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(44, 62, 80, 0.07);
        padding: 2rem 2rem 1.5rem 2rem;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #22223b;
        margin-bottom: 1rem;
        border-left: 6px solid #fde725;
        padding-left: 0.75rem;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #22223b;
        margin-bottom: 1rem;
    }
    </style>
    ''',
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    :root {
        --primary-color: #fde725 !important;
    }
    /* Sidebar slider thumb and track */
    [data-testid="stSidebar"] .stSlider [role="slider"] {
        background: #fde725 !important;
        border: 2px solid #fde725 !important;
    }
    [data-testid="stSidebar"] .stSlider .css-1c5b3bq {
        background: #fde725 !important;
    }
    /* Sidebar checkboxes */
    [data-testid="stSidebar"] .stCheckbox > label > div:first-child {
        border: 2px solid #fde725 !important;
    }
    [data-testid="stSidebar"] .stCheckbox input:checked + div {
        background: #fde725 !important;
        border-color: #fde725 !important;
    }
    /* Sidebar selectbox (label and border) */
    [data-testid="stSidebar"] .stSelectbox > div {
        border: 2px solid #fde725 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label {
        color: #fde725 !important;
    }
    [data-testid="stSidebar"] select {
        background-color: #fde72522 !important;
        border: 2px solid #fde725 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Toggle datasets
with st.sidebar:
    st.markdown('<div class="sidebar-header">Data Layers</div>', unsafe_allow_html=True)
    show_emit = st.checkbox("EMIT", value=True)
    show_emit_polygons = st.checkbox("EMIT Polygons", value=False)
    show_cams = st.checkbox("CAMS", value=True)
    show_crude = st.checkbox("Crude Oil Refineries", value=False)
    show_gathering = st.checkbox("Gathering and Processing", value=False)
    show_flaring = st.checkbox("Nat Gas Flaring Detections", value=False)
    show_terminals = st.checkbox("Petroleum Terminals", value=False)
    show_ogim = st.checkbox("OGIM", value=False)
    show_global_tracker = st.checkbox("Global Oil and Gas Extraction Tracker", value=False)

# GIS file paths
GIS_DIR = "data/GIS"
GIS_FILES = {
    "crude": (show_crude, f"{GIS_DIR}/Crude_oil_refinaries.csv", "Crude Oil Refineries"),
    "gathering": (show_gathering, f"{GIS_DIR}/Gathering_and_Processing.csv", "Gathering and Processing"),
    "natgas": (show_flaring, f"{GIS_DIR}/Nat_gas_flaring_detections.csv", "Nat Gas Flaring Detections"),
    "terminals": (show_terminals, f"{GIS_DIR}/Petroleum_terminals.csv", "Petroleum Terminals"),
    "ogim": (show_ogim, f"{GIS_DIR}/OGIM.csv", "OGIM"),
    "global": (show_global_tracker, f"{GIS_DIR}/Global-Oil-and-Gas-Extraction-Tracker-Feb-2025 - Main data.csv", "Global Oil and Gas Extraction Tracker"),
}

@st.cache_data(show_spinner=False)
def fetch_data(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

layers = []
df_emit = pd.DataFrame()
df_cams = pd.DataFrame()

# Time slider for EMIT and CAMS
# Fetch min/max dates from both datasets
emit_data = fetch_data(f"{BACKEND_URL}/emit")
cams_data = fetch_data(f"{BACKEND_URL}/cams")

def get_date_range(df):
    if "date" in df.columns:
        try:
            dates = pd.to_datetime(df["date"], errors="coerce")
            return dates.min(), dates.max()
        except Exception:
            return None, None
    return None, None

emit_df = pd.DataFrame(emit_data)
cams_df = pd.DataFrame(cams_data)
emit_min, emit_max = get_date_range(emit_df)
cams_min, cams_max = get_date_range(cams_df)

# Compute global min/max
all_min = min([d for d in [emit_min, cams_min] if d is not pd.NaT and d is not None], default=None)
all_max = max([d for d in [emit_max, cams_max] if d is not pd.NaT and d is not None], default=None)

if all_min is not None and all_max is not None:
    date_range = st.sidebar.slider(
        "Select date range",
        min_value=all_min.date(),
        max_value=all_max.date(),
        value=(all_min.date(), all_max.date()),
        format="YYYY-MM-DD"
    )
else:
    date_range = None

# EMIT
if show_emit:
    df_emit = emit_df.copy()
    if date_range is not None:
        df_emit = df_emit[pd.to_datetime(df_emit["date"], errors="coerce").between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))]
    df_emit_points = df_emit.dropna(subset=["latitude", "longitude"])
    if len(df_emit_points) > 1000:
        df_emit_points = df_emit_points.sample(1000, random_state=42)
    # EMIT points (dark purple)
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=df_emit_points,
        get_position="[longitude, latitude]",
        get_radius=4000,
        get_fill_color="[68, 1, 84, 160]",  # #440154
        pickable=True,
        auto_highlight=True,
        id="emit-points"
    ))

# CAMS
if show_cams:
    df_cams = cams_df.copy()
    if date_range is not None:
        df_cams = df_cams[pd.to_datetime(df_cams["date"], errors="coerce").between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))]
    # st.write("CAMS columns:", df_cams.columns.tolist())  # Debug: show columns in Streamlit
    if not df_cams.empty and "latitude" in df_cams.columns and "longitude" in df_cams.columns:
        df_cams_points = df_cams.dropna(subset=["latitude", "longitude"])
        if len(df_cams_points) > 1000:
            df_cams_points = df_cams_points.sample(1000, random_state=42)
        # CAMS points (viridis green)
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=df_cams_points,
            get_position="[longitude, latitude]",
            get_radius=4000,
            get_fill_color="[41, 175, 127, 160]",  # #29af7f
            pickable=True,
            auto_highlight=True,
            id="cams-points"
        ))

# Add GIS layers if toggled
for key, (show, path, label) in GIS_FILES.items():
    if show:
        try:
            df_gis = pd.read_csv(path)
            # Try to find latitude/longitude columns
            lat_col = next((c for c in df_gis.columns if c.lower() in ["lat", "latitude"]), None)
            lon_col = next((c for c in df_gis.columns if c.lower() in ["lon", "longitude"]), None)
            if lat_col and lon_col:
                df_gis_points = df_gis.dropna(subset=[lat_col, lon_col])
                if len(df_gis_points) > 10000:
                    df_gis_points = df_gis_points.sample(10000, random_state=42)
                layers.append(pdk.Layer(
                    "ScatterplotLayer",
                    data=df_gis_points,
                    get_position=f"[{lon_col}, {lat_col}]",
                    get_radius=4000,
                    get_fill_color="[0, 0, 0, 120]",
                    pickable=True,
                    auto_highlight=True,
                    id=f"gis-{key}-points"
                ))
        except Exception as e:
            st.warning(f"Could not load {label}: {e}")

# --- MapStand Organisation Company WFS Integration ---
import requests

# Remove Organisation Company toggle from sidebar

# GIS file paths
GIS_DIR = "data/GIS"
GIS_FILES = {
    "crude": (show_crude, f"{GIS_DIR}/Crude_oil_refinaries.csv", "Crude Oil Refineries"),
    "gathering": (show_gathering, f"{GIS_DIR}/Gathering_and_Processing.csv", "Gathering and Processing"),
    "natgas": (show_flaring, f"{GIS_DIR}/Nat_gas_flaring_detections.csv", "Nat Gas Flaring Detections"),
    "terminals": (show_terminals, f"{GIS_DIR}/Petroleum_terminals.csv", "Petroleum Terminals"),
    "ogim": (show_ogim, f"{GIS_DIR}/OGIM.csv", "OGIM"),
    "global": (show_global_tracker, f"{GIS_DIR}/Global-Oil-and-Gas-Extraction-Tracker-Feb-2025 - Main data.csv", "Global Oil and Gas Extraction Tracker"),
}

# Remove Organisation Company data loading and related functions

# Maintain map view state in session_state to prevent reset on slider change
if 'view_state' not in st.session_state:
    st.session_state['view_state'] = dict(latitude=0, longitude=0, zoom=2, pitch=0)

# Set initial view
all_points = []
if show_emit and not df_emit.empty and "latitude" in df_emit.columns and "longitude" in df_emit.columns:
    all_points.append(df_emit.dropna(subset=["latitude", "longitude"])[["latitude", "longitude"]])
if show_cams and not df_cams.empty and "latitude" in df_cams.columns and "longitude" in df_cams.columns:
    all_points.append(df_cams.dropna(subset=["latitude", "longitude"])[["latitude", "longitude"]])
if all_points and st.session_state['view_state']['latitude'] == 0 and st.session_state['view_state']['longitude'] == 0:
    concat_points = pd.concat(all_points)
    st.session_state['view_state']['latitude'] = concat_points["latitude"].mean()
    st.session_state['view_state']['longitude'] = concat_points["longitude"].mean()

view_state = pdk.ViewState(
    latitude=st.session_state['view_state']['latitude'],
    longitude=st.session_state['view_state']['longitude'],
    zoom=st.session_state['view_state']['zoom'],
    pitch=st.session_state['view_state']['pitch'],
)

def on_view_state_change(view_state):
    st.session_state['view_state'] = dict(latitude=view_state.latitude, longitude=view_state.longitude, zoom=view_state.zoom, pitch=view_state.pitch)

# --- Overview Map Section ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Overview Map</div>', unsafe_allow_html=True)
st.subheader("Methane Emissions Data Map")

# Tooltip logic
if show_emit and not show_cams:
    tooltip = {"text": "Date: {date}\nTime: {time_UTC}\nFlux: {flux t/h} t/h\nUncertainty: {uncertainty t/h} t/h"}
elif show_cams and not show_emit:
    tooltip = {"text": "Date: {date}\nTime: {time_UTC}\nFlux: {flux t/h} t/h\nUncertainty: {uncertainty t/h} t/h"}
else:
    tooltip = {"text": "Date: {date}\nTime: {time_UTC}\nFlux: {flux t/h} t/h\nUncertainty: {uncertainty t/h} t/h"}

# Define ESRI satellite basemap layer before using it
esri_satellite = pdk.Layer(
    "TileLayer",
    data=None,
    min_zoom=0,
    max_zoom=19,
    tile_size=256,
    get_tile_url="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    id="esri-satellite-base",
)

# Initialize layers list with the basemap
layers = [esri_satellite]

# Add EMIT points if toggled
if show_emit:
    df_emit_points = df_emit.dropna(subset=["latitude", "longitude"])
    if len(df_emit_points) > 1000:
        df_emit_points = df_emit_points.sample(1000, random_state=42)
    if not df_emit_points.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=df_emit_points,
            get_position="[longitude, latitude]",
            get_radius=4000,
            get_fill_color="[68, 1, 84, 160]",  # #440154
            pickable=True,
            auto_highlight=True,
            id="emit-points"
        ))

# Add CAMS points if toggled
if show_cams:
    df_cams_points = df_cams.dropna(subset=["latitude", "longitude"])
    if len(df_cams_points) > 1000:
        df_cams_points = df_cams_points.sample(1000, random_state=42)
    if not df_cams_points.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=df_cams_points,
            get_position="[longitude, latitude]",
            get_radius=4000,
            get_fill_color="[41, 175, 127, 160]",  # #29af7f
            pickable=True,
            auto_highlight=True,
            id="cams-points"
        ))

# Load EMIT polygon data before any map/layers logic
emit_polygons_df = pd.DataFrame()
try:
    emit_polygons_df = pd.read_csv('data/emission/emit_polygons.csv')
except Exception:
    emit_polygons_df = pd.DataFrame()

# Add EMIT polygons to map if toggled
if show_emit_polygons and not emit_polygons_df.empty and 'geometry' in emit_polygons_df.columns:
    import ast
    # Parse WKT or GeoJSON-like geometry to coordinates for pydeck
    def parse_polygon(geom):
        try:
            # Assume geometry is a stringified list of [lon, lat] pairs
            coords = ast.literal_eval(geom)
            # pydeck expects [[[lon, lat], ...]] for PolygonLayer
            if isinstance(coords, list) and isinstance(coords[0], list) and isinstance(coords[0][0], (float, int)):
                return [coords]
            return coords
        except Exception:
            return None
    emit_polygons_df = emit_polygons_df.copy()
    emit_polygons_df['polygon_coords'] = emit_polygons_df['geometry'].apply(parse_polygon)
    emit_polygons_df = emit_polygons_df.dropna(subset=['polygon_coords'])
    layers.append(pdk.Layer(
        "PolygonLayer",
        data=emit_polygons_df,
        get_polygon="polygon_coords",
        get_fill_color="[68, 1, 84, 80]",
        get_line_color="[68, 1, 84, 200]",
        pickable=True,
        auto_highlight=True,
        stroked=True,
        filled=True,
        extruded=False,
        wireframe=True,
        id="emit-polygons"
    ))

# Center map on mean of all points
all_points = []
if show_emit and not df_emit.empty and "latitude" in df_emit.columns and "longitude" in df_emit.columns:
    all_points.append(df_emit.dropna(subset=["latitude", "longitude"])[["latitude", "longitude"]])
if show_cams and not df_cams.empty and "latitude" in df_cams.columns and "longitude" in df_cams.columns:
    all_points.append(df_cams.dropna(subset=["latitude", "longitude"])[["latitude", "longitude"]])
if all_points:
    concat_points = pd.concat(all_points)
    center_lat = concat_points["latitude"].mean()
    center_lon = concat_points["longitude"].mean()
else:
    center_lat, center_lon = 0, 0

view_state = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lon,
    zoom=3,
    pitch=0,
)

st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=view_state,
    layers=layers,
    tooltip=tooltip
))

# --- Dynamic Legend ---
legend_items = []
if show_emit:
    legend_items.append(("#440154", "EMIT", "circle"))
if show_emit_polygons:
    legend_items.append(("#440154", "EMIT Polygons", "polygon"))
if show_cams:
    legend_items.append(("#29af7f", "CAMS", "circle"))
if show_crude:
    legend_items.append(("#000000", "Crude Oil Refineries", "circle"))
if show_gathering:
    legend_items.append(("#000000", "Gathering and Processing", "circle"))
if show_flaring:
    legend_items.append(("#000000", "Nat Gas Flaring Detections", "circle"))
if show_terminals:
    legend_items.append(("#000000", "Petroleum Terminals", "circle"))
if show_ogim:
    legend_items.append(("#000000", "OGIM", "circle"))
if show_global_tracker:
    legend_items.append(("#000000", "Global Oil and Gas Extraction Tracker", "circle"))

legend_html = "<div style='display: flex; align-items: center; flex-wrap: wrap;'>"
for color, label, shape in legend_items:
    if shape == "circle":
        legend_html += f"<div style='width: 20px; height: 20px; background: {color}; margin-right: 8px; border-radius: 50%; display: inline-block;'></div>"
    elif shape == "polygon":
        legend_html += f"<div style='width: 20px; height: 20px; background: {color}33; border: 2px solid {color}; margin-right: 8px; display: inline-block; clip-path: polygon(5% 95%, 50% 5%, 95% 95%);'></div>"
    legend_html += f"<span style='margin-right: 16px;'>{label}</span>"
legend_html += "</div>"
if legend_items:
    st.markdown(legend_html, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_admin_boundaries():
    return gpd.read_file("data/GIS/geoBoundariesCGAZ_ADM1.geojson")

admin_gdf = load_admin_boundaries()

# --- Load emissions data for country mapping ---
try:
    emit_df = pd.read_csv('data/emission/emit_clean.csv')
except Exception:
    emit_df = pd.DataFrame()

# Build ISO3 -> country name mapping from emissions data
iso3_to_country = {}
if not emit_df.empty:
    for _, row in emit_df.iterrows():
        iso3 = str(row.get('ISO3166-1-Alpha-3', '')).strip()
        name = str(row.get('name', '')).strip() or str(row.get('COUNTRY', '')).strip()
        if iso3 and name:
            iso3_to_country[iso3] = name

# --- Emissions Analysis Section ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Emissions Analysis</div>', unsafe_allow_html=True)
col_menu, col_emissions, col_gis = st.columns([2, 2, 2])

# --- Country and region selection menu (left) ---
with col_menu:
    # Country selection is now unified (selected_country)
    # Region selection dropdown for adm_region in the selected country (from emit_df and cams_df)
    region_options = []
    selected_region = None
    if selected_country:
        # Get all unique adm_region values for the selected country from both datasets
        emit_regions = []
        cams_regions = []
        if 'adm_region' in emit_df.columns and 'name' in emit_df.columns:
            emit_df = emit_df.copy()
            emit_df['name'] = emit_df['name'].astype(str)
            emit_df['adm_region'] = emit_df['adm_region'].astype(str)
            country_name = get_country_name(selected_country)
            emit_regions = emit_df[emit_df['name'].astype(str).str.strip().str.lower() == country_name.strip().lower()]['adm_region'].dropna().unique()
        if 'adm_region' in df_cams.columns and 'name' in df_cams.columns:
            df_cams = df_cams.copy()
            df_cams['name'] = df_cams['name'].astype(str)
            df_cams['adm_region'] = df_cams['adm_region'].astype(str)
            country_name = get_country_name(selected_country)
            cams_regions = df_cams[df_cams['name'].astype(str).str.strip().str.lower() == country_name.strip().lower()]['adm_region'].dropna().unique()
        region_options = sorted(set(list(emit_regions) + list(cams_regions)))
        if region_options:
            selected_region = st.selectbox("Select region (ADM1):", region_options, key="region_selectbox")
    update_map = st.button("Go", key="update_map")
    spinner_placeholder = st.empty()
    if update_map and selected_country:
        with spinner_placeholder:
            st.markdown(
                '''
                <div style="display: flex; align-items: center;">
                    <div class="loader" style="border: 4px solid #f3f3f3; border-top: 4px solid #fde725; border-radius: 50%; width: 24px; height: 24px; animation: spin 1s linear infinite; margin-right: 8px;"></div>
                    <span style="font-size: 1rem;">Processing...</span>
                </div>
                <style>
                @keyframes spin {
                    0% { transform: rotate(0deg);}
                    100% { transform: rotate(360deg);}
                }
                </style>
                ''',
                unsafe_allow_html=True
            )
            time.sleep(2.0)
        st.session_state['selected_country_code'] = selected_country
        st.session_state['selected_region'] = selected_region
        spinner_placeholder.empty()
    elif 'selected_country_code' not in st.session_state and selected_country:
        st.session_state['selected_country_code'] = selected_country
        st.session_state['selected_region'] = selected_region
active_country = st.session_state.get('selected_country_code', None)
active_region = st.session_state.get('selected_region', None)

# --- Helper for country/region filtering ---
def filter_for_analysis(df):
    # Use ISO3 code for country filtering
    if active_country and 'ISO3166-1-Alpha-3' in df.columns:
        df = df[df['ISO3166-1-Alpha-3'] == active_country]
    # Use adm_region column for region filtering if present
    if active_region and 'adm_region' in df.columns:
        df = df[df['adm_region'] == active_region]
    return df

# --- Prepare analysis data ---
def monthly_emissions(df, label):
    if "date" in df.columns and "flux t/h" in df.columns:
        df = df.copy()
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date_parsed"])
        df["Month"] = df["date_parsed"].dt.strftime("%Y-%m")
        summary = df.groupby("Month")["flux t/h"].sum().reset_index()
        summary.columns = ["Month", f"{label} Emissions (t/h sum)"]
        return summary
    return pd.DataFrame()

emit_analysis = filter_for_analysis(df_emit)
cams_analysis = filter_for_analysis(df_cams)

# After loading emit_df and cams_df, add debug printouts and robust filtering
if 'ISO3166-1-Alpha-3' in emit_df.columns:
    emit_df['ISO3166-1-Alpha-3'] = emit_df['ISO3166-1-Alpha-3'].astype(str).str.strip().str.upper()
else:
    selected_country_upper = selected_country.strip().upper()

country_name = get_country_name(selected_country)

# --- Monthly emissions chart (center) ---
with col_emissions:
    emit_country = emit_df.copy()
    cams_country = df_cams.copy()
    if country_name and 'name' in emit_country.columns:
        emit_country = emit_country[emit_country['name'].astype(str).str.strip().str.lower() == country_name.strip().lower()]
    if country_name and 'name' in cams_country.columns:
        cams_country = cams_country[cams_country['name'].astype(str).str.strip().str.lower() == country_name.strip().lower()]
    emit_monthly = monthly_emissions(emit_country, "EMIT")
    cams_monthly = monthly_emissions(cams_country, "CAMS")
    emit_has_data = not emit_monthly.empty and "Month" in emit_monthly.columns
    cams_has_data = not cams_monthly.empty and "Month" in cams_monthly.columns
    color_map = {"EMIT Emissions (t/h sum)": "#440154", "CAMS Emissions (t/h sum)": "#29af7f"}
    if emit_has_data and cams_has_data:
        merged = pd.merge(emit_monthly, cams_monthly, on="Month", how="outer").sort_values("Month").fillna(0)
        st.write(f"Monthly summed emissions (t/h) for {get_country_name(active_country)}:")
        chart = alt.Chart(merged).transform_fold(
            [col for col in merged.columns if col != "Month"],
            as_=["Dataset", "Emissions"]
        ).mark_bar(size=18).encode(
            x=alt.X("Month:O", axis=alt.Axis(labelAngle=0, title="Month")),
            y=alt.Y("Emissions:Q", title="Emissions (t/h sum)"),
            color=alt.Color("Dataset:N", scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())), legend=alt.Legend(title="Dataset")),
            tooltip=["Month:O", "Dataset:N", "Emissions:Q"]
        ).properties(
            width=1100,
            height=350
        )
        st.altair_chart(chart, use_container_width=True)
    elif emit_has_data:
        st.write(f"Monthly summed EMIT emissions (t/h) for {get_country_name(active_country)}:")
        chart = alt.Chart(emit_monthly).mark_bar(size=18, color="#440154").encode(
            x=alt.X("Month:O", axis=alt.Axis(labelAngle=0, title="Month")),
            y=alt.Y(f"EMIT Emissions (t/h sum):Q", title="Emissions (t/h sum)"),
            tooltip=["Month:O", f"EMIT Emissions (t/h sum):Q"]
        ).properties(
            width=1100,
            height=350
        )
        st.altair_chart(chart, use_container_width=True)
    elif cams_has_data:
        st.write(f"Monthly summed CAMS emissions (t/h) for {get_country_name(active_country)}:")
        chart = alt.Chart(cams_monthly).mark_bar(size=18, color="#29af7f").encode(
            x=alt.X("Month:O", axis=alt.Axis(labelAngle=0, title="Month")),
            y=alt.Y(f"CAMS Emissions (t/h sum):Q", title="Emissions (t/h sum)"),
            tooltip=["Month:O", f"CAMS Emissions (t/h sum):Q"]
        ).properties(
            width=1100,
            height=350
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No monthly emissions data available for this country.")

    # --- Region-level emissions analysis (below country chart) ---
    if active_country and active_region:
        # Filter for selected region (case-insensitive, strip whitespace)
        emit_region = emit_country.copy()
        cams_region = cams_country.copy()
        if 'adm_region' in emit_region.columns:
            emit_region = emit_region.copy()
            emit_region = emit_region[emit_region['adm_region'].astype(str).str.strip().str.lower() == active_region.strip().lower()]
        if 'adm_region' in cams_region.columns:
            cams_region = cams_region.copy()
            cams_region = cams_region[cams_region['adm_region'].astype(str).str.strip().str.lower() == active_region.strip().lower()]
        emit_monthly_region = monthly_emissions(emit_region, "EMIT")
        cams_monthly_region = monthly_emissions(cams_region, "CAMS")
        emit_region_has_data = not emit_monthly_region.empty and "Month" in emit_monthly_region.columns
        cams_region_has_data = not cams_monthly_region.empty and "Month" in cams_monthly_region.columns
        color_map_region = {"EMIT Emissions (t/h sum)": "#440154", "CAMS Emissions (t/h sum)": "#29af7f"}
        if emit_region_has_data or cams_region_has_data:
            st.write(f"Monthly summed emissions (t/h) for {active_region} (ADM1 region):")
            if emit_region_has_data and cams_region_has_data:
                merged_region = pd.merge(emit_monthly_region, cams_monthly_region, on="Month", how="outer").sort_values("Month").fillna(0)
                chart_region = alt.Chart(merged_region).transform_fold(
                    [col for col in merged_region.columns if col != "Month"],
                    as_=["Dataset", "Emissions"]
                ).mark_bar(size=18).encode(
                    x=alt.X("Month:O", axis=alt.Axis(labelAngle=0, title="Month")),
                    y=alt.Y("Emissions:Q", title="Emissions (t/h sum)"),
                    color=alt.Color("Dataset:N", scale=alt.Scale(domain=list(color_map_region.keys()), range=list(color_map_region.values())), legend=alt.Legend(title="Dataset")),
                    tooltip=["Month:O", "Dataset:N", "Emissions:Q"]
                ).properties(
                    width=1100,
                    height=350
                )
                st.altair_chart(chart_region, use_container_width=True)
            elif emit_region_has_data:
                chart_region = alt.Chart(emit_monthly_region).mark_bar(size=18, color="#440154").encode(
                    x=alt.X("Month:O", axis=alt.Axis(labelAngle=0, title="Month")),
                    y=alt.Y(f"EMIT Emissions (t/h sum):Q", title="Emissions (t/h sum)"),
                    tooltip=["Month:O", f"EMIT Emissions (t/h sum):Q"]
                ).properties(
                    width=1100,
                    height=350
                )
                st.altair_chart(chart_region, use_container_width=True)
            elif cams_region_has_data:
                chart_region = alt.Chart(cams_monthly_region).mark_bar(size=18, color="#29af7f").encode(
                    x=alt.X("Month:O", axis=alt.Axis(labelAngle=0, title="Month")),
                    y=alt.Y(f"CAMS Emissions (t/h sum):Q", title="Emissions (t/h sum)"),
                    tooltip=["Month:O", f"CAMS Emissions (t/h sum):Q"]
                ).properties(
                    width=1100,
                    height=350
                )
                st.altair_chart(chart_region, use_container_width=True)
        else:
            st.info(f"No monthly emissions data available for {active_region} (ADM1 region).")

# --- GIS Facility Count Analysis (right column) ---
with col_gis:
    if isinstance(gis_df, pd.DataFrame) and not gis_df.empty and active_country:
        # Filter by country (case-insensitive, handle missing COUNTRY column)
        if "COUNTRY" in gis_df.columns:
            gis_country = gis_df[gis_df["COUNTRY"].astype(str).str.upper() == iso3_to_country.get(active_country, active_country).upper()]
        elif "name" in gis_df.columns:
            gis_country = gis_df[gis_df["name"].astype(str).str.upper() == iso3_to_country.get(active_country, active_country).upper()]
        else:
            gis_country = pd.DataFrame()

        if not gis_country.empty:
            facility_counts = gis_country["_dataset"].value_counts().reset_index()
            facility_counts.columns = ["Facility Type", "Count"]
            st.write(f"Facility count by type in {get_country_name(active_country)}:")
            chart = alt.Chart(facility_counts).mark_bar(size=18, color="#fde725").encode(
                x=alt.X("Facility Type:N", sort="-y", title="Facility Type", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Count:Q", title="Number of Facilities"),
                tooltip=["Facility Type", "Count"]
            ).properties(
                width=1100,
                height=350
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info(f"No GIS facilities found for {get_country_name(active_country)}.")
    else:
        st.info("GIS data not loaded or available.")

# --- Unified Trade Analysis Section (Oil & Gas) ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Country Oil & Gas Trade Analysis (2021–2025)</div>', unsafe_allow_html=True)

# Prepare filtered data for each commodity
def prepare_filtered(df, commodity_label):
    df = df.copy()
    df['reporterDesc'] = df['reporterDesc'].astype(str).str.strip()
    df['partnerDesc'] = df['partnerDesc'].astype(str).str.strip()
    df = df[df['partnerDesc'].astype(str).str.lower() != 'world']
    filtered = df[
        (df['reporterISO'] == selected_country) &
        (df['flowCode'] == selected_flow)
    ]
    if not filtered.empty:
        filtered['actual_qty'] = pd.to_numeric(filtered['isAltQtyEstimated'], errors='coerce')
        filtered = filtered.dropna(subset=['actual_qty'])
        filtered = filtered[filtered['actual_qty'] > 0]
        filtered = filtered[filtered['partnerISO'].astype(str).str.lower() != 'world']
        filtered['year'] = filtered['refPeriodId'].astype(str).str[:4]
        filtered['commodity'] = commodity_label
        return filtered
    return pd.DataFrame()

if selected_commodity == 'Both':
    filtered_gas = prepare_filtered(gas_df, 'Gas') if gas_df is not None else pd.DataFrame()
    filtered_oil = prepare_filtered(oil_df, 'Oil') if oil_df is not None else pd.DataFrame()
    filtered = pd.concat([filtered_gas, filtered_oil], ignore_index=True)
else:
    df = gas_df if selected_commodity == 'Gas' else oil_df
    filtered = prepare_filtered(df, selected_commodity)

if not filtered.empty:
    qty_by_year_partner_commodity = filtered.groupby(['year', 'partnerISO', 'commodity'], as_index=False)['actual_qty'].sum()
    st.subheader(f"{selected_flow} of {selected_country} ({selected_commodity}) by Year and Partner Country")
    import altair as alt
    if selected_commodity == 'Both':
        col1, col2 = st.columns(2)
        # Gas chart
        with col1:
            gas_data = qty_by_year_partner_commodity[qty_by_year_partner_commodity['commodity'] == 'Gas']
            st.markdown('**Gas**')
            if not gas_data.empty:
                chart_gas = alt.Chart(gas_data).mark_bar(size=18).encode(
                    x=alt.X('year:O', title='Year', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('actual_qty:Q', title='Total Quantity'),
                    color=alt.Color('partnerISO:N', title='Partner Country (ISO)'),
                    tooltip=['year', 'partnerISO', 'actual_qty']
                ).properties(width=400, height=350)
                st.altair_chart(chart_gas, use_container_width=True)
                st.dataframe(gas_data, use_container_width=True)
            else:
                st.info('No gas data available for the selected filters.')
        # Oil chart
        with col2:
            oil_data = qty_by_year_partner_commodity[qty_by_year_partner_commodity['commodity'] == 'Oil']
            st.markdown('**Oil**')
            if not oil_data.empty:
                chart_oil = alt.Chart(oil_data).mark_bar(size=18).encode(
                    x=alt.X('year:O', title='Year', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('actual_qty:Q', title='Total Quantity'),
                    color=alt.Color('partnerISO:N', title='Partner Country (ISO)'),
                    tooltip=['year', 'partnerISO', 'actual_qty']
                ).properties(width=400, height=350)
                st.altair_chart(chart_oil, use_container_width=True)
                st.dataframe(oil_data, use_container_width=True)
            else:
                st.info('No oil data available for the selected filters.')
    else:
        chart = alt.Chart(qty_by_year_partner_commodity).mark_bar(size=18).encode(
            x=alt.X('year:O', title='Year'),
            y=alt.Y('actual_qty:Q', title='Total Quantity'),
            color=alt.Color('partnerISO:N', title='Partner Country (ISO)'),
            tooltip=['year', 'partnerISO', 'commodity', 'actual_qty']
        ).properties(width=900, height=350)
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(qty_by_year_partner_commodity, use_container_width=True)
else:
    st.info("No data available for the selected filters.")
st.markdown('</div>', unsafe_allow_html=True)

# --- Methane Emission Attribution Section ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Methane Emission Attribution</div>', unsafe_allow_html=True)

# Use unified selectors
attr_country = selected_country
attr_commodity = selected_commodity

if attr_commodity == 'Both':
    commodities = ['Oil', 'Gas']
else:
    commodities = [attr_commodity]

# --- DEBUG: Print unique years, source_type, and country names in CAMS ---
cams_df['year'] = cams_df['date'].astype(str).str[:4]
cams_df['source_type'] = cams_df['source_type'].fillna('Unknown')
# Remove debug print statements (e.g., print(), st.write() for debugging)

# --- Exporter to CAMS country name mapping (expand as needed) ---
exporter_to_cams_name = {
    'USA': 'United States of America',
    'Russian Federation': 'Russia',
    'Rep. of Korea': 'South Korea',
    'Dem. Rep. of the Congo': 'Democratic Republic of the Congo',
    'Czechia': 'Czech Republic',
    'Türkiye': 'Turkey',
    'United Kingdom': 'United Kingdom',
    'Viet Nam': 'Vietnam',
    'Iran (Islamic Rep. of)': 'Iran',
    'Syrian Arab Republic': 'Syria',
    'Lao People\'s Dem. Rep.': 'Laos',
    'China, Hong Kong SAR': 'Hong Kong',
    'China, Macao SAR': 'Macau',
    'North Macedonia': 'Macedonia',
    'Republic of Moldova': 'Moldova',
    'Bolivia (Plurinational State of)': 'Bolivia',
    'Brunei Darussalam': 'Brunei',
    'United Rep. of Tanzania': 'Tanzania',
    # Add more as needed
}
# Remove debug print statements (e.g., print(), st.write() for debugging)

attr_rows = []
for commodity in commodities:
    trade_df = gas_df if commodity == 'Gas' else oil_df
    # Only consider imports to the selected country
    trade_df = trade_df.copy()
    trade_df['year'] = trade_df['refPeriodId'].astype(str).str[:4]
    trade_df = trade_df[trade_df['partnerISO'] == attr_country]
    trade_df = trade_df[trade_df['partnerISO'].astype(str).str.lower() != 'world']
    # Before any use of trade_flows['commodity'] in the attribution section:
    if 'commodity' not in trade_df.columns:
        if 'commodityDesc' in trade_df.columns:
            trade_df['commodity'] = trade_df['commodityDesc']
        else:
            trade_df['commodity'] = 'Unknown'
    # Remove debug print statements (e.g., print(), st.write() for debugging)
    exporters = set(trade_df['reporterISO'].unique())
    # Remove debug print statements (e.g., print(), st.write() for debugging)
    for year in sorted(trade_df['year'].unique()):
        trade_year = trade_df[trade_df['year'] == year]
        total_exports = trade_year.groupby('reporterISO')['isAltQtyEstimated'].sum().rename('total_export')
        pair_exports = trade_year.groupby(['reporterISO', 'partnerISO'])['isAltQtyEstimated'].sum().rename('pair_export')
        pair_exports = pair_exports.reset_index().merge(total_exports.reset_index(), on='reporterISO')
        pair_exports['export_share'] = pair_exports['pair_export'] / pair_exports['total_export']
        cams_em_by_country = cams_df[cams_df['year'] == year]
        cams_em_by_country = cams_em_by_country[cams_em_by_country['source_type'].astype(str).str.lower() == commodity.lower()]
        cams_countries = set(cams_em_by_country['name'].unique())
        # Remove debug print statements (e.g., print(), st.write() for debugging)
        # Intersection using mapping
        mapped_exporters = set([exporter_to_cams_name.get(e, e) for e in exporters])
        intersection = set([m.upper() for m in mapped_exporters]) & set([c.upper() for c in cams_countries])
        # Remove debug print statements (e.g., print(), st.write() for debugging)
        cams_em_by_country = cams_em_by_country.groupby(['name', 'year', 'source_type'])['flux t/h'].sum().reset_index()
        for _, row in pair_exports.iterrows():
            exporter = row['reporterISO']
            mapped_exporter = exporter_to_cams_name.get(exporter, exporter)
            share = row['export_share']
            cams_row = cams_em_by_country[cams_em_by_country['name'].astype(str).str.upper() == mapped_exporter.upper()]
            if not cams_row.empty:
                for _, em_row in cams_row.iterrows():
                    attributed = em_row['flux t/h'] * share
                    attr_rows.append({
                        'Exporter': exporter,
                        'Importer': attr_country,
                        'Year': year,
                        'Source Type': em_row['source_type'],
                        'Attributed Emissions (t/h)': attributed,
                        'Commodity': commodity
                    })
attr_df = pd.DataFrame(attr_rows)

if not attr_df.empty:
    st.write(f'Attributed CAMS emissions (t/h) to {attr_country} by exporter, year, and source type:')
    st.dataframe(attr_df, use_container_width=True)
    chart = alt.Chart(attr_df).mark_bar().encode(
        x=alt.X('Exporter:N', title='Exporter Country'),
        y=alt.Y('Attributed Emissions (t/h):Q', title='Attributed Emissions (t/h)'),
        color=alt.Color('Source Type:N', title='Source Type'),
        column=alt.Column('Year:N', title='Year'),
        tooltip=['Exporter', 'Importer', 'Year', 'Source Type', 'Attributed Emissions (t/h)', 'Commodity']
    ).properties(width=300, height=400)
    st.altair_chart(chart, use_container_width=True)
else:
    st.info('No attributed emissions found for the selected country and commodity.') 