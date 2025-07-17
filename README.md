# Methane Emissions Monitoring Dashboard

This project provides a dashboard for visualizing and analyzing methane emissions data (EMIT, CAMS, and GIS facility layers) with country and region (ADM1) filtering, time sliders, and interactive maps.

## Features
- Interactive map with emissions and facility overlays
- Region/country filtering and analysis
- Data sourced from EMIT, CAMS, and GIS datasets
- Backend API (FastAPI) and Streamlit frontend

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/GHG_monitoring.git
cd GHG_monitoring
```

### 2. Create and Activate a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Data Files
**Note:** Large data files (in `data/`) are not included in this repository (see `.gitignore`).
- Please contact the maintainer or use the provided download link to obtain the necessary data files.
- Place the data files in the correct directories (e.g., `data/emission/`, `data/GIS/`).

### 5. Run the Backend (FastAPI)
```bash
cd backend
uvicorn main:app --reload
```
- The backend will be available at `http://localhost:8000`

### 6. Run the Frontend (Streamlit)
Open a new terminal in the project root:
```bash
streamlit run frontend_app.py
```
- The frontend will be available at the local URL shown in the terminal (e.g., `http://localhost:8501`)

## Deployment to Streamlit Cloud
- Push your code (without large data files) to GitHub.
- Deploy via [Streamlit Cloud](https://streamlit.io/cloud) by linking your repo and specifying `frontend_app.py` as the main file.
- For public demos, use sample data or provide code to download data at runtime.

## Notes
- Data files are ignored by git via `.gitignore` and must be shared separately.
- For any issues, please open an issue or contact the maintainer. 