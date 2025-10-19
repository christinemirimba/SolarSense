# ☀️ SolarSense: AI-Powered Solar Forecasting for SDG 7

**Author**: Christine Mirimba  | Alfred Nyongesa | Hannah Shekinah
**Repository**: [github.com/christinemirimba/SolarSense](https://github.com/christinemirimba/SolarSense)  
**SDG Focus**: SDG 7 – Affordable and Clean Energy  
**Tools**: Python, Scikit-learn, TensorFlow, NASA POWER API, Jupyter Notebook

---

## 🌍 Project Overview

SolarSense is a machine learning solution that predicts daily solar energy potential (kWh/m²/day) using routine weather variables. By enabling accurate solar forecasting, this project supports clean energy planning, grid optimization, and equitable access to renewable energy—especially in underserved regions.

---

## 📊 Dataset

**Source**: [NASA POWER API](https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN,T2M,RH2M,WS2M,PRECTOTCORR&start=20180101&end=20231231&latitude=-1.2921&longitude=36.8219&community=RE&format=CSV&header=true)  
**Location**: Nairobi, Kenya  
**Period**: Jan 2018 – Dec 2023  
**Features**:
- `T2M`: Air Temperature at 2 meters  
- `RH2M`: Relative Humidity at 2 meters  
- `WS2M`: Wind Speed at 2 meters  
- `PRECTOTCORR`: Precipitation  
- `ALLSKY_SFC_SW_DWN`: Target variable – Solar irradiance

---

## 💻 How to Run the Project (Terminal Instructions)

### 1. Clone the Repository
```bash
git clone https://github.com/christinemirimba/SolarSense.git
cd SolarSense/solar_env

### 2. Create and Activate Virtual Environment
bash
python -m venv venv
# Activate the environment
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
### 3. Install Dependencies
bash
pip install -r requirements.txt
### 4. Run the Jupyter Notebook
bash
jupyter notebook solar_energy_analysis.ipynb
###

5. Run the Python Script (Optional)
bash
python solar_energy_predictor.py
⚖️ Ethical Reflection
Bias Risks: Temporal and regional biases mitigated through multi-year data and localized training

Fairness: Promotes clean energy access and supports sustainable infrastructure planning

Sustainability: Reduces reliance on fossil fuels by enabling smarter solar deployment

🚀 Stretch Goals
Integrate real-time weather data via API

Deploy as a web app using Flask or Streamlit

Compare multiple ML models for performance optimization


