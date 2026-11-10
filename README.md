# ThermoLoop AI — Multi-storey Residential Building Simulator

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
'''

## What it simulates
- 2–30 storeys, floor plate area, envelope U-values, window ratio, ACH and thermal mass.
- Synthetic urban weather, urban heat-island uplift, solar shading and orientation.
- Synthetic occupancy, people/appliance/solar gains.
- One lumped thermal zone per floor, with roof/ground exposure and inter-floor heat transfer.
- Heating demand in kWth and interval energy in kWhth.
- HistGradientBoosting or Random Forest trained with a chronological 80/20 hold-out.
- Rolling 6/12/24/48-hour demonstration forecast and CSV exports.

## Limitations
This is a synthetic physics-informed simulator, not a calibrated digital twin. The Pune-like weather profile is illustrative, not actual historical/live Pune weather. One zone per floor is a simplification: individual flats, rooms, corridors, thermal bridges, HVAC distribution, detailed glazing and CFD are not represented. Occupancy and internal gains are synthetic. The future weather and occupancy in the forecast are synthetic scenario inputs, not an external weather forecast. The ML validation scores only show fit to labels produced by this same simulator; they do not establish real-building forecasting accuracy. Calibrate U-values, infiltration, thermal capacity, occupancy, controller behaviour and heating-system efficiency against measured building/BMS/meter data before making real-world performance claims. Cooling, domestic hot water, heat-pump COP, network losses and thermal storage are not included.

## Units
Heating demand: kWth. Interval thermal energy: kWhth = kWth × interval duration in hours. U-values: W/(m²·K).
