
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="ThermoLoop Building Simulator", layout="wide")
st.title("ThermoLoop AI | Multi-storey Residential Building Simulator")
st.caption("Synthetic floor-level thermal simulation + chronological ML forecast. This is a prototype, not a calibrated digital twin.")

def solar_curve(hour, peak):
    return peak * max(0.0, math.sin(math.pi * (hour - 6.0) / 12.0))

def weather_series(n, dt_h, seed, mean_t, swing, solar_peak, cloud, uhi):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=n, freq=pd.Timedelta(hours=dt_h))
    hour = ts.hour.to_numpy() + ts.minute.to_numpy()/60
    day = np.arange(n)*dt_h/24
    daily = swing*np.sin(2*np.pi*(hour-9)/24)
    slow = 1.5*np.sin(2*np.pi*day/max(day[-1]+1, 1))
    synoptic = 1.5*np.sin(2*np.pi*day/5)
    night = 0.5 + 0.5*np.cos(2*np.pi*(hour-2)/24)
    tout = mean_t + daily + slow + synoptic + uhi*night + rng.normal(0, .6, n)
    clouds = np.clip(1-cloud+rng.normal(0,.12,n), .15, 1)
    solar = np.array([solar_curve(h, solar_peak) for h in hour])*clouds
    wind = np.clip(2+rng.normal(0,.5,n), .1, 8)
    return pd.DataFrame({"timestamp":ts, "outdoor_temperature_C":tout,
                         "solar_irradiance_W_m2":solar, "wind_speed_m_s":wind,
                         "hour":hour, "day_of_week":ts.dayofweek})

def build_floor_table(c):
    rows=[]
    perimeter=4*math.sqrt(c["area"])
    wall_gross=perimeter*3.0
    win=wall_gross*c["wwr"]
    wall=wall_gross-win
    for f in range(c["floors"]):
        ground=(f==0); top=(f==c["floors"]-1)
        ua_env=c["uwall"]*wall+c["uwin"]*win
        if top: ua_env += c["uroof"]*c["area"]
        if ground: ua_env += .45*c["area"]
        volume=c["area"]*3
        ua_vent=.33*c["ach"]*volume
        exposure=1.18 if ground or top else 1.0
        cap=c["cap"]*(1.15 if ground or top else 1.0)
        solar_factor=c["shade"]*(.72+.28*(f+1)/c["floors"])*c["orientation"]
        rows.append({"floor":f+1,"ua_total_W_K":(ua_env+ua_vent)*exposure,
                     "thermal_capacity_kWh_K":cap,"solar_factor":solar_factor,
                     "is_ground_floor":int(ground),"is_top_floor":int(top)})
    return pd.DataFrame(rows)

def occupancy(ts, floor, residents, weekend_fraction):
    h=ts.hour+ts.minute/60
    morning=.62*np.exp(-.5*((h-7.5)/1.8)**2)
    evening=.90*np.exp(-.5*((h-20)/2.5)**2)
    daytime=weekend_fraction if ts.dayofweek>=5 else .18
    return residents*np.clip(daytime+morning+evening, .05, 1)*(0.94+.025*(floor%4))

def simulate(c):
    dt_h=c["dt"]/60
    n=int(c["days"]*24/dt_h)
    wx=weather_series(n,dt_h,c["seed"],c["mean_t"],c["swing"],c["solar_peak"],c["cloud"],c["uhi"])
    fp=build_floor_table(c)
    nf=c["floors"]
    tin=np.full(nf,c["setpoint"]-1.0)
    out=[]
    for i,w in wx.iterrows():
        qheat=np.zeros(nf); qgain=np.zeros(nf); rows=[]
        for j in range(nf):
            p=fp.iloc[j]
            occ=occupancy(w.timestamp,j+1,c["residents"],c["weekend"])
            people=occ*c["person_gain"]
            appliances=c["appliance_gain"]*c["area"]
            solar=w.solar_irradiance_W_m2*c["wwr"]*c["area"]*.55*p.solar_factor
            gains=people+appliances+solar
            ua=p.ua_total_W_K
            # Simplified thermostat: offset envelope losses, gains and indoor error.
            demand=max(0,ua*(c["setpoint"]-w.outdoor_temperature_C)-gains+
                       c["kp"]*(c["setpoint"]-tin[j]))
            demand=min(demand,c["max_heat"]*1000)
            qheat[j]=demand; qgain[j]=gains
            rows.append({"timestamp":w.timestamp,"floor":j+1,
                "outdoor_temperature_C":w.outdoor_temperature_C,
                "solar_irradiance_W_m2":w.solar_irradiance_W_m2,
                "wind_speed_m_s":w.wind_speed_m_s,"occupancy_count":occ,
                "indoor_temperature_C":tin[j],"heating_setpoint_C":c["setpoint"],
                "internal_gains_kW":gains/1000,"heating_demand_kWth":demand/1000,
                "ua_total_W_K":ua,"thermal_capacity_kWh_K":p.thermal_capacity_kWh_K,
                "solar_factor":p.solar_factor,"is_top_floor":p.is_top_floor,
                "is_ground_floor":p.is_ground_floor,"hour":w.hour,
                "day_of_week":w.day_of_week})
        # Explicit Euler thermal balance, with inter-floor heat transfer.
        for j in range(nf):
            cap_j=fp.iloc[j].thermal_capacity_kWh_K*3.6e6
            qloss=fp.iloc[j].ua_total_W_K*(tin[j]-w.outdoor_temperature_C)
            qinter=0
            if j>0: qinter += c["inter"]*(tin[j-1]-tin[j])
            if j<nf-1: qinter += c["inter"]*(tin[j+1]-tin[j])
            tin[j]=np.clip(tin[j]+(qheat[j]+qgain[j]-qloss+qinter)*dt_h*3600/cap_j,5,35)
        for j,r in enumerate(rows):
            r["indoor_temperature_next_C"]=tin[j]
            r["heating_energy_kWhth"]=r["heating_demand_kWth"]*dt_h
            out.append(r)
    floor=pd.DataFrame(out)
    agg=floor.groupby("timestamp",as_index=False).agg(
        outdoor_temperature_C=("outdoor_temperature_C","mean"),
        solar_irradiance_W_m2=("solar_irradiance_W_m2","mean"),
        wind_speed_m_s=("wind_speed_m_s","mean"),
        occupancy_count=("occupancy_count","sum"),
        indoor_temperature_C=("indoor_temperature_C","mean"),
        indoor_temperature_next_C=("indoor_temperature_next_C","mean"),
        heating_setpoint_C=("heating_setpoint_C","mean"),
        internal_gains_kW=("internal_gains_kW","sum"),
        heating_demand_kWth=("heating_demand_kWth","sum"),
        hour=("hour","first"),day_of_week=("day_of_week","first"))
    agg["heating_energy_kWhth"]=agg.heating_demand_kWth*dt_h
    agg["ua_total_W_K"]=fp.ua_total_W_K.sum()
    agg["thermal_capacity_kWh_K"]=fp.thermal_capacity_kWh_K.sum()
    return floor,agg,wx,fp

FEATURES=["outdoor_temperature_C","solar_irradiance_W_m2","wind_speed_m_s",
"occupancy_count","indoor_temperature_C","heating_setpoint_C","internal_gains_kW",
"hour_sin","hour_cos","dow_sin","dow_cos","demand_lag_1","demand_lag_4",
"indoor_lag_1","demand_roll_4","demand_roll_24","temp_delta"]

def add_features(d,dt):
    x=d.copy().sort_values("timestamp").reset_index(drop=True)
    x["hour_sin"]=np.sin(2*np.pi*x.hour/24); x["hour_cos"]=np.cos(2*np.pi*x.hour/24)
    x["dow_sin"]=np.sin(2*np.pi*x.day_of_week/7); x["dow_cos"]=np.cos(2*np.pi*x.day_of_week/7)
    x["demand_lag_1"]=x.heating_demand_kWth.shift(1)
    x["demand_lag_4"]=x.heating_demand_kWth.shift(4)
    x["indoor_lag_1"]=x.indoor_temperature_C.shift(1)
    x["demand_roll_4"]=x.heating_demand_kWth.shift(1).rolling(4).mean()
    x["demand_roll_24"]=x.heating_demand_kWth.shift(1).rolling(max(1,int(24*60/dt))).mean()
    x["temp_delta"]=x.heating_setpoint_C-x.indoor_temperature_C
    return x.dropna().reset_index(drop=True)

def fit_model(agg,dt,horizon,kind,c,fp):
    x=add_features(agg,dt)
    split=int(.8*len(x)); train=x.iloc[:split]; test=x.iloc[split:]
    if kind=="Random Forest":
        model=RandomForestRegressor(n_estimators=220,max_depth=16,min_samples_leaf=2,
                                    random_state=42,n_jobs=-1)
    else:
        model=HistGradientBoostingRegressor(max_iter=220,learning_rate=.06,
                    max_leaf_nodes=20,l2_regularization=1,random_state=42)
    model.fit(train[FEATURES],train.heating_demand_kWth)
    pred=np.maximum(0,model.predict(test[FEATURES]))
    scores={"MAE (kWth)":mean_absolute_error(test.heating_demand_kWth,pred),
            "RMSE (kWth)":mean_squared_error(test.heating_demand_kWth,pred)**.5,
            "R²":r2_score(test.heating_demand_kWth,pred)}
    # Synthetic future exogenous drivers: demo forecast only, not a real weather forecast.
    dt_h=dt/60; steps=int(horizon/dt_h)
    last=agg.iloc[-1]; last_ts=pd.Timestamp(last.timestamp)
    future=pd.date_range(last_ts+pd.Timedelta(minutes=dt),periods=steps,freq=pd.Timedelta(minutes=dt))
    hist=list(agg.heating_demand_kWth.tail(96).astype(float))
    indoor=float(last.indoor_temperature_next_C)
    base_hour=last_ts.hour+last_ts.minute/60
    rows=[]
    for k,ts in enumerate(future):
        h=ts.hour+ts.minute/60
        tout=float(last.outdoor_temperature_C+3.0*math.sin(2*math.pi*(h-9)/24)
                   -3.0*math.sin(2*math.pi*(base_hour-9)/24))
        solar=solar_curve(h,550)*.72
        # Preserve the aggregate scale of the building's simulated occupancy.
        occ=sum(occupancy(ts,j+1,c["residents"],c["weekend"]) for j in range(c["floors"]))
        solar_gain=sum(solar*c["wwr"]*c["area"]*.55*sf for sf in fp["solar_factor"])
        people_gain=occ*c["person_gain"]
        appliance_gain=c["appliance_gain"]*c["area"]*c["floors"]
        gains=(people_gain+appliance_gain+solar_gain)/1000
        lag1=hist[-1]; lag4=hist[-4] if len(hist)>=4 else lag1
        row={"outdoor_temperature_C":tout,"solar_irradiance_W_m2":solar,
             "wind_speed_m_s":float(last.wind_speed_m_s),"occupancy_count":occ,
             "indoor_temperature_C":indoor,"heating_setpoint_C":float(last.heating_setpoint_C),
             "internal_gains_kW":gains,"hour_sin":math.sin(2*math.pi*h/24),
             "hour_cos":math.cos(2*math.pi*h/24),
             "dow_sin":math.sin(2*math.pi*ts.dayofweek/7),
             "dow_cos":math.cos(2*math.pi*ts.dayofweek/7),
             "demand_lag_1":lag1,"demand_lag_4":lag4,"indoor_lag_1":indoor,
             "demand_roll_4":float(np.mean(hist[-4:])),
             "demand_roll_24":float(np.mean(hist[-int(24*60/dt):])),
             "temp_delta":float(last.heating_setpoint_C)-indoor}
        q=max(0,float(model.predict(pd.DataFrame([row])[FEATURES])[0]))
        rows.append({"timestamp":ts,"forecast_heating_demand_kWth":q,
                     "forecast_heating_energy_kWhth":q*dt_h,
                     "forecast_outdoor_temperature_C":tout,
                     "forecast_occupancy":occ,"forecast_indoor_temperature_C":indoor})
        hist.append(q)
        qloss=float(last.ua_total_W_K)*(indoor-tout)/1000
        indoor=float(np.clip(indoor+(q+gains-qloss)*dt_h/
                             max(float(last.thermal_capacity_kWh_K),.1),5,35))
    return model,pd.DataFrame(rows),test,pred,scores

with st.sidebar:
    st.header("Urban climate")
    climate=st.selectbox("Climate archetype",["Pune-like urban (synthetic)","Hot-humid urban","Cold urban","Custom"])
    presets={"Pune-like urban (synthetic)":(24,7,650,.28,1.8),
             "Hot-humid urban":(29,5,700,.42,1.2),
             "Cold urban":(7,8,500,.35,1.0),
             "Custom":(22,6,600,.3,1.0)}
    mt,sw,sp,cl,uhi=presets[climate]
    if climate=="Custom":
        mt=st.slider("Mean outdoor temperature °C",-5.,40.,22.)
        sw=st.slider("Daily swing °C",1.,15.,6.)
        sp=st.slider("Peak solar W/m²",100,1000,600)
        cl=st.slider("Cloudiness",0.,.9,.3)
        uhi=st.slider("Urban heat-island uplift °C",0.,5.,1.)
    st.header("Building geometry")
    floors=st.slider("Storeys",2,30,8)
    area=st.slider("Floor plate area m²",80,1500,450,10)
    days=st.slider("Historical simulation days",14,90,30)
    dt=st.select_slider("Time interval (minutes)",[15,30,60],value=15)
    seed=st.number_input("Random seed",0,99999,17)
    st.header("Envelope")
    uwall=st.slider("Wall U-value W/m²K",.2,3.,.8,.05)
    uroof=st.slider("Roof U-value W/m²K",.2,4.,1.,.05)
    uwin=st.slider("Window U-value W/m²K",.8,5.,2.6,.1)
    wwr=st.slider("Window-to-wall ratio",.05,.65,.25,.01)
    ach=st.slider("Infiltration / ventilation ACH",.1,2.,.5,.05)
    cap=st.slider("Thermal capacity per floor kWh/K",2.,80.,18.,1.)
    shade=st.slider("Urban shading factor",.1,1.,.65,.05)
    orientation=st.slider("Solar orientation multiplier",.5,1.5,1.,.05)
    st.header("Occupancy and heating")
    residents=st.slider("Residents per floor",1,30,8)
    weekend=st.slider("Weekend daytime occupancy fraction",.1,1.,.55)
    person_gain=st.slider("Sensible gain per person W",50,150,75)
    appliance_gain=st.slider("Appliance / lighting gains W/m²",1.,15.,4.,.5)
    setpoint=st.slider("Heating setpoint °C",16.,24.,20.,.5)
    max_heat=st.slider("Heating capacity per floor kWth",2.,40.,15.,.5)
    kp=st.slider("Thermostat correction W/K",50,1500,350,25)
    inter=st.slider("Inter-floor coupling W/K",0,500,120,10)
    st.header("Forecast")
    horizon=st.selectbox("Forecast horizon hours",[6,12,24,48],index=2)
    model_kind=st.selectbox("ML algorithm",["HistGradientBoosting","Random Forest"])

config={"floors":floors,"area":area,"days":days,"dt":dt,"seed":int(seed),
"mean_t":mt,"swing":sw,"solar_peak":sp,"cloud":cl,"uhi":uhi,
"uwall":uwall,"uroof":uroof,"uwin":uwin,"wwr":wwr,"ach":ach,"cap":cap,
"shade":shade,"orientation":orientation,"residents":residents,"weekend":weekend,
"person_gain":person_gain,"appliance_gain":appliance_gain,"setpoint":setpoint,
"max_heat":max_heat,"kp":kp,"inter":inter}

if st.button("Generate simulation + train forecast",type="primary",use_container_width=True):
    with st.spinner("Simulating building and fitting the forecasting model..."):
        floor,agg,wx,fp=simulate(config)
        model,forecast,test,pred,scores=fit_model(agg,dt,horizon,model_kind,config,fp)
        st.session_state["result"]={"floor":floor,"agg":agg,"wx":wx,"fp":fp,
                                    "forecast":forecast,"test":test,"pred":pred,
                                    "scores":scores,"config":config}

if "result" not in st.session_state:
    st.info("Set the building parameters in the sidebar and click Generate. The Pune-like climate is synthetic, not live weather.")
    st.stop()

r=st.session_state["result"]
floor,agg,forecast=r["floor"],r["agg"],r["forecast"]
st.caption(f"Showing last generated run: {r['config']['floors']} floors, {r['config']['area']} m²/floor, {r['config']['days']} days, {r['config']['dt']}-minute steps.")
a,b,c,d=st.columns(4)
a.metric("Peak demand",f"{agg.heating_demand_kWth.max():.1f} kWth")
b.metric("Mean demand",f"{agg.heating_demand_kWth.mean():.1f} kWth")
c.metric("Total heat energy",f"{agg.heating_energy_kWhth.sum():,.0f} kWhth")
d.metric("Forecast energy",f"{forecast.forecast_heating_energy_kWhth.sum():,.1f} kWhth")

t1,t2,t3,t4,t5=st.tabs(["Building overview","Thermal demand","Forecast","Validation","CSV export"])
with t1:
    left,right=st.columns(2)
    with left:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=agg.timestamp,y=agg.outdoor_temperature_C,name="Outdoor"))
        fig.add_trace(go.Scatter(x=agg.timestamp,y=agg.indoor_temperature_C,name="Indoor mean"))
        fig.add_trace(go.Scatter(x=agg.timestamp,y=agg.heating_setpoint_C,name="Setpoint",line=dict(dash="dash")))
        fig.update_layout(yaxis_title="Temperature °C",height=360)
        st.plotly_chart(fig,use_container_width=True)
    with right:
        fs=floor.groupby("floor",as_index=False).heating_demand_kWth.mean()
        fig=px.bar(fs,x="floor",y="heating_demand_kWth",labels={"floor":"Storey","heating_demand_kWth":"Mean demand kWth"})
        fig.update_layout(height=360)
        st.plotly_chart(fig,use_container_width=True)
    st.subheader("Floor-specific thermal parameters")
    st.dataframe(r["fp"].round(2),use_container_width=True,hide_index=True)
with t2:
    fig=px.line(agg,x="timestamp",y="heating_demand_kWth",labels={"heating_demand_kWth":"Building heating demand kWth"})
    st.plotly_chart(fig,use_container_width=True)
    fig=px.box(floor,x="floor",y="heating_demand_kWth",labels={"floor":"Storey","heating_demand_kWth":"Demand kWth"})
    st.plotly_chart(fig,use_container_width=True)
with t3:
    st.warning("Future weather and occupancy are synthetic demonstration inputs. Replace them with weather forecasts and building schedules before treating this as an operational forecast.")
    fig=go.Figure()
    hist=agg.tail(int(48*60/dt))
    fig.add_trace(go.Scatter(x=hist.timestamp,y=hist.heating_demand_kWth,name="Historical simulated demand"))
    fig.add_trace(go.Scatter(x=forecast.timestamp,y=forecast.forecast_heating_demand_kWth,name="ML forecast",line=dict(dash="dash")))
    fig.update_layout(yaxis_title="Heating demand kWth",height=420)
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(forecast.round(3),use_container_width=True,hide_index=True)
with t4:
    s=r["scores"]
    m1,m2,m3=st.columns(3)
    m1.metric("MAE",f"{s['MAE (kWth)']:.3f} kWth")
    m2.metric("RMSE",f"{s['RMSE (kWth)']:.3f} kWth")
    m3.metric("R²",f"{s['R²']:.3f}")
    q=r["test"].copy(); q["prediction_kWth"]=r["pred"]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=q.timestamp,y=q.heating_demand_kWth,name="Actual simulated"))
    fig.add_trace(go.Scatter(x=q.timestamp,y=q.prediction_kWth,name="Predicted",line=dict(dash="dash")))
    fig.update_layout(yaxis_title="Heating demand kWth",height=400)
    st.plotly_chart(fig,use_container_width=True)
    st.warning("The chronological 80/20 hold-out tests how well ML reproduces this simulator's labels. It does not prove real-building accuracy. Compare with a persistence baseline and calibrate against BMS/meter data.")
with t5:
    def csvdata(df): return df.to_csv(index=False).encode("utf-8")
    st.download_button("Building aggregate CSV",csvdata(agg),"building_aggregate.csv","text/csv")
    st.download_button("Floor-level CSV",csvdata(floor),"building_floor_level.csv","text/csv")
    st.download_button("Synthetic weather CSV",csvdata(r["wx"]),"synthetic_weather.csv","text/csv")
    st.download_button("Forecast CSV",csvdata(forecast),"thermal_demand_forecast.csv","text/csv")
    st.caption("Power = kWth. Energy per interval = kWth × interval hours = kWhth.")

with st.expander("Equations and model boundary"):
    st.markdown(r"""
For floor \(i\), the lumped thermal balance is:

\[
C_i \frac{dT_i}{dt}=Q_{heat,i}+Q_{internal,i}+Q_{solar,i}
-UA_i(T_i-T_{out})+\sum_j K_{ij}(T_j-T_i)
\]

The envelope/ventilation conductance is approximated by:

\[
UA_i=\sum_k U_kA_k+0.33\,ACH\,V
\]

The synthetic heating target is generated by a simplified thermostat:

\[
Q_{heat,i}=\mathrm{clip}\left(UA_i(T_{set}-T_{out})-Q_{gains,i}
+K_p(T_{set}-T_i),0,Q_{max}\right)
\]

This is one well-mixed zone per floor. It does not resolve individual apartments, rooms, thermal bridges, HVAC distribution, detailed glazing, or measured occupant behaviour.
""")
