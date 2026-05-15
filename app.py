import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import math

# ==========================================
# 1. PHYSICS & DYNAMIC FUEL MODEL
# ==========================================

class CarDatabase:
    @staticmethod
    def get_cars():
        return {
            "Toyota Corolla": {"weight": 1380, "cd": 0.29, "A": 2.1, "idle_p": 0.8, "eff": 0.25, "is_ev": False},
            "Honda Civic": {"weight": 1300, "cd": 0.27, "A": 2.1, "idle_p": 0.8, "eff": 0.26, "is_ev": False},
            "Hyundai Elantra": {"weight": 1350, "cd": 0.28, "A": 2.1, "idle_p": 0.85, "eff": 0.25, "is_ev": False},
            "BMW 320i": {"weight": 1520, "cd": 0.23, "A": 2.2, "idle_p": 1.1, "eff": 0.28, "is_ev": False},
            "Mercedes C180": {"weight": 1500, "cd": 0.24, "A": 2.2, "idle_p": 1.1, "eff": 0.27, "is_ev": False},
            "Ford Focus": {"weight": 1360, "cd": 0.29, "A": 2.1, "idle_p": 0.9, "eff": 0.25, "is_ev": False},
            "Nissan Sunny": {"weight": 1230, "cd": 0.31, "A": 2.0, "idle_p": 0.8, "eff": 0.24, "is_ev": False},
            "Tesla Model 3": {"weight": 1610, "cd": 0.23, "A": 2.2, "idle_p": 1.5, "eff": 0.90, "is_ev": True}
        }

def dynamic_fuel_model(v_kmh, car_params, road_type="Highway", weather="Clear"):
    v_kmh = max(v_kmh, 1.0)
    v_ms = v_kmh / 3.6
    g = 9.81
    rho = 1.225
    energy_density_gas = 34.2e6
    m = car_params["weight"]
    cd = car_params["cd"]
    A = car_params["A"]
    eff = car_params["eff"]
    is_ev = car_params["is_ev"]
    idle_power = car_params["idle_p"]
    crr = 0.015
    if road_type == "City": crr *= 1.5
    if road_type == "Mountain": crr *= 2.0
    aero_mod = 1.0
    if weather == "Rain": crr *= 1.2
    if weather == "Windy": aero_mod *= 1.3
    p_rr = crr * m * g * v_ms
    p_aero = 0.5 * rho * cd * A * aero_mod * (v_ms ** 3)
    p_wheels = p_rr + p_aero
    
    if is_ev:
        p_idle = (idle_power + 1.5) * 1000
        p_total_motor = (p_wheels / eff) + p_idle
        energy_per_km = p_total_motor / v_ms
        consumption = (energy_per_km * 100000) / 3.6e6
    else:
        load_penalty = 3.5 
        idle_consumption = ((idle_power * load_penalty) / v_kmh) * 100
        fuel_flow_ls = (p_wheels / eff) / energy_density_gas
        fuel_per_m = fuel_flow_ls / v_ms
        driving_consumption = fuel_per_m * 100000
        consumption = idle_consumption + driving_consumption
    return consumption

# ==========================================
# 2. OPTIMIZATION ALGORITHM (GSS)
# ==========================================

def golden_section_search(f, a, b, tol=1e-5):
    golden_ratio = (np.sqrt(5) - 1) / 2
    x1 = b - golden_ratio * (b - a)
    x2 = a + golden_ratio * (b - a)
    f1 = f(x1)
    f2 = f(x2)
    history = []
    iterations = 0
    while abs(b - a) > tol:
        history.append((a, b, x1, x2, f1, f2))
        iterations += 1
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - golden_ratio * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + golden_ratio * (b - a)
            f2 = f(x2)
    optimal_x = (a + b) / 2
    history.append((a, b, optimal_x, optimal_x, f(optimal_x), f(optimal_x)))
    return optimal_x, f(optimal_x), iterations, history

# ==========================================
# 3. UI LOGIC & STREAMLIT APP
# ==========================================

st.set_page_config(page_title="GoldenTune AI: Optimization Suite", page_icon="🧮", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .stApp { 
        background-image: url('https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?q=80&w=3540&auto=format&fit=crop');
        background-size: cover; 
        background-position: center;
        background-attachment: fixed;
        color: #f8fafc; 
    }
    
    .stApp::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(15, 23, 42, 0.35);
        pointer-events: none;
        z-index: 0;
    }
    
    .block-container {
        position: relative;
        z-index: 1;
        padding-top: 1rem !important;
    }
    
    /* Structural & Stylistic rules */
    .glass-card {
        background: rgba(15, 23, 42, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 30px; 
        padding: 40px; 
        margin-bottom: 40px; 
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        transition: border-color 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(196, 217, 148, 0.4);
    }
    
    .glass-card-highlight {
        background: linear-gradient(145deg, rgba(196, 217, 148, 0.1) 0%, rgba(15, 23, 42, 0.4) 100%);
        border: 1px solid rgba(196, 217, 148, 0.5);
    }

    h1, h2 {
        font-family: 'Inter', sans-serif;
        color: #fff;
        font-weight: 500;
        letter-spacing: -1px;
    }
    
    /* Specific Typography Hierarchy */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: #cbd5e1;
        margin-bottom: 15px;
        text-align: center;
        font-weight: 500;
    }
    
    .card-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 500;
        color: #f8fafc;
        margin-bottom: 5px;
        line-height: 1.2;
    }
    
    .sub-tagline {
        font-size: 1rem;
        color: #cbd5e1;
        margin-bottom: 30px;
        display: block;
        font-weight: 300;
    }

    .metric-value { font-size: 3.5rem; font-family: 'Inter', sans-serif; font-weight: 500; color: #c4d994; line-height: 1; margin-top: 20px; }
    .metric-value-bad { font-size: 3.5rem; font-family: 'Inter', sans-serif; font-weight: 500; color: #ef4444; line-height: 1; margin-top: 20px; }
    .metric-label { color: #cbd5e1; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 500; }
    
    /* Pill-shaped buttons with hover effects */
    div[data-testid="stButton"] > button, .stButton>button { 
        border-radius: 50px !important; 
        font-weight: 500; 
        font-size: 1.1rem;
        padding: 12px 32px;
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
        background: rgba(0,0,0,0.3);
        color: #fff;
    }
    
    div[data-testid="stButton"] > button:hover, .stButton>button:hover {
        border-color: #c4d994;
        background: rgba(196, 217, 148, 0.1);
        transform: translateY(-2px);
    }
    
    /* Primary buttons */
    div[data-testid="stButton"] > button[kind="primary"], .stButton>button[kind="primary"] {
        background: #c4d994 !important;
        color: #1e293b !important;
        border: none !important;
        font-weight: 500 !important;
    }
    
    div[data-testid="stButton"] > button[kind="primary"]:hover, .stButton>button[kind="primary"]:hover {
        background: #d4e8a3 !important;
        transform: translateY(-2px);
    }

    .math-step { font-size: 1.1rem; margin-bottom: 10px; color: #cbd5e1; }
    
    /* Feature lists */
    .feature-list { list-style: none; padding-left: 0; margin-top: 20px; }
    .feature-list li { position: relative; padding-left: 35px; margin-bottom: 20px; font-weight: 500; font-size: 1.1rem; color: #e2e8f0; }
    .feature-list li::before { content: ''; position: absolute; left: 0; top: 8px; width: 14px; height: 14px; background-color: #c4d994; border-radius: 50%; }
    .feature-desc { font-size: 0.9rem; color: #cbd5e1; font-weight: 300; display: block; margin-top: 5px; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# State initialization
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Menu"

# --- MAIN MENU PAGE ---
if st.session_state.current_page == "Menu":
    st.markdown("""
    <style>
    /* Hide top padding and generic Streamlit header on Menu page */
    .block-container {
        padding-top: 0rem !important;
        max-width: 100% !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    header { visibility: hidden; }
    
    /* Top Navigation Bar */
    .lumena-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 30px 60px;
        background: transparent;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    .lumena-logo {
        font-size: 1.2rem;
        font-weight: 400;
        letter-spacing: 3px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .lumena-logo span {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 2px solid #f8fafc;
        border-radius: 50%;
        border-right-color: transparent;
        border-bottom-color: transparent;
        transform: rotate(45deg);
    }
    .lumena-links {
        display: flex;
        gap: 40px;
        font-size: 0.95rem;
        color: #cbd5e1;
        font-weight: 400;
    }
    .lumena-links span { cursor: pointer; transition: color 0.3s; }
    .lumena-links span:hover { color: #fff; }
    .lumena-auth {
        display: flex;
        gap: 30px;
        align-items: center;
        font-size: 0.95rem;
        font-weight: 400;
    }
    .lumena-auth .sign-in { cursor: pointer; color: #cbd5e1; transition: color 0.3s; }
    .lumena-auth .sign-in:hover { color: #fff; }
    .lumena-auth .get-started {
        border: 1px solid rgba(255,255,255,0.4);
        border-radius: 30px;
        padding: 10px 24px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .lumena-auth .get-started:hover { background: rgba(255,255,255,0.1); color: #fff; }

    /* Hero Section */
    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 60vh;
        text-align: center;
        padding: 0 20px;
        margin-top: 20px;
    }
    .badge-container {
        display: flex;
        justify-content: center;
        margin-bottom: 30px;
    }
    .badge {
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 30px;
        padding: 8px 20px;
        font-size: 0.75rem;
        letter-spacing: 1.5px;
        color: #e2e8f0;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        text-transform: uppercase;
    }
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 5rem;
        font-weight: 400;
        color: #fff;
        margin-bottom: 25px;
        line-height: 1.1;
        letter-spacing: -2px;
    }
    .hero-title .highlight {
        color: #c4d994; /* Pale soft green from Lumena */
        font-weight: 500;
    }
    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.25rem;
        max-width: 650px;
        line-height: 1.6;
        margin-bottom: 40px;
        font-weight: 300;
    }
    
    /* Override Streamlit Buttons for Hero */
    .stButton>button {
        background: #c4d994 !important;
        color: #1e293b !important;
        border: none !important;
        border-radius: 40px !important;
        padding: 16px 32px !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        box-shadow: none !important;
        min-height: unset !important;
        height: auto !important;
        display: inline-flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin: 0 auto;
        width: 100% !important;
    }
    .stButton>button p {
        font-family: 'Inter', sans-serif !important;
        font-size: 1.05rem !important;
        font-weight: 500 !important;
        color: #1e293b !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        background: #d4e8a3 !important;
        box-shadow: 0 10px 25px rgba(196, 217, 148, 0.2) !important;
    }
    
    .secondary-btn>div>div>button {
        background: rgba(255,255,255,0.05) !important;
        color: #fff !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        backdrop-filter: blur(10px);
    }
    .secondary-btn>div>div>button p {
        color: #fff !important;
    }
    .secondary-btn>div>div>button:hover {
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.5) !important;
    }

    /* Footer */
    .trusted-by {
        text-align: center;
        color: #cbd5e1;
        font-size: 0.75rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 60px;
        font-weight: 600;
        padding-bottom: 40px;
    }
    .trusted-logos {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 40px;
        margin-top: 25px;
        opacity: 0.9;
    }
    .trusted-logos span {
        font-weight: 600;
        font-size: 1.3rem;
        color: #fff;
        font-family: 'Inter', sans-serif;
    }
    </style>

    <div class="lumena-nav">
        <div class="lumena-logo"><span></span> GOLDENTUNE</div>
        <div class="lumena-links">
            <span>Features ⌄</span>
            <span>Gallery</span>
            <span>Pricing</span>
            <span>API ⌄</span>
            <span>Resources ⌄</span>
        </div>
        <div class="lumena-auth">
            <span class="sign-in">Sign in</span>
            <span class="get-started">Get Started</span>
        </div>
    </div>

    <div class="hero-container">
        <div class="badge-container">
            <div class="badge">✦ OPTIMIZATION SUITE</div>
        </div>
        <div class="hero-title">Optimization becomes <span class="highlight">intuitive.</span></div>
        <div class="hero-subtitle">
            Simulate physics-based fuel consumption and explore Golden Section Search algorithms.<br>Fast, precise, and built for education.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='max-width: 600px; margin: 0 auto;'>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns([1, 0.5, 4, 4, 1])
    
    with col3:
        if st.button("Start GSS Solver ➔", use_container_width=True):
            st.session_state.animating_to = "GSS"
            st.session_state.current_page = "Animation"
            st.rerun()
            
    with col4:
        st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
        if st.button("Fuel System ➔", use_container_width=True):
            st.session_state.animating_to = "Fuel"
            st.session_state.current_page = "Animation"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="trusted-by">
        POWERED BY ALGORITHMS FOR
        <div class="trusted-logos">
            <span>OpenAI</span>
            <span style="font-weight: 400;">Midjourney</span>
            <span style="font-family: serif; font-style: italic; font-weight: bold; font-size: 1.5rem;">Adobe</span>
            <span style="letter-spacing: -1px;">runway</span>
            <span style="font-family: 'Brush Script MT', cursive; font-size: 1.8rem;">Canva</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- ANIMATION PAGE ---
elif st.session_state.current_page == "Animation":
    st.markdown("""
    <style>
    .road-container { margin-top: 20vh; }
    .road {
      width: 100%; height: 120px; background-color: #1e293b;
      position: relative; overflow: hidden;
      border-top: 5px solid #475569; border-bottom: 5px solid #475569;
      border-radius: 10px;
    }
    .road::before {
      content: ""; position: absolute; top: 50%; left: 0;
      width: 200%; height: 6px;
      background-image: linear-gradient(to right, #e2e8f0 50%, transparent 50%);
      background-size: 80px 100%;
      animation: move-road 0.4s linear infinite;
    }
    .car {
      position: absolute; top: 10px; left: -150px; font-size: 70px;
      animation: drive-car 2.5s cubic-bezier(0.4, 0.0, 0.2, 1) forwards;
    }
    @keyframes move-road {
      0% { transform: translateX(0); }
      100% { transform: translateX(-80px); }
    }
    @keyframes drive-car {
      0% { left: -150px; transform: scale(1) rotate(0deg); }
      10% { transform: scale(1.1) rotate(-5deg); }
      80% { transform: scale(1.1) rotate(2deg); }
      100% { left: 110%; transform: scale(1) rotate(0deg); }
    }
    </style>
    <div class="road-container">
        <h2 style="text-align: center; color: #38bdf8; margin-bottom: 30px;">Transporting to Workspace...</h2>
        <div class="road">
          <div class="car">🏎️</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(2.5)
    st.session_state.current_page = st.session_state.animating_to
    st.rerun()

# --- FUEL OPTIMIZATION PAGE ---
elif st.session_state.current_page == "Fuel":
    if st.button("⬅️ Back to Main Menu"):
        st.session_state.current_page = "Menu"
        st.rerun()
        
    st.markdown("<h1 style='text-align: center; font-size: 3rem;'>Fuel Optimization System</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #94a3b8;'>Physics-Based Engine Simulator</h3>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

    if 'fuel_optimized' not in st.session_state:
        st.session_state.fuel_optimized = False

    st.markdown("""
    <div class="glass-card">
        <div class="card-title">Configuration Panel</div>
        <span class="sub-tagline">Set your vehicle physics and trip parameters below</span>
    </div>
    """, unsafe_allow_html=True)
    
    col_input1, col_input2, col_input3 = st.columns(3)
    
    with col_input1:
        st.markdown("**🚘 Vehicle & Physics**")
        cars_db = CarDatabase.get_cars()
        selected_car = st.selectbox("Select Car Model", list(cars_db.keys()))
        car_params = cars_db[selected_car]
        is_ev = car_params['is_ev']
        fuel_unit = "kWh" if is_ev else "L"
        curr_str = "$"
        road_type = st.selectbox("Road Type", ["Highway", "City", "Mountain"])
        weather = st.selectbox("Weather Conditions", ["Clear", "Rain", "Windy"])
        
    with col_input2:
        st.markdown("**🗺️ Trip Parameters**")
        trip_distance = st.number_input("Total Trip Distance (km)", min_value=1.0, value=300.0, step=10.0)
        fuel_price = st.number_input(f"Energy Price ({curr_str}/{fuel_unit})", min_value=0.01, value=0.15 if is_ev else 1.50, step=0.05)
        
    with col_input3:
        st.markdown("**🚥 Constraints**")
        speed_min = st.number_input("Min km/h", 10.0, value=20.0, step=5.0)
        speed_max = st.number_input("Max km/h", 60.0, value=150.0, step=5.0)
        manual_speed = st.slider("Your Manual Driving Speed", float(speed_min), float(speed_max), float((speed_min+speed_max)/2))
        
    st.markdown("<br>", unsafe_allow_html=True)
    run_opt = st.button("🚀 Run Fuel Optimization", type="primary", use_container_width=True)

    def get_trip_fuel(speed):
        cons_100 = dynamic_fuel_model(speed, car_params, road_type, weather)
        return (cons_100 / 100.0) * trip_distance

    manual_total_fuel = get_trip_fuel(manual_speed)

    if run_opt:
        if speed_min >= speed_max:
            st.error("Min speed must be less than Max speed!")
        else:
            with st.spinner("Solving physical models & minimizing via GSS..."):
                time.sleep(0.5)
                opt_speed, opt_total_fuel, iterations, history = golden_section_search(get_trip_fuel, speed_min, speed_max)
                st.session_state.opt_speed = opt_speed
                st.session_state.opt_fuel = opt_total_fuel
                st.session_state.iterations = iterations
                st.session_state.history = history
                st.session_state.manual_speed = manual_speed
                st.session_state.manual_fuel = manual_total_fuel
                st.session_state.fuel_optimized = True
                st.session_state.car = selected_car
                st.session_state.road = road_type
                
                if abs(opt_speed - speed_min) < 0.5:
                    st.warning(f"⚠️ The mathematical optimal speed is actually lower than your Minimum Limit ({speed_min} km/h). The algorithm returned the lowest allowed speed.")
                elif abs(opt_speed - speed_max) < 0.5:
                    st.warning(f"⚠️ The mathematical optimal speed is higher than your Maximum Limit ({speed_max} km/h).")

    if st.session_state.fuel_optimized:
        m_speed = st.session_state.manual_speed
        m_fuel = st.session_state.manual_fuel
        o_speed = st.session_state.opt_speed
        o_fuel = st.session_state.opt_fuel
        fuel_saved = max(0, m_fuel - o_fuel)
        saving_pct = (fuel_saved / m_fuel) * 100 if m_fuel > 0 else 0
        
        st.markdown("<div class='section-header'>Optimization Results</div>", unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(f'<div class="glass-card glass-card-highlight"><div class="metric-label">Optimal Speed</div><div class="metric-value">{o_speed:.1f} <span style="font-size: 1.5rem; color: #94a3b8;">km/h</span></div></div>', unsafe_allow_html=True)
        with r2:
            st.markdown(f'<div class="glass-card"><div class="metric-label">Total Energy Used</div><div class="metric-value">{o_fuel:.1f} <span style="font-size: 1.5rem; color: #94a3b8;">{fuel_unit}</span></div></div>', unsafe_allow_html=True)
        with r3:
            st.markdown(f'<div class="glass-card"><div class="metric-label">Energy Saved</div><div class="metric-value">{fuel_saved:.1f} {fuel_unit} <span style="font-size: 1rem; color: #94a3b8;"><br>({saving_pct:.1f}%)</span></div></div>', unsafe_allow_html=True)
        with r4:
            st.markdown(f'<div class="glass-card"><div class="metric-label">Money Saved</div><div class="metric-value">${(fuel_saved * fuel_price):.2f}</div></div>', unsafe_allow_html=True)

        v_vals = np.linspace(speed_min, speed_max, 300)
        f_vals = [get_trip_fuel(v) for v in v_vals]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=v_vals, y=f_vals, mode='lines', name='Physics Energy Curve', line=dict(color='#3b82f6', width=3)))
        fig.add_trace(go.Scatter(x=[m_speed], y=[m_fuel], mode='markers+text', name='Manual Speed', marker=dict(color='#ef4444', size=12, symbol='x'), text=[f'Manual ({m_fuel:.1f})'], textposition="top center"))
        fig.add_trace(go.Scatter(x=[o_speed], y=[o_fuel], mode='markers+text', name='Absolute Minimum', marker=dict(color='#10b981', size=14, symbol='star'), text=[f'Optimal ({o_fuel:.1f})'], textposition="bottom center"))
        
        hist = st.session_state.history
        y_base = min(f_vals) - (max(f_vals) - min(f_vals)) * 0.1
        y_step = (max(f_vals) - min(f_vals)) * 0.02
        for i in range(min(15, len(hist))):
            a_val, b_val, _, _, _, _ = hist[i]
            y_pos = y_base - (i * y_step)
            fig.add_trace(go.Scatter(x=[a_val, b_val], y=[y_pos, y_pos], mode='lines+markers', name=f'Iter {i+1} Bracket', line=dict(color='rgba(255,255,255,0.4)', width=2), marker=dict(symbol='line-ns-open', color='rgba(255,255,255,0.6)', size=8), showlegend=(i==0)))
        
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#f8fafc'), height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 40px;">
            <h2 style="color: #94a3b8;">👆 Configure parameters above and Run the Optimization!</h2>
        </div>
        """, unsafe_allow_html=True)

# --- CUSTOM GSS SOLVER PAGE ---
elif st.session_state.current_page == "GSS":
    if st.button("⬅️ Back to Main Menu"):
        st.session_state.current_page = "Menu"
        st.rerun()

    st.markdown("<h1 style='text-align: center; font-size: 3rem;'>Custom GSS Solver</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #94a3b8;'>Step-by-Step Educational Calculator</h3>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div class="card-title">GSS Parameters</div>
        <span class="sub-tagline">Define your mathematical function and bounds</span>
    </div>
    """, unsafe_allow_html=True)
    
    col_gss1, col_gss2 = st.columns(2)
    
    with col_gss1:
        target_mode = st.radio("Optimization Goal", ["Maximize", "Minimize"], horizontal=True)
        eq_str = st.text_input("Enter f(x)", value="4*sin(x)*(1+cos(x))")
        st.markdown("<span style='color:#94a3b8; font-size:0.9rem;'>*Use python math (e.g. `sin(x)`, `x**2`, `exp(x)`)*</span>", unsafe_allow_html=True)
        
    with col_gss2:
        c1, c2 = st.columns(2)
        with c1: xl_input = st.number_input("Lower Bound (Xl)", value=0.0, step=0.1)
        with c2: xu_input = st.number_input("Upper Bound (Xu)", value=math.pi/2, step=0.1)
        num_iters = st.number_input("Number of Iterations", min_value=1, max_value=50, value=2, step=1)
        
    st.markdown("<br>", unsafe_allow_html=True)
    run_gss = st.button("🚀 Calculate Step-by-Step", type="primary", use_container_width=True)

    def safe_eval(expr, x_val):
        allowed = {k: v for k, v in np.__dict__.items() if not k.startswith("__")}
        allowed.update({"sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "pi": np.pi, "e": np.e})
        allowed["x"] = x_val
        expr = expr.replace("^", "**")
        try:
            return eval(expr, {"__builtins__": {}}, allowed)
        except Exception as e:
            return None

    if run_gss:
        test_val = safe_eval(eq_str, xl_input)
        if test_val is None:
            st.error("Error evaluating function. Please check your syntax.")
        else:
            xl = xl_input
            xu = xu_input
            R = (math.sqrt(5) - 1) / 2
            
            st.markdown(f"<br><h3 style='text-align:center;'>Solving for <strong>{target_mode}</strong> of $f(x) = {eq_str}$ over $[{xl:.4f}, {xu:.4f}]$</h3><br>", unsafe_allow_html=True)
            
            for i in range(1, num_iters + 1):
                st.markdown(f"""
                <div class="glass-card">
                    <div class="card-title">Iteration {i}</div>
                    <span class="sub-tagline">Calculating optimal bracket points</span>
                """, unsafe_allow_html=True)
                
                d = R * (xu - xl)
                x1 = xl + d
                x2 = xu - d
                
                f1 = safe_eval(eq_str, x1)
                f2 = safe_eval(eq_str, x2)
                
                col_math1, col_math2 = st.columns(2)
                with col_math1:
                    st.latex(f"X_L = {xl:.4f}, \\quad X_U = {xu:.4f}")
                    st.latex(f"d = R \\times (X_U - X_L) = 0.61803 \\times ({xu:.4f} - {xl:.4f}) = {d:.4f}")
                    st.latex(f"X_1 = X_L + d = {x1:.4f}")
                    st.latex(f"X_2 = X_U - d = {x2:.4f}")
                
                with col_math2:
                    comp_sign = ">" if f1 > f2 else ("<" if f1 < f2 else "=")
                    st.latex(f"f(X_1) = {f1:.4f} \\quad {comp_sign} \\quad f(X_2) = {f2:.4f}")
                    
                    if target_mode == "Maximize":
                        discard_left = (f1 > f2)
                    else: 
                        discard_left = (f1 < f2)
                    
                    st.markdown("**Therefore:**")
                    if discard_left:
                        new_xl = x2
                        new_xu = xu
                        st.latex(f"X_U = X_U = {new_xu:.4f}")
                        st.latex(f"X_L = X_2 = {new_xl:.4f}")
                        st.latex(f"\\text{{New }} X_1 = {new_xl:.4f} + 0.61803 \\times ({new_xu:.4f} - {new_xl:.4f}) = {(new_xl + R*(new_xu-new_xl)):.4f}")
                    else:
                        new_xl = xl
                        new_xu = x1
                        st.latex(f"X_L = X_L = {new_xl:.4f}")
                        st.latex(f"X_U = X_1 = {new_xu:.4f}")
                        st.latex(f"\\text{{New }} X_2 = {new_xu:.4f} - 0.61803 \\times ({new_xu:.4f} - {new_xl:.4f}) = {(new_xu - R*(new_xu-new_xl)):.4f}")
                
                x_vals = np.linspace(xl_input, xu_input, 400)
                y_vals = [safe_eval(eq_str, xv) for xv in x_vals]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='f(x)', line=dict(color='#3b82f6')))
                
                fig.add_trace(go.Scatter(
                    x=[xl, x2, x1, xu],
                    y=[safe_eval(eq_str, xl), f2, f1, safe_eval(eq_str, xu)],
                    mode='markers+text',
                    name='Points',
                    marker=dict(color=['#94a3b8', '#f59e0b', '#f59e0b', '#94a3b8'], size=12),
                    text=['X_L', 'X_2', 'X_1', 'X_U'],
                    textposition="top center",
                    textfont=dict(size=14, color='white')
                ))
                
                if discard_left:
                    fig.add_vrect(x0=xl, x1=x2, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Remove", annotation_position="top left")
                else:
                    fig.add_vrect(x0=x1, x1=xu, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Remove", annotation_position="top right")

                fig.update_layout(
                    title=f"Iteration {i} Segment Graph",
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#f8fafc'),
                    xaxis=dict(title="x", gridcolor='rgba(255,255,255,0.1)', range=[xl_input, xu_input]),
                    yaxis=dict(title="f(x)", gridcolor='rgba(255,255,255,0.1)'),
                    height=400, showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                xl = new_xl
                xu = new_xu
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 40px;">
            <h2 style="color: #94a3b8;">👆 Enter your function above and calculate the Iterations!</h2>
        </div>
        """, unsafe_allow_html=True)
