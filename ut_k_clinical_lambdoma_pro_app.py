import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="UT/K Simulator", layout="wide")

# --- 1. MOTORE MATEMATICO (UT/K + MUSICA) ---
def cents_diff(r1, r2):
    """Calcola la differenza in cents tra due rapporti di frequenza."""
    if r1 <= 0 or r2 <= 0: return 0
    return 1200 * math.log2(r1 / r2)

def get_note_name(ratio):
    """Trova il nome della nota (Do mobile) più vicina."""
    notes_map = {
        1.0: "Do", 16/15: "Si (min)", 9/8: "Re", 6/5: "Mib", 5/4: "Mi", 4/3: "Fa",
        45/32: "Fa#", 3/2: "Sol", 8/5: "Lab", 5/3: "La", 9/5: "Sib", 15/8: "Si", 2.0: "Do (8va)"
    }
    best_note = "?"
    min_diff = float('inf')
    for r, name in notes_map.items():
        if abs(ratio - r) < min_diff:
            min_diff = abs(ratio - r)
            best_note = name
    return best_note

def best_rational_in_octave(val, limit=16):
    """Trova la frazione p/q, l'ottava e l'errore in cents."""
    if val <= 0: return 1, 1, 0, 0, "N/A"
    norm = val
    octave = 0
    # Normalizza nell'ottava [1, 2)
    if norm < 1:
        while norm < 1: norm *= 2; octave -= 1
    elif norm >= 2:
        while norm >= 2: norm /= 2; octave += 1
            
    best_fract = (1, 1)
    best_error = float('inf')
    
    # Cerca la frazione migliore
    for q in range(1, limit + 1):
        for p in range(q, q * 2 + 1): 
            ratio = p / q
            diff = abs(math.log2(norm) - math.log2(ratio))
            if diff < best_error:
                best_error = diff
                best_fract = (p, q)
                
    p, q = best_fract
    cents_err = cents_diff(norm, p/q)
    return p, q, octave, cents_err, get_note_name(p/q)

def calcola_live(df):
    """Esegue tutti i calcoli UT/K sul dataframe."""
    d = df.copy()
    
    # Ratios ematici
    d['NLR'] = d['Neutrofili'] / d['Linfociti']
    d['PLR'] = d['Piastrine'] / d['Linfociti']
    d['LMR'] = d['Linfociti'] / d['Monociti']
    
    # Vettori Forma
    d['I1'] = d['NLR'] / d['MPV']
    d['I2'] = d['MPV'] / d['PLR']
    d['I3'] = d['PLR'] / d['LMR']
    
    # Indici Globali
    d['U_imm'] = (d['I1'] * d['I2'] * d['I3'])**(1/3)
    
    # Gestione Metabolica (Priorità Insulina, Fallback Glicemia)
    if 'Insulina' in d.columns and d['Insulina'].sum() > 0:
         d['U_met'] = d['Cortisolo'] / d['Insulina']
    else:
         d['U_met'] = (d['Cortisolo'] / 15) * (d['Glicemia'] / 90) * 5

    d['UT'] = d['U_met'] * d['U_imm']
    d['K'] = d['U_met'] / d['U_imm']
    
    # Ancoraggio a T0
    base_ut = d.iloc[0]['UT'] if len(d) > 0 and d.iloc[0]['UT'] > 0 else 1.0
    d['UT_Anchored'] = d['UT'] / base_ut
    
    # Calcolo Musicale
    harm_res = [best_rational_in_octave(x) for x in d['UT_Anchored']]
    d['p'] = [x[0] for x in harm_res]
    d['q'] = [x[1] for x in harm_res]
    d['Octave'] = [x[2] for x in harm_res]
    d['Cents'] = [x[3] for x in harm_res]
    d['Note'] = [x[4] for x in harm_res]
    
    return d

# --- 2. INIZIALIZZAZIONE DATI (SESSION STATE) ---
if 'df_data' not in st.session_state:
    # Dati iniziali (Template modificabile)
    st.session_state.df_data = pd.DataFrame({
        'Tempo': ['T0 (Base)', 'T1 (Oggi)'],
        'Neutrofili': [3.5, 8.5],
        'Linfociti': [2.1, 0.9],
        'Monociti': [0.5, 0.4],
        'Piastrine': [220, 160],
        'MPV': [9.5, 11.2],
        'Cortisolo': [12.0, 32.0],
        'Insulina': [8.0, 15.0],
        'Glicemia': [85, 140]
    })

# --- 3. INTERFACCIA GRAFICA ---
st.title("🎛️ UT/K Simulator: Medical & Musical")

# Layout: Colonna Sinistra (Controlli) - Colonna Destra (Grafici)
col_sx, col_dx = st.columns([1, 3])

with col_sx:
    st.header("🎚️ Live Tuning")
    st.info("Modifica i valori di T1 (Oggi) in tempo reale.")
    
    if len(st.session_state.df_data) > 1:
        # Recuperiamo i valori attuali
        curr = st.session_state.df_data.iloc[1]
        
        # SLIDERS CON CHIAVI UNIVOCHE (Evita errore Duplicate ID)
        n_neut = st.slider("Neutrofili", 0.1, 25.0, float(curr['Neutrofili']), 0.1, key="sl_neut")
        n_linf = st.slider("Linfociti", 0.1, 10.0, float(curr['Linfociti']), 0.1, key="sl_linf")
        n_plt = st.slider("Piastrine", 10.0, 900.0, float(curr['Piastrine']), 10.0, key="sl_plt")
        n_mpv = st.slider("MPV (fL)", 5.0, 16.0, float(curr['MPV']), 0.1, key="sl_mpv")
        n_cort = st.slider("Cortisolo", 1.0, 100.0, float(curr['Cortisolo']), 1.0, key="sl_cort")
        
        # Aggiorna il DataFrame se lo slider si muove
        st.session_state.df_data.at[1, 'Neutrofili'] = n_neut
        st.session_state.df_data.at[1, 'Linfociti'] = n_linf
        st.session_state.df_data.at[1, 'Piastrine'] = n_plt
        st.session_state.df_data.at[1, 'MPV'] = n_mpv
        st.session_state.df_data.at[1, 'Cortisolo'] = n_cort

with col_dx:
    # Esegui calcoli
    df_calc = calcola_live(st.session_state.df_data)
    last = df_calc.iloc[-1]
    
    # KPI CARDS
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("UT (Costo)", f"{last['UT']:.2f}", f"{last['UT_Anchored']:.2f}x vs T0")
    k2.metric("Guida K", f"{last['K']:.2f}", "Top-Down" if last['K']>1 else "Bottom-Up")
    k3.metric("Nota", f"{last['Note']}", f"Ottava {last['Octave']}")
    
    delta_col = "off" if abs(last['Cents']) > 20 else "normal"
    k4.metric("Intonazione", f"{last['Cents']:.1f} cents", delta_color=delta_col)

    # GRAFICI
    tab1, tab2, tab3 = st.tabs(["🧬 Radar Funzionale", "🎵 Lambdoma Armonico", "📝 Dati"])
    
    with tab1:
        # RADAR CHART
        fig = px.scatter(df_calc, x="U_met", y="U_imm", size="UT", color="K",
                         text="Tempo", hover_name="Tempo",
                         color_continuous_scale="RdBu_r", size_max=60, range_color=[0, 3])
        # Aggiungi croce sul punto attuale
        fig.add_hline(y=last['U_imm'], line_dash="dot", line_color="gray", opacity=0.5)
        fig.add_vline(x=last['U_met'], line_dash="dot", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig, use_container_width=True, key="plot_radar") # Key aggiunta
        
    with tab2:
        # LAMBDOMA CHART
        # Colore basato sull'errore (Cents)
        colors = ['#2ca02c' if abs(c)<15 else '#ff7f0e' if abs(c)<30 else '#d62728' for c in df_calc['Cents']]
        
        fig_l = go.Figure()
        # Raggi di sfondo
        for i in range(1, 10):
            fig_l.add_trace(go.Scatter(x=[0, 16], y=[0, 16*i/8], mode='lines', line=dict(color='#f0f0f0'), showlegend=False))
            
        fig_l.add_trace(go.Scatter(
            x=df_calc['q'], y=df_calc['p'],
            mode='lines+markers+text',
            text=df_calc['Note'], textposition="top center",
            marker=dict(size=25, color=colors, line=dict(width=2, color='black')),
            line=dict(dash='dot', color='gray')
        ))
        fig_l.update_layout(xaxis_title="Denominatore (q)", yaxis_title="Numeratore (p)",
                            xaxis=dict(range=[0, 10]), yaxis=dict(range=[0, 10]))
        
        st.plotly_chart(fig_l, use_container_width=True, key="plot_lambdoma") # Key aggiunta

    with tab3:
        # EDIT DATA
        st.write("Modifica i dati grezzi o aggiungi righe (T2, T3...):")
        edited = st.data_editor(st.session_state.df_data, num_rows="dynamic", key="editor_data")
        if not edited.equals(st.session_state.df_data):
            st.session_state.df_data = edited
            st.rerun()

st.caption("UT/K Model v3.1 - Simulazione Real-Time attiva.")
