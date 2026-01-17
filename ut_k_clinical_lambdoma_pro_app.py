import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="UT/K Pro: Harmonic Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. MOTORE MATEMATICO MUSICALE (Dal tuo script ut_lambdoma_music.py) ---
def cents_diff(r1, r2):
    """Calcola la differenza in cents tra due rapporti."""
    if r1 <= 0 or r2 <= 0: return 0
    return 1200 * math.log2(r1 / r2)

def get_note_name(ratio):
    """Mappa i rapporti armonici semplici sui nomi delle note (Do mobile)."""
    # Tabella di Lookup per Just Intonation (approssimata)
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
    """
    Trova la migliore frazione p/q nell'ottava [1, 2) e calcola lo shift d'ottava.
    Restituisce: (num, den, octave_shift, cents_error, note_name)
    """
    if val <= 0: return 1, 1, 0, 0, "N/A"

    # 1. Normalizzazione nell'ottava [1, 2)
    norm = val
    octave = 0

    # Gestione valori < 1 (scendiamo di ottava)
    if norm < 1:
        while norm < 1:
            norm *= 2
            octave -= 1
    # Gestione valori >= 2 (saliamo di ottava)
    elif norm >= 2:
        while norm >= 2:
            norm /= 2
            octave += 1

    # 2. Ricerca Brute-force della frazione migliore con den <= limit
    best_fract = (1, 1)
    best_error = float('inf')

    # Iteriamo sui denominatori (q)
    for q in range(1, limit + 1):
        # p parte da q (perché ratio >= 1) fino a 2*q (perché ratio < 2)
        p_start = q
        p_end = q * 2

        for p in range(p_start, p_end + 1): 
            ratio = p / q
            # Calcoliamo la distanza logaritmica per essere precisi musicalmente
            diff = abs(math.log2(norm) - math.log2(ratio))
            if diff < best_error:
                best_error = diff
                best_fract = (p, q)

    p, q = best_fract
    cents_err = cents_diff(norm, p/q)
    note = get_note_name(p/q)

    return p, q, octave, cents_err, note

# --- 2. ELABORAZIONE DATI CLINICI ---
def normalize_columns(df):
    """Gestisce sinonimi Inglese/Italiano e varianti comuni."""
    cols_map = {
        "Neutrofili": ["Neutrophils", "Neut", "neutrofili"],
        "Linfociti": ["Lymphocytes", "Lymph", "linfociti"],
        "Monociti": ["Monocytes", "Mono", "monociti"],
        "Piastrine": ["Platelets", "PLT", "piastrine", "Plt"],
        "MPV": ["MPV_fL", "VolumePiastrinico", "mpv"],
        "Cortisolo": ["Cortisol", "cortisolo"],
        "Glicemia": ["Glucose", "Glu", "glicemia"],
        "Insulina": ["Insulin", "Ins", "insulina"]
    }

    df_out = df.copy()
    # Rinomina le colonne trovate
    for standard, alts in cols_map.items():
        for alt in alts:
            # Cerca match case-insensitive
            match = next((c for c in df_out.columns if c.lower() == alt.lower()), None)
            if match and standard not in df_out.columns:
                df_out.rename(columns={match: standard}, inplace=True)

    return df_out

def calcola_dataset_completo(df):
    # Converti in numerico forzato
    numeric_cols = ['Neutrofili', 'Linfociti', 'Monociti', 'Piastrine', 'MPV', 'Cortisolo', 'Glicemia', 'Insulina']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Se mancano colonne essenziali, fermati o gestisci errore
    required = ['Neutrofili', 'Linfociti', 'Piastrine', 'MPV']
    if not all(col in df.columns for col in required):
        return df # Ritorna il df parziale o vuoto

    # Calcoli Ratios Base
    df['NLR'] = df.apply(lambda x: x['Neutrofili']/x['Linfociti'] if x['Linfociti']>0 else 0, axis=1)
    df['PLR'] = df.apply(lambda x: x['Piastrine']/x['Linfociti'] if x['Linfociti']>0 else 0, axis=1)

    if 'Monociti' in df.columns:
        df['LMR'] = df.apply(lambda x: x['Linfociti']/x['Monociti'] if x['Monociti']>0 else 0, axis=1)
    else:
        df['LMR'] = 1 # Fallback neutro

    # Vettori UT
    df['I1'] = df.apply(lambda x: x['NLR'] / x['MPV'] if x['MPV'] > 0 else 0, axis=1)
    df['I2'] = df.apply(lambda x: x['MPV'] / x['PLR'] if x['PLR'] > 0 else 0, axis=1)
    df['I3'] = df.apply(lambda x: x['PLR'] / x['LMR'] if x['LMR'] > 0 else 0, axis=1)

    # U_imm (Media Geometrica)
    df['U_imm'] = (df['I1'] * df['I2'] * df['I3'])**(1/3)

    # U_met (Metabolismo)
    if 'Insulina' in df.columns and df['Insulina'].notnull().any():
         df['U_met'] = df.apply(lambda x: x['Cortisolo']/x['Insulina'] if x['Insulina']>0 else 0, axis=1)
         df['Source_Met'] = "Cort/Ins"
    elif 'Glicemia' in df.columns:
         df['U_met'] = (df['Cortisolo'] / 15) * (df['Glicemia'] / 90) * 5
         df['Source_Met'] = "Cort/Glu (Est)"
    else:
         df['U_met'] = 1
         df['Source_Met'] = "N/A"

    df['UT'] = df['U_met'] * df['U_imm']
    df['K'] = df.apply(lambda x: x['U_met'] / x['U_imm'] if x['U_imm'] > 0 else 0, axis=1)

    # Ancoraggio a T0 e calcolo musicale
    if not df.empty and 'UT' in df.columns:
        base_ut = df.iloc[0]['UT']
        if base_ut == 0: base_ut = 1

        df['UT_Anchored'] = df['UT'] / base_ut

        harmonic_data = []
        for val in df['UT_Anchored']:
            p, q, oct_k, cents, note = best_rational_in_octave(val)
            harmonic_data.append({
                'p': p, 'q': q, 'Octave': oct_k, 'Cents': cents, 'Note': note,
                'Fraction': f"{p}/{q}"
            })

        harm_df = pd.DataFrame(harmonic_data)
        df = pd.concat([df.reset_index(drop=True), harm_df.reset_index(drop=True)], axis=1)

    return df

# --- 3. UI LAYOUT ---
st.title("🧬 UT/K Clinical Lambdoma Pro")
st.markdown("""
**Analisi Funzionale Ematochimica e Geometria Armonica**
Carica i dati per calcolare le traiettorie di salute nello spazio delle fasi (UT/K) e nella griglia armonica (Lambdoma).
""")

st.sidebar.header("Dati Paziente")
uploaded_file = st.sidebar.file_uploader("Carica Excel o CSV", type=["xlsx", "csv"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
        st.sidebar.success(f"Caricato: {uploaded_file.name}")
    except Exception as e:
        st.sidebar.error(f"Errore: {e}")
        df_raw = pd.DataFrame()
else:
    st.sidebar.info("Modalità DEMO attiva")
    demo_data = {
        'Tempo': ['T0 (Base)', 'T1 (Sepsi)', 'T2 (Infarto)', 'T3 (Stallo)'],
        'Neutrofili': [3.5, 8.5, 7.0, 6.5],
        'Linfociti': [2.0, 0.5, 2.0, 1.0],
        'Monociti': [0.5, 0.4, 0.6, 0.7],
        'Piastrine': [220, 160, 250, 450],
        'MPV': [9.5, 11.5, 12.8, 9.0],
        'Cortisolo': [12, 35, 22, 15],
        'Insulina': [8, 15, 10, 5]
    }
    df_raw = pd.DataFrame(demo_data)

if not df_raw.empty:
    df_norm = normalize_columns(df_raw)
    df_final = calcola_dataset_completo(df_norm)

    if 'UT' in df_final.columns:
        last = df_final.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("UT (Costo Totale)", f"{last['UT']:.2f}", delta=f"Base: {df_final.iloc[0]['UT']:.2f}")
        col2.metric("K (Guida)", f"{last['K']:.2f}", delta="Top-Down" if last['K']>1 else "Bottom-Up")
        col3.metric("Armonia (Nota)", f"{last['Note']}", delta=f"Ottava: {last['Octave']:+d}")
        col4.metric("Deviazione (Cents)", f"{last['Cents']:.1f}", delta_color="inverse")

        tab1, tab2, tab3 = st.tabs(["🧬 Radar Funzionale (UT/K)", "🎵 Lambdoma Armonico", "📄 Dati Tabellari"])

        with tab1:
            st.subheader("Traiettoria nello Spazio delle Fasi")
            st.caption("Asse X: Metabolismo | Asse Y: Immunità | Dimensione: Costo (UT) | Colore: Guida (K)")

            fig_ut = px.scatter(
                df_final, x="U_met", y="U_imm", size="UT", color="K",
                hover_name="Tempo", text="Tempo",
                color_continuous_scale="RdBu_r",
                size_max=60
            )
            max_ax = max(df_final['U_met'].max(), df_final['U_imm'].max()) * 1.1
            fig_ut.add_shape(type="line", x0=0, y0=0, x1=max_ax, y1=max_ax,
                             line=dict(color="Gray", dash="dash"), layer="below")
            fig_ut.update_traces(textposition='top center')
            fig_ut.update_layout(height=500, xaxis_title="U_met (Spinta Metabolica)", yaxis_title="U_imm (Forma Immunitaria)")
            st.plotly_chart(fig_ut, use_container_width=True)

        with tab2:
            st.subheader("Griglia Lambdoma (p/q)")
            st.caption("Mapping musicale dell'evoluzione del Costo UT rispetto a T0.")

            def get_color(c):
                ac = abs(c)
                if ac < 10: return '#2ca02c'
                if ac < 30: return '#ff7f0e'
                return '#d62728'

            colors = [get_color(c) for c in df_final['Cents']]

            fig_lamb = go.Figure()
            for i in range(1, 17):
                 fig_lamb.add_trace(go.Scatter(x=[1, 16], y=[1*i, 16*i], mode='lines',
                                               line=dict(color='#eeeeee', width=1),
                                               hoverinfo='skip', showlegend=False))

            fig_lamb.add_trace(go.Scatter(
                x=df_final['q'],
                y=df_final['p'],
                mode='lines+markers+text',
                text=df_final['Tempo'] + "<br>" + df_final['Note'],
                textposition="bottom center",
                line=dict(color='gray', dash='dot'),
                marker=dict(size=25, color=colors, line=dict(width=2, color='black')),
                hovertext=df_final.apply(lambda r: f"Frazione: {r['Fraction']}<br>Cents Err: {r['Cents']:.1f}<br>Ottava: {r['Octave']}", axis=1),
                name="Traiettoria"
            ))

            fig_lamb.update_layout(
                xaxis_title="Denominatore (q) - Espansione",
                yaxis_title="Numeratore (p) - Intensità",
                xaxis=dict(range=[0, 17], dtick=1),
                yaxis=dict(range=[0, 17], dtick=1),
                width=700, height=650,
                showlegend=False
            )
            st.plotly_chart(fig_lamb, use_container_width=True)
            st.info("Legenda Colori Punti: 🟢 Intonato (Coerente) | 🟠 Teso | 🔴 Stonato (Rumore)")

        with tab3:
            st.dataframe(df_final[['Tempo', 'NLR', 'MPV', 'UT', 'K', 'UT_Anchored', 'Fraction', 'Octave', 'Cents']].style.format({
                'UT': '{:.2f}', 'K': '{:.2f}', 'UT_Anchored': '{:.3f}', 'Cents': '{:.1f}'
            }))

            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df(df_final)
            st.download_button("Scarica Analisi CSV", csv, "ut_analisi.csv", "text/csv")
    else:
        st.error("Impossibile calcolare i parametri. Controlla che il file Excel abbia le colonne necessarie.")
