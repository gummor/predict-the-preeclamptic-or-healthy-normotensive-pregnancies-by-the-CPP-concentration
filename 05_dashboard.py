import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import plotly.graph_objects as go

# Configuração da Página
st.set_page_config(page_title="Preeclampsia Risk Engine", layout="wide")

# Carregar o Modelo e Scaler
@st.cache_resource
def load_models():
    return joblib.load('modelo_rf_preeclampsia.joblib'), joblib.load('scaler_robusto.joblib')

model, scaler = load_models()

st.title("Preeclampsia Risk Assessment & Meta-Analysis Dashboard")
st.markdown("---")

col1, col2 = st.columns([1, 2])

# Lado Esquerdo: Preditor Individual
with col1:
    st.header("Individual Patient Inference")
    age = st.slider("Patient Age (years)", 18, 45, 30)
    copeptin = st.slider("Copeptin Level (pmol/L)", 1.0, 40.0, 10.0)
    
    if st.button("Predict Clinical Risk"):
        features = scaler.transform(np.array([[age, copeptin]]))
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]
        
        if pred == 1:
            st.error(f"**High Risk Detected** (Probability: {prob:.1%})")
        else:
            st.success(f"**Low Risk** (Probability: {prob:.1%})")

# Lado Direito: O seu Forest Plot Científico
with col2:
    st.header("Global Meta-Analysis: HNS in Preeclampsia")
    st.write("Findings from our systematic review on Copeptin surrogate levels worldwide.")
    
    # Dados exatos da sua publicação
    labels = ['Sensitivity Analysis', 'Overall Effect', 'North America', 'Europe', 'Asia', 'Africa']
    smd = [1.77, 1.78, 0.98, 1.32, 3.46, 2.73]
    ci_lower = [1.20, 1.26, 0.69, 0.65, 1.32, 1.94]
    ci_upper = [2.33, 2.30, 1.26, 1.99, 5.60, 4.52]
    
    # Cálculo dos erros para o Plotly
    error_minus = [s - l for s, l in zip(smd, ci_lower)]
    error_plus = [u - s for u, s in zip(ci_upper, smd)]
    
    fig = go.Figure(data=go.Scatter(
        x=smd, y=labels, mode='markers',
        marker=dict(size=12, color=['black', 'darkblue', 'teal', 'teal', 'teal', 'teal'], 
                    symbol=['diamond', 'diamond', 'square', 'square', 'square', 'square']),
        error_x=dict(type='data', symmetric=False, array=error_plus, arrayminus=error_minus, color='teal')
    ))
    
    fig.add_shape(type="line", x0=0, y0=-0.5, x1=0, y1=5.5, line=dict(color="grey", width=1, dash="dash"))
    fig.update_layout(title="Standardised Mean Difference (SMD) [95% CI]",
                      xaxis_title="<-- Favours Control       |       Favours Preeclampsia -->",
                      height=400)
    st.plotly_chart(fig, use_container_width=True)

# Secção Inferior: Estatísticas da Cohort via SQL
st.markdown("---")
st.subheader("Simulated Local Cohort Database (SQLite)")
conn = sqlite3.connect('preeclampsia.db')
df_sql = pd.read_sql_query("SELECT Continent, Country, COUNT(*) as Patients, AVG(Copeptin) as Avg_Copeptin FROM patients GROUP BY Continent, Country", conn)
conn.close()
st.dataframe(df_sql, use_container_width=True)
