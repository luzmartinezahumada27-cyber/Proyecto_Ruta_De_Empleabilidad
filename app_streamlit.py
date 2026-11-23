# -*- coding: utf-8 -*-
"""
Dashboard interactivo de la Ruta de Empleabilidad de Barranquilla (Streamlit).

Cómo ejecutarlo:
    pip install streamlit pandas altair
    streamlit run app_streamlit.py

Necesita el archivo salidas/registros_limpios.csv, que genera el notebook (sección 7).
"""
import json
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

AZUL, NARANJA = "#2a78d6", "#eb6834"
COLORES_CANAL = alt.Scale(domain=["Agencia", "Autoregistro"], range=[AZUL, NARANJA])
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
ACCIONES = {
    "Jóvenes buscando primer empleo": "Ferias de primer empleo, habilidades blandas y alianzas con instituciones educativas.",
    "Desempleados con formación media": "Cursos cortos certificables y vacantes operativas.",
    "Desempleados con formación técnica o superior": "Intermediación directa con vacantes calificadas.",
    "Desempleados de 40 años o más": "Orientación personalizada y empleadores con apertura a mayor edad.",
    "Ocupados que buscan mejorar": "Movilidad laboral; menor prioridad que quienes no tienen empleo.",
}

st.set_page_config(page_title="Ruta de Empleabilidad", layout="wide")


@st.cache_data
def cargar():
    df = pd.read_csv("salidas/registros_limpios.csv", parse_dates=["periodo"])
    df["canal"] = df["canal"].replace({"Autoregistro": "Autoregistro", "Agencia": "Agencia"})
    resultados = {}
    if os.path.exists("salidas/resultados.json"):
        with open("salidas/resultados.json", encoding="utf-8") as f:
            resultados = json.load(f)
    return df, resultados


df, res = cargar()

# ───────────────────────── Filtros ─────────────────────────
st.sidebar.header("Filtros")
anios = st.sidebar.multiselect("Año", sorted(df["anio"].unique()), default=sorted(df["anio"].unique()))
puntos = st.sidebar.multiselect("Punto de atención", sorted(df["punto"].unique()), default=sorted(df["punto"].unique()))
canales = st.sidebar.multiselect("Canal", ["Agencia", "Autoregistro"], default=["Agencia", "Autoregistro"])
generos = st.sidebar.multiselect("Género", ["F", "M"], default=["F", "M"])

f = df[df["anio"].isin(anios) & df["punto"].isin(puntos) & df["canal"].isin(canales) & df["genero"].isin(generos)]
if f.empty:
    st.warning("Los filtros no dejan ningún registro. Amplíe la selección.")
    st.stop()

st.title("Ruta de Empleabilidad: calidad del registro")
st.caption("Un registro es **útil** cuando informa la situación laboral y el nivel de estudio. "
           "Fuente: Secretaría de Desarrollo Económico de Barranquilla, ene 2020 a ago 2022.")

# ───────────────────────── KPIs ─────────────────────────
informados = f[f["sit_laboral"] != "NO REGISTRA"]
prioritario = informados["sit_laboral"].isin(["Desempleado", "Primer Empleo", "Cesante por Emergencia Sanitaria"]).mean()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Registros", f"{len(f):,}".replace(",", "."), delta=None)
c2.metric("Tasa de registro útil", f"{f['registro_util'].mean():.1%}", f"{f['registro_util'].mean() - df['registro_util'].mean():+.1%} vs. global")
c3.metric("Por autoregistro", f"{(f['canal'] == 'Autoregistro').mean():.1%}")
c4.metric("Población prioritaria", f"{prioritario:.1%}")

# ───────────────────────── Qué está pasando ─────────────────────────
st.header("Qué está pasando")
izq, der = st.columns(2)

with izq:
    st.subheader("Registros por mes y canal")
    mensual = f.groupby(["periodo", "canal"]).size().reset_index(name="registros")
    st.altair_chart(
        alt.Chart(mensual).mark_bar().encode(
            x=alt.X("periodo:T", title=None),
            y=alt.Y("registros:Q", title="Registros"),
            color=alt.Color("canal:N", scale=COLORES_CANAL, title=None),
            tooltip=["periodo:T", "canal:N", "registros:Q"],
        ).properties(height=280),
        use_container_width=True,
    )

with der:
    st.subheader("Tasa de registro útil por mes y canal")
    tasa = f.groupby(["periodo", "canal"]).agg(tasa_util=("registro_util", "mean"), n=("registro_util", "size")).reset_index()
    tasa = tasa[tasa["n"] >= 15]  # se omiten meses con muy pocos registros
    st.altair_chart(
        alt.Chart(tasa).mark_line(point=True).encode(
            x=alt.X("periodo:T", title=None),
            y=alt.Y("tasa_util:Q", title="Tasa de registro útil", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            color=alt.Color("canal:N", scale=COLORES_CANAL, title=None),
            tooltip=["periodo:T", "canal:N", alt.Tooltip("tasa_util:Q", format=".1%"), "n:Q"],
        ).properties(height=280),
        use_container_width=True,
    )

izq, der = st.columns(2)
with izq:
    st.subheader("Registro útil por punto y canal")
    pc = f.groupby(["punto", "canal"]).agg(tasa_util=("registro_util", "mean"), n=("registro_util", "size")).reset_index()
    st.altair_chart(
        alt.Chart(pc).mark_bar().encode(
            y=alt.Y("punto:N", title=None, sort=alt.SortField("tasa_util")),
            x=alt.X("tasa_util:Q", title="Tasa de registro útil", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            yOffset="canal:N",
            color=alt.Color("canal:N", scale=COLORES_CANAL, title=None),
            tooltip=["punto:N", "canal:N", alt.Tooltip("tasa_util:Q", format=".1%"), "n:Q"],
        ).properties(height=280),
        use_container_width=True,
    )

with der:
    st.subheader("Registro útil según día y franja horaria")
    h = f.dropna(subset=["franja", "dia_semana"]).copy()
    h["dia"] = h["dia_semana"].astype(int).map(dict(enumerate(DIAS)))
    mapa = h.groupby(["franja", "dia"]).agg(tasa_util=("registro_util", "mean"), n=("registro_util", "size")).reset_index()
    mapa = mapa[mapa["n"] >= 30]
    base = alt.Chart(mapa).encode(
        x=alt.X("dia:N", sort=DIAS, title=None),
        y=alt.Y("franja:N", sort=["0-6 h", "7-9 h", "10-12 h", "13-15 h", "16-18 h", "19-23 h"], title=None),
    )
    st.altair_chart(
        (base.mark_rect().encode(color=alt.Color("tasa_util:Q", scale=alt.Scale(scheme="blues"), title="Tasa útil", legend=alt.Legend(format="%")),
                                 tooltip=["dia:N", "franja:N", alt.Tooltip("tasa_util:Q", format=".1%"), "n:Q"])
         + base.mark_text(fontSize=11).encode(text=alt.Text("tasa_util:Q", format=".0%"),
                                              color=alt.condition(alt.datum.tasa_util > 0.6, alt.value("white"), alt.value("black")))
         ).properties(height=280),
        use_container_width=True,
    )

# ───────────────────────── Quiénes se registran ─────────────────────────
st.header("Quiénes se registran")
izq, der = st.columns(2)
with izq:
    st.subheader("Situación laboral declarada")
    sit = f["sit_laboral"].value_counts().rename_axis("situación").reset_index(name="registros")
    st.altair_chart(
        alt.Chart(sit).mark_bar(color=AZUL).encode(
            y=alt.Y("situación:N", sort="-x", title=None), x=alt.X("registros:Q", title="Registros"),
            tooltip=["situación:N", "registros:Q"],
        ).properties(height=280),
        use_container_width=True,
    )
with der:
    st.subheader("Perfiles de personas con registro útil")
    seg = f["segmento"].value_counts().rename_axis("segmento").reset_index(name="personas")
    seg["acción sugerida"] = seg["segmento"].map(ACCIONES)
    st.dataframe(seg, use_container_width=True, hide_index=True)

# ───────────────────────── Simulador ─────────────────────────
st.header("Qué decidir: simulador de la intervención C")
st.caption("Trasladar una parte de los autoregistros a atención asistida. Los costos y valores son supuestos que puede cambiar.")

n_meses = df["periodo"].nunique()
n_auto_mes = (df["canal"] == "Autoregistro").sum() / n_meses
n_agencia_mes = (df["canal"] == "Agencia").sum() / n_meses
uplift = float(res.get("uplift_ajustado", 0.4475))  # efecto ajustado del logit (se guarda en el notebook)

s1, s2 = st.columns([1, 2])
with s1:
    share = st.slider("Autoregistros trasladados (%)", 0, 50, 20, 5) / 100
    c_asist = st.slider("Costo por atención asistida (COP)", 3000, 20000, 8000, 500)
    valor = st.slider("Valor de un registro útil (COP)", 10000, 50000, 25000, 1000)
    factor = st.slider("Factor de selección (efecto que se conserva)", 0.3, 1.0, 0.7, 0.05)

movidos = share * n_auto_mes
extra = movidos * uplift * factor
costo = movidos * c_asist
neto = extra * valor - costo
equilibrio = uplift * factor * valor
with s2:
    a, b, c = st.columns(3)
    a.metric("Atenciones extra al mes", f"{movidos:,.0f}")
    b.metric("Registros útiles extra al mes", f"{extra:,.0f}")
    c.metric("Costo mensual (COP)", f"{costo:,.0f}")
    d, e, g = st.columns(3)
    d.metric("Costo por útil extra (COP)", "–" if extra == 0 else f"{costo / extra:,.0f}")
    e.metric("Beneficio neto mensual (COP)", f"{neto:,.0f}")
    g.metric("Costo de equilibrio (COP)", f"{equilibrio:,.0f}")
    if share == 0:
        st.info("Sin traslado no hay cambio.")
    elif neto > 0:
        st.success(f"Con estos supuestos se paga solo. Carga extra en agencias: {movidos / n_agencia_mes:.0%}.")
    else:
        st.error(f"Con estos supuestos no se paga solo: la atención debería costar menos de COP {equilibrio:,.0f}.")

# Tabla de sensibilidad
costos = [5000, 8000, 11000, 14000, 17000]
shares = [0.1, 0.2, 0.3, 0.4, 0.5]
sens = pd.DataFrame(
    [[(s * n_auto_mes * uplift * factor * valor - s * n_auto_mes * c) / 1e6 for s in shares] for c in costos],
    index=[f"COP {c:,}" for c in costos], columns=[f"{int(s * 100)} %" for s in shares],
)
st.subheader("Sensibilidad: beneficio neto mensual (millones de COP)")
st.dataframe(sens.style.format("{:.2f}").background_gradient(cmap="RdBu", axis=None, vmin=-1.5, vmax=1.5), use_container_width=True)
