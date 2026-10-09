"""App Streamlit: predicción de partidos próximos + explicación + decisión Kelly.

Correr local:   streamlit run app.py
En Hugging Face Spaces corre con el Dockerfile (puerto 7860).
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import betting, config, explain, features
from src.pipeline import run_pipeline

st.set_page_config(page_title="Value Betting ML · USFQ", page_icon="⚽", layout="wide")

CACHE_DIR = os.environ.get("DATA_DIR", "data/raw")
OUT = ["H", "D", "A"]


@st.cache_resource(ttl=6 * 3600, show_spinner="Descargando datos y entrenando el modelo (≈1 min la primera vez)…")
def load_bundle():
    return run_pipeline(cache_dir=CACHE_DIR)


# --------------------------------------------------------------------------- UI
st.title("⚽ Modelo predictivo y de optimización financiera para apuestas")
st.caption("Proyecto final · IA y Analítica Aplicada · MADN USFQ — Carlos Gómez, Samir Hernández, "
           "Glory Terán, Carlos Pareja. **Uso académico: no es una recomendación de apuesta.**")

try:
    B = load_bundle()
except Exception as e:  # noqa: BLE001
    st.error(f"No se pudo construir el modelo: {e}")
    st.stop()

with st.sidebar:
    st.header("⚙️ Reglas de decisión")
    bankroll = st.number_input("Bankroll (capital) en $", 10.0, 1_000_000.0, 1000.0, 50.0)
    min_ev = st.slider("Valor esperado mínimo para apostar", 0.0, 0.20, config.MIN_EV, 0.01, format="%.2f",
                       help="0.05 = solo apostar si esperamos ganar ≥5 centavos por cada $1.")
    kelly_mult = st.select_slider("Fracción de Kelly", [0.1, 0.25, 0.5, 1.0], config.KELLY_FRACTION,
                                  help="Kelly completo maximiza crecimiento pero es muy volátil.")
    max_stake = st.slider("Tope por apuesta (% del bankroll)", 0.01, 0.20, config.MAX_STAKE, 0.01)
    max_odds = st.slider("Cuota máxima aceptada", 2.0, 15.0, config.MAX_ODDS, 0.5)
    st.divider()
    last = B.history["Date"].max()
    st.caption(f"Datos: Football-Data.co.uk · {len(B.history):,} partidos · último resultado "
               f"{last:%d/%m/%Y} · modelo con {B.n_trees} árboles.")
    if st.button("🔄 Recargar datos"):
        st.cache_resource.clear()
        st.rerun()


def decision(p, odds):
    return betting.decide(p, odds, min_ev=min_ev, kelly_mult=kelly_mult,
                          max_stake=max_stake, max_odds=max_odds)


def shap_chart(contrib: pd.DataFrame, cls_name: str):
    c = contrib.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(c) + 0.6))
    ax.barh(c["Variable"], c["Impacto"], color=np.where(c["Impacto"] > 0, "#2e7d32", "#c62828"))
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlabel(f"Empuje hacia '{cls_name}' (log-odds)   ← baja | sube →")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def render_match(x: pd.Series, odds: np.ndarray, title: str):
    """Detalle completo de un partido: probabilidades, decisión y por qué."""
    p = B.final_model.predict_proba(pd.DataFrame([x[features.FEATURES]]))[0]
    d = decision(p, odds)
    st.subheader(title)

    c1, c2 = st.columns([1.1, 1])
    with c1:
        tbl = pd.DataFrame({"Resultado": config.CLASS_NAMES, "Modelo": p})
        if d["p_fair"] is not None:
            tbl["Mercado (sin margen)"] = d["p_fair"]
            tbl["Cuota Bet365"] = odds
            tbl["Valor esperado"] = d["ev"]
        fmt = {"Modelo": "{:.1%}", "Mercado (sin margen)": "{:.1%}", "Cuota Bet365": "{:.2f}",
               "Valor esperado": "{:+.1%}"}
        st.dataframe(tbl.style.format({k: v for k, v in fmt.items() if k in tbl}), hide_index=True,
                     width="stretch")
        if d["p_fair"] is not None:
            st.caption(f"Margen de la casa (overround): {d['overround']:.1%}")
    with c2:
        if d["bet"] is not None:
            st.success(f"### ✅ Apostar a **{config.CLASS_NAMES[d['bet']]}**\n"
                       f"Monto sugerido: **${bankroll * d['stake']:,.2f}** ({d['stake']:.2%} del bankroll)")
        else:
            st.info("### ⏸️ No apostar")
        st.write(d["reason"])
        if d["bet"] is not None:
            k = d["kelly_full"][d["bet"]]
            st.caption(f"Kelly completo sugeriría {k:.2%}; aplicamos ×{kelly_mult} y tope {max_stake:.0%}.")

    cls = d["bet"] if d["bet"] is not None else int(np.argmax(p))
    sv = explain.shap_values(B.explainer, pd.DataFrame([x[features.FEATURES]]))[0]
    contrib = explain.explain_one(sv, x, cls)
    st.markdown(f"#### 🧠 ¿Por qué el modelo da {p[cls]:.1%} a '{config.CLASS_NAMES[cls]}'?")
    c3, c4 = st.columns([1.3, 1])
    with c3:
        st.pyplot(shap_chart(contrib, config.CLASS_NAMES[cls]), clear_figure=True)
    with c4:
        st.markdown("\n".join(explain.narrative(contrib, cls)))
        st.caption("Verde empuja hacia arriba la probabilidad de ese resultado; rojo la empuja hacia abajo "
                   "(valores SHAP).")


tab1, tab2, tab3, tab4 = st.tabs(["📅 Partidos próximos", "🧪 Simulador", "📈 Backtest y métricas",
                                  "🧩 Cómo funciona"])

# ------------------------------------------------------------- Partidos próximos
with tab1:
    fx = B.fixtures
    if fx.empty:
        st.warning("No hay partidos próximos de las 5 grandes ligas en el archivo de Football-Data "
                   "(puede ser fecha FIFA o el archivo aún no se actualiza: se carga martes y viernes). "
                   "Usa el **Simulador** mientras tanto.")
    else:
        rows = []
        for i, r in fx.iterrows():
            o = r[betting.ODDS].to_numpy(dtype=float)
            d = decision(B.fixtures_proba[i], o)
            rows.append({
                "Fecha": f"{r['Date']:%d/%m} {r['Time'] if pd.notna(r['Time']) else ''}",
                "Liga": config.LEAGUES[r["Div"]].split(" (")[0],
                "Partido": f"{r['HomeTeam']} vs {r['AwayTeam']}",
                "P(L)": B.fixtures_proba[i][0], "P(E)": B.fixtures_proba[i][1], "P(V)": B.fixtures_proba[i][2],
                "Decisión": (f"✅ {config.CLASS_NAMES[d['bet']]} @ {o[d['bet']]:.2f}" if d["bet"] is not None
                             else "⏸️ No apostar"),
                "EV": (d["ev"][d["bet"]] if d["bet"] is not None else np.nan),
            })
        t = pd.DataFrame(rows)
        t["EV"] = pd.to_numeric(t["EV"], errors="coerce")
        st.markdown(f"**{len(t)} partidos** · {int(t['EV'].notna().sum())} con valor según las reglas actuales.")
        st.dataframe(t.style.format({"P(L)": "{:.0%}", "P(E)": "{:.0%}", "P(V)": "{:.0%}", "EV": "{:+.1%}"},
                                    na_rep="—"), hide_index=True, width="stretch", height=330)
        pick = st.selectbox("Elige un partido para ver el detalle y la explicación", range(len(t)),
                            format_func=lambda i: f"{t.loc[i, 'Fecha']} · {t.loc[i, 'Partido']}")
        r = fx.iloc[pick]
        render_match(r, r[betting.ODDS].to_numpy(dtype=float), f"{r['HomeTeam']} vs {r['AwayTeam']}")

# ------------------------------------------------------------------- Simulador
with tab2:
    st.markdown("Elige cualquier par de equipos y escribe las cuotas que ves en tu casa de apuestas.")
    teams = B.state.teams_by_league()
    divs = [d for d in config.LEAGUES if d in teams]
    c1, c2, c3 = st.columns(3)
    div = c1.selectbox("Liga", divs, format_func=lambda d: config.LEAGUES[d])
    recent = sorted(B.history[(B.history["Div"] == div) & (B.history["Season"] == B.history["Season"].max())]
                    [["HomeTeam", "AwayTeam"]].stack().unique()) or teams[div]
    home = c2.selectbox("Local", recent, index=0)
    away = c3.selectbox("Visitante", [t for t in recent if t != home], index=0)
    o1, o2, o3 = st.columns(3)
    oh = o1.number_input("Cuota Local", 1.01, 50.0, 2.10, 0.05)
    od = o2.number_input("Cuota Empate", 1.01, 50.0, 3.40, 0.05)
    oa = o3.number_input("Cuota Visitante", 1.01, 50.0, 3.50, 0.05)
    x = pd.Series(B.state.features_for(home, away, div, features.season_of(pd.Timestamp.today())))
    render_match(x, np.array([oh, od, oa]), f"{home} vs {away} (simulado)")

# -------------------------------------------------------------------- Backtest
with tab3:
    seasons = sorted(B.test["Season"].unique())
    st.markdown(f"Evaluación en temporadas **que el modelo nunca vio**: "
                f"{', '.join(s[:2] + '-' + s[2:] for s in seasons)} ({len(B.test):,} partidos).")
    st.markdown("##### Calidad de las probabilidades (menor log loss / Brier = mejor)")
    st.dataframe(B.metrics, hide_index=True, width="stretch")
    st.caption("La vara real es el **mercado**: si el modelo no se acerca a su log loss, sus 'ineficiencias' "
               "son probablemente error del modelo, no del mercado.")
    curves, summ = betting.backtest(B.test, B.test_proba, min_ev=min_ev, max_stake=max_stake, max_odds=max_odds,
                                    start_bankroll=bankroll)
    st.markdown("##### Evolución del bankroll con las reglas de la barra lateral")
    st.line_chart(curves.groupby(level=0).last())
    st.dataframe(summ, hide_index=True, width="stretch")
    hr = summ.attrs.get("hit_rate")
    st.caption(f"Apuestas con valor: {summ.attrs['n_value_bets']} · tasa de acierto: "
               f"{hr:.1%}" if hr == hr else "Sin apuestas con estas reglas.")

# ------------------------------------------------------------- Cómo funciona
with tab4:
    st.markdown("""
### El pipeline en 4 pasos
| Paso | Qué hace | Pregunta que responde |
|---|---|---|
| **1. Datos** | Resultados y cuotas Bet365 de las 5 grandes ligas desde 2011 (Football-Data.co.uk) | ¿Qué pasó y a qué precio? |
| **2. Variables** | Elo de cada equipo, forma de los últimos 5, goles a favor/en contra, rendimiento de local/visitante. Se calculan **solo con partidos anteriores** | ¿Qué tan fuerte llega cada equipo? |
| **3. XGBoost** | Clasificador multiclase → probabilidad de Local / Empate / Visitante | ¿Qué tan probable es cada resultado? |
| **4. Decisión** | Quita el margen de la cuota, compara con el modelo, calcula valor esperado y Kelly fraccional | ¿Apuesto? ¿A qué? ¿Cuánto? |

### Las tres fórmulas
- **Probabilidad implícita** = 1 / cuota → luego se normaliza para que las tres sumen 100%.
- **Valor esperado** = p_modelo × cuota − 1 → si es > umbral, hay "valor".
- **Kelly** = (p × cuota − 1) / (cuota − 1) → % del capital; usamos una fracción y un tope por prudencia.
""")
    st.markdown("##### Variables que más pesan en el modelo (SHAP promedio en prueba)")
    imp = B.importance.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp["Variable"], imp["Total"], color="#1565c0")
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Importancia promedio |SHAP|")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    st.caption("Juego responsable: las casas de apuestas tienen ventaja estructural. Este proyecto mide si un "
               "modelo puede encontrar precios mal fijados; no garantiza ganancias.")
