"""App Streamlit: predicción de partidos + explicación + decisión Kelly.

Principios de UX aplicados:
  1. Una tarea principal visible al entrar: elegir un partido y ver la recomendación.
  2. Revelación progresiva: veredicto → por qué → detalle técnico (en expanders).
  3. Pocos controles a la vista: solo el capital; las reglas avanzadas van plegadas.
  4. Colores con significado fijo: azul = modelo, naranja = mercado.

Correr local:   streamlit run app.py
En Hugging Face Spaces corre con el Dockerfile (puerto 7860).
"""
from __future__ import annotations

import os

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import betting, config, explain, features
from src.pipeline import run_pipeline

st.set_page_config(page_title="Value Betting ML · USFQ", page_icon="⚽", layout="centered")

CACHE_DIR = os.environ.get("DATA_DIR", "data/raw")
BLUE, ORANGE = "#2457B8", "#C26A12"
LEAGUE_SHORT = {d: n.split(" (")[0] for d, n in config.LEAGUES.items()}
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]


@st.cache_resource(ttl=6 * 3600, show_spinner="Preparando el modelo (≈1 min la primera vez)…")
def load_bundle():
    return run_pipeline(cache_dir=CACHE_DIR)


try:
    B = load_bundle()
except Exception as e:  # noqa: BLE001
    st.error(f"No se pudo construir el modelo: {e}")
    st.stop()

# ------------------------------------------------------------------ encabezado
st.title("⚽ ¿Hay valor en este partido?")
st.caption("El modelo estima la probabilidad de cada resultado, la compara con la cuota de Bet365 "
           "y recomienda apostar solo si el precio está barato. Proyecto académico MADN USFQ.")

top1, top2 = st.columns([1, 1.4])
bankroll = top1.number_input("Tu capital ($)", 10.0, 1_000_000.0, 1000.0, 100.0)
with top2.expander("⚙️ Reglas avanzadas"):
    min_ev = st.slider("Valor esperado mínimo", 0.0, 0.20, config.MIN_EV, 0.01, format="%.2f",
                       help="0.05 = apostar solo si esperamos ganar ≥ 5 centavos por cada $1.")
    kelly_mult = st.select_slider("Fracción de Kelly", [0.1, 0.25, 0.5, 1.0], config.KELLY_FRACTION)
    max_stake = st.slider("Tope por apuesta", 0.01, 0.20, config.MAX_STAKE, 0.01, format="%.2f")
    max_odds = st.slider("Cuota máxima", 2.0, 15.0, config.MAX_ODDS, 0.5)


def decision(p, odds):
    return betting.decide(p, odds, min_ev=min_ev, kelly_mult=kelly_mult,
                          max_stake=max_stake, max_odds=max_odds)


# ------------------------------------------------------------- detalle partido
def prob_chart(p_model, p_fair):
    rows = [{"Resultado": n, "Fuente": "Modelo", "Prob": float(a)} for n, a in zip(config.CLASS_NAMES, p_model)]
    if p_fair is not None:
        rows += [{"Resultado": n, "Fuente": "Mercado", "Prob": float(b)} for n, b in zip(config.CLASS_NAMES, p_fair)]
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(
        x=alt.X("Resultado:N", sort=config.CLASS_NAMES, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=14)),
        xOffset=alt.XOffset("Fuente:N", sort=["Modelo", "Mercado"]),
        y=alt.Y("Prob:Q", axis=alt.Axis(format="%", title=None, grid=False), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Fuente:N", scale=alt.Scale(domain=["Modelo", "Mercado"], range=[BLUE, ORANGE]),
                        legend=alt.Legend(orient="top", title=None, labelFontSize=13)),
    )
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    labels = base.mark_text(dy=-8, fontSize=12).encode(text=alt.Text("Prob:Q", format=".0%"))
    return (bars + labels).properties(height=260)


def shap_chart(contrib: pd.DataFrame, cls_name: str):
    c = contrib.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(c) + 0.7))
    ax.barh(c["Variable"], c["Impacto"], color=np.where(c["Impacto"] > 0, BLUE, ORANGE))
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlabel(f"← baja   |   sube la probabilidad de '{cls_name}' →")
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def render_match(x: pd.Series, odds: np.ndarray, title: str, subtitle: str = ""):
    p = B.final_model.predict_proba(pd.DataFrame([x[features.FEATURES]]))[0]
    d = decision(p, odds)
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)

    # 1) Veredicto
    if d["bet"] is not None:
        name = config.CLASS_NAMES[d["bet"]]
        st.success(f"### ✅ Apostar **${bankroll * d['stake']:,.2f}** a **{name}**\n"
                   f"Cuota {odds[d['bet']]:.2f} · valor esperado {d['ev'][d['bet']]:+.1%} por cada $1 · "
                   f"{d['stake']:.1%} de tu capital")
    else:
        st.info("### ⏸️ No apostar\nEl mercado ya paga estos resultados a un precio justo o caro.")

    # 2) Modelo vs mercado
    st.markdown("**Modelo vs. mercado**")
    st.altair_chart(prob_chart(p, d["p_fair"]), width="stretch")

    # 3) Por qué (lenguaje simple)
    cls = d["bet"] if d["bet"] is not None else int(np.argmax(p))
    sv = explain.shap_values(B.explainer, pd.DataFrame([x[features.FEATURES]]))[0]
    contrib = explain.explain_one(sv, x, cls)
    st.markdown(f"**¿Por qué el modelo da {p[cls]:.0%} a '{config.CLASS_NAMES[cls]}'?**")
    st.markdown("\n".join(explain.narrative(contrib, cls)))

    # 4) Detalle técnico, plegado
    with st.expander("🔍 Ver números y detalle técnico"):
        tbl = pd.DataFrame({"Resultado": config.CLASS_NAMES, "Modelo": p})
        fmt = {"Modelo": "{:.1%}"}
        if d["p_fair"] is not None:
            tbl["Mercado"], tbl["Cuota"], tbl["Valor esperado"] = d["p_fair"], odds, d["ev"]
            fmt.update({"Mercado": "{:.1%}", "Cuota": "{:.2f}", "Valor esperado": "{:+.1%}"})
        st.dataframe(tbl.style.format(fmt), hide_index=True, width="stretch")
        st.write(d["reason"])
        if d["p_fair"] is not None:
            st.caption(f"Margen de la casa: {d['overround']:.1%}. Mercado = probabilidad sin ese margen.")
        if d["bet"] is not None:
            st.caption(f"Kelly completo sugeriría {d['kelly_full'][d['bet']]:.1%}; aplicamos ×{kelly_mult} "
                       f"con tope de {max_stake:.0%}.")
        st.pyplot(shap_chart(contrib, config.CLASS_NAMES[cls]), clear_figure=True)
        st.caption("Valores SHAP: cuánto empujó cada variable (escala log-odds).")


tab1, tab2, tab3 = st.tabs(["📅 Partidos de la semana", "🧪 Simular un partido", "❓ Cómo funciona"])

# --------------------------------------------------------- partidos de la semana
with tab1:
    fx = B.fixtures
    if fx.empty:
        st.warning("No hay partidos próximos de las 5 grandes ligas en este momento (fecha FIFA o el archivo "
                   "aún no se actualiza: se carga martes y viernes). Prueba **Simular un partido**.")
    else:
        rows = []
        for i, r in fx.iterrows():
            o = r[betting.ODDS].to_numpy(dtype=float)
            p = B.fixtures_proba[i]
            d = decision(p, o)
            fav = int(np.argmax(p))
            rows.append({
                "i": i,
                "Fecha": f"{DIAS[r['Date'].weekday()]} {r['Date']:%d/%m}",
                "Partido": f"{r['HomeTeam']} – {r['AwayTeam']}",
                "Liga": LEAGUE_SHORT[r["Div"]],
                "Favorito del modelo": f"{config.CLASS_NAMES[fav]} ({p[fav]:.0%})",
                "Recomendación": (f"✅ {config.CLASS_NAMES[d['bet']]}" if d["bet"] is not None else "—"),
                "valor": d["bet"] is not None,
            })
        t = pd.DataFrame(rows)

        f1, f2 = st.columns([1.3, 1])
        ligas = f1.multiselect("Ligas", sorted(t["Liga"].unique()), placeholder="Todas")
        solo_valor = f2.toggle(f"Solo con valor ({int(t['valor'].sum())})", value=False)
        mask = pd.Series(True, index=t.index)
        if ligas:
            mask &= t["Liga"].isin(ligas)
        if solo_valor:
            mask &= t["valor"]
        view = t[mask]

        if view.empty:
            st.info("Ningún partido cumple el filtro.")
        else:
            st.caption("👆 Toca una fila para ver el análisis.")
            ev = st.dataframe(view.drop(columns=["i", "valor"]), hide_index=True, width="stretch",
                              on_select="rerun", selection_mode="single-row", height=min(38 * len(view) + 40, 360))
            sel = ev.selection.rows[0] if ev.selection.rows else 0
            r = fx.loc[view.iloc[sel]["i"]]
            st.divider()
            render_match(r, r[betting.ODDS].to_numpy(dtype=float), f"{r['HomeTeam']} vs {r['AwayTeam']}",
                         f"{LEAGUE_SHORT[r['Div']]} · {r['Date']:%d/%m/%Y}")

# ------------------------------------------------------------- simular partido
with tab2:
    st.caption("Elige dos equipos y escribe las cuotas que ves en tu casa de apuestas.")
    teams = B.state.teams_by_league()
    divs = [d for d in config.LEAGUES if d in teams]
    div = st.selectbox("Liga", divs, format_func=lambda d: config.LEAGUES[d])
    cur = B.history[(B.history["Div"] == div) & (B.history["Season"] == B.history["Season"].max())]
    recent = sorted(cur[["HomeTeam", "AwayTeam"]].stack().unique()) or teams[div]
    c1, c2 = st.columns(2)
    home = c1.selectbox("Local", recent, index=0)
    away = c2.selectbox("Visitante", [t for t in recent if t != home], index=0)
    o1, o2, o3 = st.columns(3)
    oh = o1.number_input("Cuota Local", 1.01, 50.0, 2.10, 0.05)
    od = o2.number_input("Cuota Empate", 1.01, 50.0, 3.40, 0.05)
    oa = o3.number_input("Cuota Visitante", 1.01, 50.0, 3.50, 0.05)
    st.divider()
    x = pd.Series(B.state.features_for(home, away, div, features.season_of(pd.Timestamp.today())))
    render_match(x, np.array([oh, od, oa]), f"{home} vs {away}", "Simulación con tus cuotas")

# --------------------------------------------------------------- cómo funciona
with tab3:
    st.markdown("""
#### En 4 pasos
1. **Datos:** resultados y cuotas Bet365 de las 5 grandes ligas desde 2011-12.
2. **Variables:** Elo y forma reciente de cada equipo, calculadas **solo con partidos anteriores**.
3. **Modelo:** XGBoost estima la probabilidad de Local, Empate y Visitante.
4. **Decisión:** si la probabilidad del modelo supera a la del mercado lo suficiente, se apuesta un monto
   según **Kelly fraccional**.
""")
    with st.expander("📐 Las tres fórmulas"):
        st.markdown("""
- **Probabilidad implícita** = 1 ÷ cuota, luego se normaliza para quitar el margen de la casa.
- **Valor esperado** = p modelo × cuota − 1.
- **Kelly** = (p × cuota − 1) ÷ (cuota − 1); usamos una fracción y un tope por prudencia.
""")
    with st.expander("📊 ¿Qué tan bueno es el modelo?"):
        seasons = sorted(B.test["Season"].unique())
        st.caption(f"Temporadas que el modelo nunca vio: {', '.join(s[:2] + '-' + s[2:] for s in seasons)} "
                   f"({len(B.test):,} partidos). Menor log loss / Brier = mejores probabilidades.")
        st.dataframe(B.metrics, hide_index=True, width="stretch")
    with st.expander("💰 Backtest: ¿habríamos ganado dinero?"):
        curves, summ = betting.backtest(B.test, B.test_proba, min_ev=min_ev, max_stake=max_stake,
                                        max_odds=max_odds, start_bankroll=bankroll)
        st.line_chart(curves.groupby(level=0).last(), height=280)
        st.dataframe(summ, hide_index=True, width="stretch")
    with st.expander("🧠 Variables que más pesan"):
        imp = B.importance.head(8).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.barh(imp["Variable"], imp["Total"], color=BLUE)
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    st.divider()
    last = B.history["Date"].max()
    c1, c2 = st.columns([3, 1])
    c1.caption(f"Datos: Football-Data.co.uk · {len(B.history):,} partidos · último resultado {last:%d/%m/%Y}. "
               "Equipo: Carlos Gómez, Samir Hernández, Glory Terán, Carlos Pareja. "
               "Uso académico, no es recomendación de apuesta. Juega responsable.")
    if c2.button("🔄 Actualizar datos"):
        st.cache_resource.clear()
        st.rerun()
