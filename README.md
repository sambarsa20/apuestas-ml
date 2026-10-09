---
title: Value Betting ML USFQ
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# ⚽ Modelo Predictivo y de Optimización Financiera para Apuestas Deportivas

Proyecto final · IA y Analítica Aplicada · MADN USFQ
**Equipo:** Carlos Gómez · Samir Hernández · Glory Terán · Carlos Pareja

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sambarsa20/apuestas-ml/blob/main/notebooks/Proyecto_Apuestas_ML.ipynb)

> Repo del equipo: https://github.com/sambarsa20/apuestas-ml

## Qué hace
1. **Datos**: resultados y cuotas Bet365 de las 5 grandes ligas desde 2011-12 (Football-Data.co.uk) + próximos partidos (`fixtures.csv`).
2. **Variables sin fuga**: Elo, forma de los últimos 5 partidos, goles, rendimiento local/visitante, todo calculado solo con el pasado.
3. **XGBoost multiclase**: probabilidad de Local / Empate / Visitante.
4. **Decisión**: quita el margen de la casa, calcula valor esperado y aplica **Kelly fraccional** (1/4, tope 5%, cuota máx. 5).
5. **Explicación**: SHAP muestra qué variables empujaron cada predicción.

## Estructura
```
apuestas-ml/
├── notebooks/Proyecto_Apuestas_ML.ipynb   ← explicación paso a paso (para la defensa)
├── src/                                   ← código compartido por notebook y app
│   ├── config.py      parámetros (ligas, Elo, XGBoost, reglas de apuesta)
│   ├── data.py        descarga y limpieza
│   ├── features.py    variables sin fuga (clase TeamState)
│   ├── model.py       split temporal, entrenamiento, métricas
│   ├── betting.py     margen, EV, Kelly, backtest
│   ├── explain.py     SHAP
│   └── pipeline.py    todo junto (lo usa la app)
├── app.py                                 ← app Streamlit
├── Dockerfile                             ← despliegue en Hugging Face Spaces
├── tests/make_synthetic_data.py           ← datos simulados para probar sin internet
└── .github/workflows/sync_to_hf.yml       ← GitHub → Hugging Face automático
```

## Correr en local
```bash
pip install -r requirements.txt
streamlit run app.py            # la primera vez descarga datos y entrena (~1 min)
```

Guía de trabajo en equipo y despliegue: ver **[COLABORAR.md](COLABORAR.md)**.

*Uso académico. Las casas de apuestas tienen ventaja estructural; este proyecto no es una recomendación de apuesta.*
