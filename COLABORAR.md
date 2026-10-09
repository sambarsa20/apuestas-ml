# Guía del equipo: GitHub + Colab + Hugging Face

## Parte A · Repositorio (ya creado)
- Repo: **https://github.com/sambarsa20/apuestas-ml** (público, para que Colab y Hugging Face lo lean sin claves).
- Samir: **Settings → Collaborators → Add people** y agrega a Carlos G., Glory y Carlos P. para que puedan guardar cambios.

## Parte B · Cómo trabaja cada uno en Colab
- Abrir: botón **"Abrir en Colab"** del README → *Entorno de ejecución → Ejecutar todo*. La primera celda clona el repo y descarga los datos.
- Cada sección del notebook tiene un **Responsable** (tabla al inicio). Edita solo tu sección para no pisar a otros.
- Guardar cambios: *Archivo → Guardar una copia en GitHub* → repo `apuestas-ml`, rama **`main`**, mensaje descriptivo (ej. "Sección 5: interpretación de métricas").
- ⚠️ Regla de oro: **antes de editar, vuelve a abrir el notebook desde GitHub** (no desde una pestaña vieja). Si dos personas guardan versiones distintas, la última borra los cambios de la otra.
- Cambios a la lógica (`src/*.py`) → se editan en GitHub (lápiz ✏️) y se avisa al grupo, porque afectan notebook y app a la vez.

## Parte C · Desplegar la app en Hugging Face (Samir, ~10 min)
1. Crea cuenta en [huggingface.co](https://huggingface.co/join).
2. [huggingface.co/new-space](https://huggingface.co/new-space): nombre `apuestas-ml`, **SDK: Docker → Blank**, hardware **CPU basic (free)**, **Public**.
3. Conectar GitHub → Hugging Face (cada cambio en `main` se publica solo):
   - En Hugging Face: *Settings → Access Tokens → Create new token* tipo **Write**. Copia el token.
   - En GitHub, repo `apuestas-ml` → *Settings → Secrets and variables → Actions*:
     - pestaña **Secrets** → `HF_TOKEN` = el token.
     - pestaña **Variables** → `HF_USER` = tu usuario de Hugging Face, `HF_SPACE` = `apuestas-ml`.
   - *Actions → Sync to Hugging Face Space → Run workflow*.
4. En el Space verás *Building* (3–5 min) y luego la app. La primera carga descarga datos y entrena (~1 min).
5. Comparte el enlace `https://huggingface.co/spaces/<HF_USER>/apuestas-ml` con el profesor.

> Alternativa sin GitHub Actions: en el Space → *Files → Add file → Upload files* y subir los mismos archivos.

### Notas de operación
- Los espacios gratuitos **se duermen** tras un tiempo sin uso; la primera visita los despierta (tarda 1–2 min). Ábrelo unos minutos antes de presentar.
- Football-Data sube cuotas de partidos próximos los **martes** (mitad de semana) y **viernes** (fin de semana). En fechas FIFA puede no haber partidos: usar la pestaña **Simulador**.
- Botón **🔄 Recargar datos** en la app: vuelve a descargar y reentrenar.

## Parte D · Probar sin internet (opcional)
```bash
python tests/make_synthetic_data.py /tmp/fd_test
DATA_DIR=/tmp/fd_test streamlit run app.py
```
Los datos simulados solo prueban que el código corre; sus resultados no significan nada.
