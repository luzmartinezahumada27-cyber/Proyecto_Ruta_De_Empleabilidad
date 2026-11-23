# Ruta de Empleabilidad Barranquilla: proyecto de analítica

## Contenido
- `notebook_proyecto.ipynb`  Notebook con las 4 fases (ya ejecutado, con salidas y figuras).
- `datos/ruta_empleabilidad.csv`  Base original (23.353 registros).
- `salidas/registros_limpios.csv`  Base limpia con probabilidad de registro incompleto y segmento (alimenta el dashboard).
- `salidas/*.csv|json`  Tablas de resultados (métricas, escenarios, perfiles, etc.).
- `figuras/`  Figuras del notebook (para el informe y las diapositivas).
- `dashboard.html`  Dashboard interactivo autocontenido: abrir con doble clic en Chrome/Edge.
- `app_streamlit.py`  Dashboard en Python (Streamlit).
- `celdas.py` y `ejecutar.py`  Generador del notebook (opcional).

## Cómo correr el notebook
1. Instalar: `pip install -r requirements.txt` (statsmodels y xgboost son opcionales).
2. Abrir `notebook_proyecto.ipynb` en Jupyter o VS Code desde esta misma carpeta y ejecutar todo.
   El notebook lee `datos/ruta_empleabilidad.csv` y escribe en `salidas/` y `figuras/`.

## Cómo correr el dashboard en Python
    pip install streamlit pandas altair
    streamlit run app_streamlit.py
(Ejecutar desde esta carpeta; necesita `salidas/registros_limpios.csv`.)

## Importante
- Los costos (atención COP 8.000, contacto COP 2.500, valor de un registro útil COP 25.000, éxito 30 %) son SUPUESTOS del grupo. Cámbielos en la sección 8 del notebook.
- `app_streamlit.py` se verificó solo en sintaxis; Streamlit no estaba instalado donde se preparó. `dashboard.html` sí se probó en Chrome (escritorio y ancho de teléfono).
