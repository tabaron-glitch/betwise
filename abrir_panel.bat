@echo off
REM Abre el panel de BetWise en el navegador (solo en este computador). Cierra esta ventana para apagarlo.
cd /d "%~dp0"
".venv\Scripts\streamlit.exe" run app\panel.py --server.port 8501
