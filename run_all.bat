@echo off
echo Checking environment...
python check_environment.py || exit /b
echo Testing Streamlit UI...
python test_ui.py || exit /b
REM Quand le modèle sera là, décommente la ligne suivante :
REM python test_inference.py || exit /b
echo All checks passed!
pause