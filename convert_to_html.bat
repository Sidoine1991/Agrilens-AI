@echo off
echo ========================================
echo   AgriLens AI - Conversion HTML
echo ========================================
echo.

echo Installation de markdown...
python -m pip install markdown

echo.
echo Conversion en cours...
python convert_to_html.py

echo.
echo Conversion terminee!
echo Le fichier HTML a ete ouvert dans votre navigateur.
echo Pour creer un PDF: Ctrl+P puis "Enregistrer en PDF"
echo.
pause 