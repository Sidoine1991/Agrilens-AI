#!/usr/bin/env python3
"""
Script de conversion de la note technique Markdown en HTML
AgriLens AI - Technical Documentation
"""

import markdown
from pathlib import Path
import os
import sys
import webbrowser

def create_html_from_markdown(md_file_path):
    """Convertit le fichier Markdown en HTML avec CSS personnalisé"""
    
    # Lire le contenu Markdown
    with open(md_file_path, 'r', encoding='utf-8') as file:
        md_content = file.read()
    
    # CSS personnalisé pour une présentation professionnelle
    css_style = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px;
            background: #ffffff;
        }
        
        h1 {
            color: #1a5f7a;
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 20px;
            border-bottom: 3px solid #28a745;
            padding-bottom: 10px;
        }
        
        h2 {
            color: #2c3e50;
            font-size: 1.8em;
            font-weight: 600;
            margin-top: 40px;
            margin-bottom: 20px;
            border-left: 4px solid #28a745;
            padding-left: 15px;
        }
        
        h3 {
            color: #34495e;
            font-size: 1.4em;
            font-weight: 500;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        
        h4 {
            color: #34495e;
            font-size: 1.2em;
            font-weight: 500;
            margin-top: 25px;
            margin-bottom: 10px;
        }
        
        p {
            margin-bottom: 15px;
            text-align: justify;
        }
        
        ul, ol {
            margin-bottom: 20px;
            padding-left: 30px;
        }
        
        li {
            margin-bottom: 8px;
        }
        
        code {
            background-color: #f8f9fa;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #e83e8c;
        }
        
        pre {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            overflow-x: auto;
            margin: 20px 0;
        }
        
        pre code {
            background: none;
            padding: 0;
            color: #333;
        }
        
        blockquote {
            border-left: 4px solid #28a745;
            margin: 20px 0;
            padding: 15px 20px;
            background-color: #f8f9fa;
            font-style: italic;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.9em;
        }
        
        th, td {
            border: 1px solid #dee2e6;
            padding: 12px;
            text-align: left;
        }
        
        th {
            background-color: #28a745;
            color: white;
            font-weight: 600;
        }
        
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        .header-section {
            text-align: center;
            margin-bottom: 40px;
            padding: 30px;
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            border-radius: 10px;
        }
        
        .header-section h1 {
            color: white;
            border: none;
            margin-bottom: 10px;
        }
        
        .competition-info {
            background-color: #e8f5e8;
            border: 1px solid #28a745;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .tech-stack {
            background-color: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 20px;
            margin: 20px 0;
        }
        
        .performance-table {
            background-color: #fff;
            border: 2px solid #28a745;
            border-radius: 8px;
            overflow: hidden;
        }
        
        .performance-table th {
            background-color: #28a745;
        }
        
        .innovation-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        
        .contact-info {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            border-top: 2px solid #28a745;
            color: #666;
            font-size: 0.9em;
        }
        
        .emoji {
            font-size: 1.2em;
        }
        
        .highlight {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 4px;
            padding: 15px;
            margin: 15px 0;
        }
        
        .warning {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 4px;
            padding: 15px;
            margin: 15px 0;
        }
        
        .success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 4px;
            padding: 15px;
            margin: 15px 0;
        }
        
        .print-instructions {
            background-color: #e3f2fd;
            border: 1px solid #2196f3;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
        }
        
        .print-instructions h3 {
            color: #1976d2;
            margin-top: 0;
        }
        
        .print-instructions ul {
            text-align: left;
            display: inline-block;
        }
        
        @media print {
            body {
                padding: 20px;
                max-width: none;
            }
            
            .header-section {
                page-break-after: avoid;
            }
            
            h2 {
                page-break-after: avoid;
            }
            
            pre {
                page-break-inside: avoid;
            }
            
            table {
                page-break-inside: avoid;
            }
            
            .print-instructions {
                display: none;
            }
        }
        
        @media screen and (max-width: 768px) {
            body {
                padding: 20px;
            }
            
            h1 {
                font-size: 2em;
            }
            
            h2 {
                font-size: 1.5em;
            }
        }
    </style>
    """
    
    # Convertir Markdown en HTML
    html_content = markdown.markdown(
        md_content,
        extensions=[
            'markdown.extensions.tables',
            'markdown.extensions.fenced_code',
            'markdown.extensions.codehilite',
            'markdown.extensions.toc',
            'markdown.extensions.attr_list'
        ]
    )
    
    # Créer le document HTML complet
    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AgriLens AI - Technical Documentation</title>
        {css_style}
    </head>
    <body>
        <div class="header-section">
            <h1>🌱 AgriLens AI</h1>
            <h2>Technical Documentation & Competition Submission</h2>
            <p><strong>Intelligent Plant Disease Diagnosis System</strong></p>
        </div>
        
        <div class="print-instructions">
            <h3>📄 Instructions pour l'impression PDF</h3>
            <p>Pour créer un PDF de ce document :</p>
            <ul>
                <li><strong>Windows/Linux :</strong> Appuyez sur <kbd>Ctrl</kbd> + <kbd>P</kbd></li>
                <li><strong>Mac :</strong> Appuyez sur <kbd>Cmd</kbd> + <kbd>P</kbd></li>
                <li>Sélectionnez "Enregistrer en PDF" comme destination</li>
                <li>Choisissez le format A4 et les marges par défaut</li>
            </ul>
        </div>
        
        {html_content}
        
        <div class="footer">
            <p><strong>Technical Documentation Version: 3.0 | Competition Submission: January 2025</strong></p>
            <p><strong>Created by:</strong> Sidoine Kolaolé YEBADOKPO</p>
            <p><strong>Location:</strong> Bohicon, Republic of Benin</p>
            <p><strong>Contact:</strong> syebadokpo@gmail.com</p>
        </div>
    </body>
    </html>
    """
    
    return full_html

def main():
    """Fonction principale"""
    
    # Chemins des fichiers
    md_file = "TECHNICAL_NOTE.md"
    output_html = "AgriLens_AI_Technical_Documentation.html"
    
    # Vérifier que le fichier Markdown existe
    if not Path(md_file).exists():
        print(f"❌ Fichier {md_file} non trouvé!")
        sys.exit(1)
    
    print("🔄 Conversion de la note technique en cours...")
    
    try:
        # Étape 1: Convertir Markdown en HTML
        print("📝 Conversion Markdown → HTML...")
        html_content = create_html_from_markdown(md_file)
        
        # Étape 2: Sauvegarder le fichier HTML
        print("💾 Sauvegarde du fichier HTML...")
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Fichier HTML créé avec succès: {output_html}")
        print(f"📏 Taille: {Path(output_html).stat().st_size / 1024:.1f} KB")
        
        # Étape 3: Ouvrir dans le navigateur
        print("🌐 Ouverture dans le navigateur...")
        try:
            webbrowser.open(f'file://{os.path.abspath(output_html)}')
            print("📖 Document ouvert dans votre navigateur")
        except Exception as e:
            print(f"⚠️ Impossible d'ouvrir automatiquement: {e}")
            print(f"💡 Ouvrez manuellement le fichier: {output_html}")
        
        print("\n🎉 Conversion terminée!")
        print("📄 Pour créer un PDF:")
        print("   1. Dans votre navigateur, appuyez sur Ctrl+P (ou Cmd+P sur Mac)")
        print("   2. Sélectionnez 'Enregistrer en PDF'")
        print("   3. Choisissez le format A4")
        print("   4. Cliquez sur 'Enregistrer'")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 