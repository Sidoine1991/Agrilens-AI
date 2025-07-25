import streamlit as st
import os
import torch
# --- Mode hybride démo/réel pour Hugging Face Spaces ---
HF_TOKEN = os.environ.get('HF_TOKEN')
IS_DEMO = os.environ.get('HF_SPACE', False) and not HF_TOKEN

if IS_DEMO:
    st.markdown("""<div style='background:#ffe082; padding:1em; border-radius:8px; text-align:center; font-size:1.1em;'>
    ⚠️ <b>Version de démonstration Hugging Face</b> :<br>
    L’inférence réelle (modèle Gemma 3n) n’est pas disponible en ligne.<br>
    Pour un diagnostic complet, utilisez la version locale (offline) ou ajoutez un token HF valide.<br>
    </div>""", unsafe_allow_html=True)
    st.image('https://huggingface.co/datasets/mishig/sample_images/resolve/main/tomato_leaf_disease.jpg', width=300, caption='Exemple de feuille malade')
    st.markdown("""**Exemple de diagnostic généré (démo)** :  
    La feuille présente des taches brunes irrégulières, probablement dues à une maladie fongique.  
    1. Diagnostic précis : Mildiou (stade initial)  
    2. Agent pathogène : Phytophthora infestans  
    3. Mode d’infection : spores transportées par l’eau et le vent  
    4. Conseils : éliminer les feuilles infectées, traiter avec un fongicide à base de cuivre, surveiller l’humidité  
    5. Prévention : rotation des cultures, variétés résistantes  
    """)
    st.stop()

if HF_TOKEN:
    st.markdown("""<div style='background:#c8e6c9; padding:1em; border-radius:8px; text-align:center; font-size:1.1em;'>
    ✅ <b>Mode test réel</b> : Le modèle <code>google/gemma-3n-E2B-it</code> est chargé depuis Hugging Face.<br>
    L’inférence peut être lente selon la puissance du Space.<br>
    </div>""", unsafe_allow_html=True)
    from transformers import AutoProcessor, AutoModelForImageTextToText
    @st.cache_resource(show_spinner=True)
    def load_gemma_hf():
        processor = AutoProcessor.from_pretrained("google/gemma-3n-E2B-it", token=HF_TOKEN)
        model = AutoModelForImageTextToText.from_pretrained("google/gemma-3n-E2B-it", token=HF_TOKEN).to("cuda" if torch.cuda.is_available() else "cpu")
        return processor, model
    # Remplacer load_gemma_multimodal partout par load_gemma_hf
    load_gemma_multimodal = load_gemma_hf

st.set_page_config(
    page_title="AgriLens AI - Plant Disease Diagnosis",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import urllib.parse
import logging
from io import BytesIO
from fpdf import FPDF
import base64
import re
import requests # Added for URL image import
import urllib3  # Ajouté pour désactiver les warnings SSL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Style CSS personnalisé
st.markdown("""
<style>
    .main {
        max-width: 1000px;
        padding: 2rem;
    }
    .title {
        color: #2e8b57;
        text-align: center;
    }
    .upload-box {
        border: 2px dashed #2e8b57;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #2e8b57;
    }
</style>
""", unsafe_allow_html=True)

# === Textes multilingues ===
UI_TEXTS = {
    'fr': {
        'title': "AgriLens AI",
        'subtitle': "Diagnostic intelligent des maladies des plantes",
        'summary': "Un outil d'aide à la décision pour les producteurs agricoles",
        'desc': "Analysez une photo de plante malade et recevez un diagnostic structuré, des conseils pratiques et des recommandations adaptées à votre contexte.",
        'advantages': [
            "100% local, aucune donnée envoyée sur Internet",
            "Conseils adaptés à l'agriculture africaine",
            "Simple, rapide, accessible à tous"
        ],
        'step1': "<b>Étape 1 :</b> Uploadez une photo nette de la plante malade",
        'step2': "<b>Étape 2 :</b> (Optionnel) Ajoutez un contexte ou une question",
        'step3': "<b>Étape 3 :</b> Cliquez sur <b>Diagnostiquer</b>",
        'upload_label': "Photo de la plante malade",
        'context_label': "Contexte ou question (optionnel)",
        'diagnose_btn': "Diagnostiquer",
        'warn_no_img': "Veuillez d'abord uploader une image de plante malade.",
        'diag_in_progress': "Diagnostic en cours... (cela peut prendre jusqu'à 2 minutes sur CPU)",
        'diag_done': "✅ Diagnostic terminé !",
        'diag_title': "### 📋 Résultat du diagnostic :",
        'decision_help': "🧑‍🌾 Cet outil est une aide à la décision : analysez le diagnostic, adaptez les conseils à votre contexte, et consultez un expert local si besoin.",
        'share_whatsapp': "Partager sur WhatsApp",
        'share_facebook': "Partager sur Facebook",
        'copy_diag': "Copier le diagnostic",
        'new_diag': "Nouveau diagnostic",
        'prompt_debug': "🔍 Afficher le prompt utilisé (debug)",
        'copy_tip': "💡 Conseil : Sélectionnez et copiez le texte ci-dessus pour l'utiliser.",
        'no_result': "❌ Aucun résultat généré. Le modèle n'a pas produit de réponse.",
        'lang_select': "Langue / Language"
    },
    'en': {
        'title': "AgriLens AI",
        'subtitle': "Smart Plant Disease Diagnosis",
        'summary': "A decision support tool for farmers",
        'desc': "Analyze a photo of a diseased plant and receive a structured diagnosis, practical advice, and recommendations tailored to your context.",
        'advantages': [
            "100% local, no data sent online",
            "Advice adapted to African agriculture",
            "Simple, fast, accessible to all"
        ],
        'step1': "<b>Step 1:</b> Upload a clear photo of the diseased plant",
        'step2': "<b>Step 2:</b> (Optional) Add context or a question",
        'step3': "<b>Step 3:</b> Click <b>Diagnose</b>",
        'upload_label': "Photo of the diseased plant",
        'context_label': "Context or question (optional)",
        'diagnose_btn': "Diagnose",
        'warn_no_img': "Please upload a photo of the diseased plant first.",
        'diag_in_progress': "Diagnosis in progress... (this may take up to 2 minutes on CPU)",
        'diag_done': "✅ Diagnosis complete!",
        'diag_title': "### 📋 Diagnosis result:",
        'decision_help': "🧑‍🌾 This tool is a decision support: analyze the diagnosis, adapt the advice to your context, and consult a local expert if needed.",
        'share_whatsapp': "Share on WhatsApp",
        'share_facebook': "Share on Facebook",
        'copy_diag': "Copy diagnosis",
        'new_diag': "New diagnosis",
        'prompt_debug': "🔍 Show used prompt (debug)",
        'copy_tip': "💡 Tip: Select and copy the text above to use it.",
        'no_result': "❌ No result generated. The model did not produce a response.",
        'lang_select': "Langue / Language"
    }
}

# === Sélecteur de langue et configuration dans la sidebar ===
st.sidebar.markdown("## 🌍 Configuration")
language = st.sidebar.selectbox(UI_TEXTS['fr']['lang_select'], options=['fr', 'en'], format_func=lambda x: 'Français' if x=='fr' else 'English', key='lang_select_box')
T = UI_TEXTS[language]

# --- Sélecteur de culture ---
cultures = [
    ('tomate', 'Tomate'),
    ('mais', 'Maïs'),
    ('manioc', 'Manioc'),
    ('riz', 'Riz'),
    ('banane', 'Banane'),
    ('cacao', 'Cacao'),
    ('cafe', 'Café'),
    ('igname', 'Igname'),
    ('arachide', 'Arachide'),
    ('coton', 'Coton'),
    ('palmier', 'Palmier à huile'),
    ('ananas', 'Ananas'),
    ('sorgho', 'Sorgho'),
    ('mil', 'Mil'),
    ('patate', 'Patate douce'),
    ('autre', 'Autre')
]
culture = st.sidebar.selectbox('🌾 Culture concernée', options=[c[0] for c in cultures], format_func=lambda x: dict(cultures)[x], key='culture_select')

# --- Champ localisation ---
localisation = st.sidebar.text_input('📍 Localisation (région, pays, village...)', key='localisation_input')

# --- Bandeau d'accueil ---
st.markdown(f"""
<div style='background: linear-gradient(90deg, #2e8b57 0%, #a8e063 100%); padding: 2em 1em; border-radius: 16px; text-align: center;'>
    <span style='font-size:4em;'>🌱</span><br>
    <span style='font-size:2.5em; font-weight:bold; color:#fff;'>{T['title']}</span><br>
    <span style='font-size:1.3em; color:#f0f8ff;'>{T['subtitle']}</span>
</div>
""", unsafe_allow_html=True)
st.markdown(f"""
<div style='margin-top:2em; text-align:center;'>
    <b>{T['summary']}</b><br>
    <span style='color:#2e8b57;'>{T['desc']}</span>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<ul style='margin-top:2em; font-size:1.1em;'>
""" + ''.join([f"<li>{adv}</li>" for adv in T['advantages']]) + """
</ul>
""", unsafe_allow_html=True)

# --- Instructions étapes ---
st.markdown(f"""
<div style='margin-top:1em; text-align:center; font-size:1.2em;'>
{T['step1']}<br>
{T['step2']}<br>
{T['step3']}
</div>
""", unsafe_allow_html=True)

MODEL_PATH = "models/gemma-3n-transformers-gemma-3n-e2b-it-v1"

@st.cache_resource(show_spinner=True)
def load_gemma_multimodal():
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH).to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model

# Section Exemples (avant l'upload)
with st.expander('📸 Exemples (images et prompts)', expanded=False):
    st.markdown('**Exemple d’image de feuille malade :**')
    st.image('https://huggingface.co/datasets/mishig/sample_images/resolve/main/tomato_leaf_disease.jpg', width=300, caption='Feuille de tomate malade')
    st.markdown('**Prompt exemple :**')
    st.code("Ma culture de tomate présente des taches brunes sur les feuilles, surtout après la pluie. Que faire ?", language=None)
    st.markdown('Vous pouvez tester l’outil avec cette image et ce prompt.')

# Responsive: ajuster la largeur des colonnes sur mobile
if st.session_state.get('is_mobile', False):
    col1, col2 = st.columns([1,1])
else:
    col1, col2 = st.columns([2,1])

with col1:
    uploaded_images = st.file_uploader(
        T['upload_label'] + " (jusqu'à 4 images différentes : feuille, tige, fruit, racine...)",
        type=["jpg", "jpeg", "png"],
        key='img_upload',
        accept_multiple_files=True,
        help="Vous pouvez sélectionner jusqu'à 4 photos différentes de la même plante."
    )
    st.write("[DEBUG] HF_TOKEN détecté :", bool(HF_TOKEN))
    if uploaded_images:
        for idx, img in enumerate(uploaded_images):
            st.write(f"[DEBUG] Image {idx+1} : name={getattr(img, 'name', None)}, size={getattr(img, 'size', None)}, type={getattr(img, 'type', None)}")
    st.write("Images uploadées :", uploaded_images)
    if uploaded_images:
        if len(uploaded_images) > 4:
            st.warning("Vous ne pouvez uploader que 4 images maximum. Seules les 4 premières seront utilisées.")
            uploaded_images = uploaded_images[:4]
        for idx, img in enumerate(uploaded_images):
            st.image(img, width=180, caption=f"Image {idx+1}")

    # === Explication sur l'import d'image ===
    st.markdown('''
    <div style='background:#fffde7; border:1px solid #ffe082; border-radius:8px; padding:1em; margin-bottom:1em;'>
    <b>ℹ️ Import d'image&nbsp;:</b><br>
    - <b>Upload</b> : Cliquez sur le bouton ou glissez-déposez une image (formats supportés : jpg, png, webp, bmp, gif).<br>
    - <b>URL directe</b> : Lien qui finit par <code>.jpg</code>, <code>.png</code>, etc. (exemple : <a href='https://upload.wikimedia.org/wikipedia/commons/6/6e/Leaf_blight_of_rice.jpg' target='_blank'>exemple</a>)<br>
    - <b>Base64</b> : Collez le texte d'une image encodée (voir <a href='https://www.base64-image.de/' target='_blank'>outil de conversion</a>).<br>
    <b>Copier/coller d'image n'est pas supporté</b> (limite technique de Streamlit).<br>
    <b>En cas d'erreur 403 ou 404</b> : Essayez un autre navigateur, une autre image, ou l'import base64.<br>
    </div>
    ''', unsafe_allow_html=True)

    # --- Bouton exemple d'image testée ---
    col1, col2 = st.columns([3,1])
    with col2:
        if st.button("Exemple d'image testée", help="Remplit automatiquement une URL d'image valide pour tester l'import."):
            st.session_state['url_input'] = "https://upload.wikimedia.org/wikipedia/commons/6/6e/Leaf_blight_of_rice.jpg"

    # --- Import par URL ---
    image_url_input = st.text_input("URL de l'image (PNG/JPEG)", key="url_input")
    image_from_url = None
    if image_url_input:
        # Vérification d'URL directe d'image
        if not re.match(r"^https?://.*\\.(jpg|jpeg|png|webp|bmp|gif)$", image_url_input, re.IGNORECASE):
            st.error("❌ L'URL doit pointer directement vers une image (.jpg, .png, .webp, .bmp, .gif). Exemple : https://upload.wikimedia.org/wikipedia/commons/6/6e/Leaf_blight_of_rice.jpg")
        else:
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(image_url_input, headers=headers, timeout=10, verify=False)
                response.raise_for_status()
                image_from_url = Image.open(BytesIO(response.content)).convert("RGB")
            except requests.exceptions.HTTPError as e:
                if response.status_code == 403:
                    st.error("❌ Erreur 403 : L'accès à cette image est refusé (protection du site). Essayez une autre image ou l'import base64.")
                elif response.status_code == 404:
                    st.error("❌ Erreur 404 : L'image demandée n'existe pas à cette adresse. Vérifiez l'URL ou essayez l'exemple proposé.")
                else:
                    st.error(f"❌ Erreur lors du téléchargement de l'image : {e}")
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement de l'image depuis l'URL : {e}\nVérifiez que l'URL est correcte et accessible.")
                st.info("Exemple d'URL valide : https://upload.wikimedia.org/wikipedia/commons/6/6e/Leaf_blight_of_rice.jpg")

    # --- Import base64 (plus visible) ---
    st.markdown('''<b>Import par base64 (optionnel)</b> :<br>Collez ici le texte d'une image encodée en base64 (voir <a href='https://www.base64-image.de/' target='_blank'>outil de conversion</a>).''', unsafe_allow_html=True)
    image_base64_input = st.text_area("Image (base64)", key="base64_input", height=100)
    image_from_base64 = None
    if image_base64_input:
        try:
            image_data = base64.b64decode(image_base64_input)
            image_from_base64 = Image.open(BytesIO(image_data)).convert("RGB")
        except Exception as e:
            st.error(f"❌ Erreur lors du décodage base64 : {e}\nVérifiez que le texte est bien une image encodée.")
            st.info("Utilisez un outil comme https://www.base64-image.de/ pour convertir une image en base64.")

    # Ajoute les images alternatives à la liste des images à diagnostiquer
    if image_from_base64:
        if not uploaded_images:
            uploaded_images = []
        uploaded_images.append(image_from_base64)
    if image_from_url:
        if not uploaded_images:
            uploaded_images = []
        uploaded_images.append(image_from_url)

with col2:
    user_prompt = st.text_area(T['context_label'], "", key='context_area')

# --- Mode rapide dans la sidebar ---
st.sidebar.markdown('---')
fast_mode = st.sidebar.checkbox('⚡ Mode rapide (réponse courte)', value=False, help="Réduit le temps d'attente en limitant la longueur de la réponse.")
max_tokens = 256 if fast_mode else 512

def resize_image(img, max_size=1024):
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_size = (int(w*scale), int(h*scale))
        return img.resize(new_size, Image.LANCZOS)
    return img

def process_image_with_gemma_multimodal(images, user_prompt=None, language='fr', fast_mode=True, max_tokens=512, progress=None):
    processor, model = load_gemma_multimodal()
    # Adapter le contexte selon la culture et la localisation
    culture_label = dict(cultures)[culture]
    loc_str = f" à {localisation}" if localisation else ""
    if language == 'fr':
        default_prompt = (
            f"Vous êtes un expert en phytopathologie et vous conseillez un producteur de {culture_label}{loc_str}.\n"
            f"Voici {len(images)} image(s) de différentes parties de la plante : " + " ".join(["<image_soft_token>"]*len(images)) + "\n"
            "Analyse les images et structure ta réponse ainsi :\n"
            "1. Diagnostic précis (nom de la maladie, gravité, stade)\n"
            "2. Agent pathogène suspecté (nom scientifique et vulgarisé)\n"
            "3. Mode d'infection et de transmission (explication simple)\n"
            "4. Conseils pratiques pour le producteur :\n"
            "   - Mesures immédiates à prendre au champ\n"
            "   - Traitements recommandés (biologiques et chimiques, avec doses précises et mode d'application détaillé, en privilégiant toujours les doses recommandées par le fabricant ou l'expert local)\n"
            "   - Précautions à respecter (protection, délai avant récolte, etc.)\n"
            "5. Conseils de prévention pour la prochaine saison\n"
            "Sois synthétique, clair, adapte-toi à un producteur non spécialiste, et termine par un message d'encouragement."
        )
    else:
        default_prompt = (
            f"You are a plant disease expert advising a {culture_label} farmer{loc_str}.\n"
            f"Here are {len(images)} images of different parts of the plant: " + " ".join(["<image_soft_token>"]*len(images)) + "\n"
            "Analyze the images and structure your answer as follows:\n"
            "1. Precise diagnosis (disease name, severity, stage)\n"
            "2. Suspected pathogen (scientific and common name)\n"
            "3. Infection and transmission mode (simple explanation)\n"
            "4. Practical advice for the farmer:\n"
            "   - Immediate actions to take in the field\n"
            "   - Recommended treatments (biological and chemical, with precise doses and detailed application method, always favoring the doses recommended by the manufacturer or local expert)\n"
            "   - Precautions to follow (protection, pre-harvest interval, etc.)\n"
            "5. Prevention tips for the next season\n"
            "Be clear, concise, adapt your answer to a non-specialist farmer, and end with an encouraging message."
        )
    if user_prompt and user_prompt.strip():
        prompt = user_prompt.strip() + "\n" + default_prompt
    else:
        prompt = default_prompt
    if progress:
        progress.progress(10, text="🔎 Préparation de l'inférence...")
    with st.spinner(T['diag_in_progress']):
        inputs = processor(images=images, text=prompt, return_tensors="pt").to(model.device)
        if progress:
            progress.progress(30, text="🧠 Génération de la réponse...")
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        if progress:
            progress.progress(90, text="📝 Finalisation...")
        result = processor.decode(outputs[0], skip_special_tokens=True)
    return result, prompt

def clean_result(result, prompt):
    # Supprime le prompt recopié en début de réponse
    if result.strip().startswith(prompt.strip()[:40]):
        # On coupe tout ce qui précède le premier vrai diagnostic (ex: '1. Diagnostic' ou 'Diagnostic précis')
        m = re.search(r'(1\.\s*Diagnostic|Diagnostic précis|1\.\s*Precise diagnosis|Precise diagnosis)', result, re.IGNORECASE)
        if m:
            return result[m.start():].strip()
        else:
            # Sinon, on enlève juste le prompt
            return result[len(prompt):].strip()
    return result.strip()

def clean_for_pdf(text):
    # Enlève balises HTML et caractères non supportés par FPDF
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('\r', '').replace('\t', ' ')
    # FPDF ne supporte pas certains caractères Unicode : on remplace par ?
    text = ''.join(c if ord(c) < 128 or c in '\n\r' else '?' for c in text)
    # Tronque si trop long (FPDF limite ~10k caractères)
    if len(text) > 9000:
        text = text[:9000] + '\n... [Texte tronqué pour export PDF] ...'
    return text

# Historique des diagnostics (stocké en session)
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Mode expert dans la sidebar
expert_mode = st.sidebar.checkbox('🧑‍🔬 Mode expert', value=False, key='expert_mode')

# Section Ressources dans la sidebar
with st.sidebar.expander('📚 Ressources', expanded=False):
    st.markdown('''
- [Guide maladies du manioc (PDF)](https://www.fao.org/3/i3278f/i3278f.pdf)
- [Guide maladies du riz (PDF)](https://www.fao.org/3/y4751f/y4751f.pdf)
- [Vidéos YouTube - Diagnostic agricole](https://www.youtube.com/results?search_query=diagnostic+maladies+plantes)
- [Contact expert local](mailto:expert@agrilens.ai)
''')

# --- Alerte GPU ---
gpu_ok = torch.cuda.is_available()
gpu_name = torch.cuda.get_device_name(0) if gpu_ok else None
if gpu_ok:
    st.success(f"✅ Accélération GPU activée : {gpu_name}")
else:
    st.warning("⚠️ Le GPU n'est pas utilisé pour l'inférence. L'application sera plus rapide sur une machine équipée d'une carte NVIDIA compatible CUDA.")


if st.button(T['diagnose_btn'], type="primary", use_container_width=True):
    if not uploaded_images or len(uploaded_images) == 0:
        st.warning(T['warn_no_img'])
    else:
        try:
            # Juste avant le diagnostic (bouton ou logique d'appel à process_image_with_gemma_multimodal)
            def to_pil(img):
                if isinstance(img, Image.Image):
                    return img
                try:
                    return Image.open(img).convert("RGB")
                except Exception as e:
                    st.error(f"Impossible de lire l'image : {e}")
                    return None

            images_for_inference = [to_pil(img) for img in uploaded_images if img is not None]
            images_for_inference = [img for img in images_for_inference if img is not None]
            if not images_for_inference:
                st.error("❌ Aucune image valide à diagnostiquer.")
                st.stop()

            st.info(T['diag_in_progress'])
            progress = st.progress(0, text="⏳ Analyse en cours...")
            try:
                result, prompt_debug = process_image_with_gemma_multimodal(images_for_inference, user_prompt=user_prompt, language=language, fast_mode=fast_mode, max_tokens=max_tokens, progress=progress)
            except RuntimeError as e:
                logger.error(f"Erreur lors du chargement du modèle ou de l'inférence : {e}")
                st.error("❌ Le modèle n'a pas pu être chargé ou l'inférence a échoué. Vérifiez la mémoire disponible ou réessayez plus tard.")
                st.info("💡 Astuce : Fermez d'autres applications pour libérer de la RAM, ou redémarrez l'ordinateur.")
                st.stop()
            except Exception as e:
                logger.error(f"Erreur inattendue lors de l'inférence : {e}")
                st.error("❌ Une erreur inattendue est survenue lors de l'analyse. Veuillez réessayer ou contacter le support.")
                st.stop()
            progress.progress(100, text="✅ Analyse terminée")
            # Nettoyage du résultat pour ne pas afficher le prompt
            result_clean = clean_result(result, prompt_debug)
            if result_clean and result_clean.strip():
                st.session_state['history'].append({
                    'culture': dict(cultures)[culture],
                    'localisation': localisation,
                    'prompt': prompt_debug,
                    'result': result_clean
                })
                st.success(T['diag_done'])
                st.markdown(T['diag_title'])
                st.markdown(result_clean)
                st.info(T['decision_help'])
                share_text = urllib.parse.quote(f"Diagnostic AgriLens AI :\n{result_clean}")
                st.sidebar.markdown(f"""
                <div style='margin-top:1em; display:flex; flex-direction:column; gap:0.7em;'>
                    <a href='https://wa.me/?text={share_text}' target='_blank' style='background:#25D366; color:#fff; padding:0.5em 1.2em; border-radius:6px; text-decoration:none; font-weight:bold; display:block; text-align:center;'>{T['share_whatsapp']}</a>
                    <a href='https://www.facebook.com/sharer/sharer.php?u=&quote={share_text}' target='_blank' style='background:#4267B2; color:#fff; padding:0.5em 1.2em; border-radius:6px; text-decoration:none; font-weight:bold; display:block; text-align:center;'>{T['share_facebook']}</a>
                    <button onclick=\"navigator.clipboard.writeText(decodeURIComponent('{share_text}'))\" style='background:#2e8b57; color:#fff; padding:0.5em 1.2em; border:none; border-radius:6px; font-weight:bold; cursor:pointer; width:100%;'>{T['copy_diag']}</button>
                    <button onclick=\"window.location.reload();\" style='background:#a8e063; color:#2e8b57; font-size:1.1em; padding:0.6em 2em; border:none; border-radius:8px; cursor:pointer; width:100%; margin-top:0.7em;'>{T['new_diag']}</button>
                </div>
                """, unsafe_allow_html=True)
                # PDF robuste avec nettoyage
                def create_pdf(text):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("helvetica", size=12)  # Remplacement Arial -> helvetica
                    for line in text.split('\n'):
                        pdf.multi_cell(0, 10, line)
                    pdf_bytes = BytesIO()
                    pdf.output(pdf_bytes)
                    pdf_bytes.seek(0)
                    return pdf_bytes.read()
                try:
                    pdf_data = create_pdf(clean_for_pdf(result_clean))
                    st.download_button(
                        label="⬇️ Télécharger le diagnostic en PDF",
                        data=pdf_data,
                        file_name="diagnostic_agri.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    logger.error(f"Erreur lors de la génération du PDF : {e}")
                    st.error(f"❌ L'export PDF a échoué. Erreur : {e}\nVeuillez réessayer ou contacter le support.")
                # Mode expert, etc. inchangés
                if expert_mode:
                    st.markdown('---')
                    st.markdown('**Prompt complet envoyé au modèle :**')
                    st.code(prompt_debug, language=None)
                    st.markdown('**Annotation / Correction :**')
                    st.text_area('Ajouter une note ou une correction (optionnel)', key=f'annot_{len(st.session_state["history"])}')
                st.info(T['copy_tip'])
            else:
                st.error(T['no_result'])
                st.info("💡 Astuce : Essayez une photo plus nette ou un autre angle de la plante.")
                # Afficher le bouton PDF désactivé si pas de diagnostic
                st.download_button(
                    label="⬇️ Télécharger le diagnostic en PDF",
                    data=b"",
                    file_name="diagnostic_agri.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    disabled=True,
                    help="Générez d'abord un diagnostic pour activer l'export PDF."
                )
        except Exception as e:
            logger.error(f"Erreur critique : {e}")
            st.error("❌ Une erreur critique est survenue. Veuillez réessayer ou contacter le support technique.")

# --- Historique des diagnostics ---
with st.expander('🗂️ Historique des diagnostics', expanded=False):
    if st.session_state['history']:
        import pandas as pd
        hist_df = pd.DataFrame(st.session_state['history'])
        st.dataframe(hist_df[['culture', 'localisation', 'result']], use_container_width=True)
        csv = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button('⬇️ Télécharger l’historique (CSV)', data=csv, file_name='historique_diagnostics.csv', mime='text/csv')
    else:
        st.info('Aucun diagnostic enregistré pour le moment.')

st.divider()

# Sidebar : bouton Aide/FAQ
with st.sidebar.expander('❓ Aide / FAQ', expanded=False):
    st.markdown('''
- **Comment obtenir un bon diagnostic ?**
    - Prenez une photo nette, bien éclairée, sans flou ni reflets.
    - Décrivez le contexte (culture, symptômes, conditions météo).
- **Le diagnostic ne correspond pas ?**
    - Essayez une autre photo ou reformulez votre question.
- **Problème technique ?**
    - Redémarrez l’application ou contactez le support :
    - 📧 [support@agrilens.ai](mailto:support@agrilens.ai)
    ''')

# --- Pied de page / Footer ---
st.markdown("""
<hr style='margin-top:2em; margin-bottom:0.5em; border:1px solid #e0e0e0;'>
<div style='text-align:center; font-size:1em; color:#888;'>
    <b>AgriLens AI</b> – © 2024 Sidoine YEBADOKPO<br>
    Expert en analyse de données, Développeur Web<br>
    <a href='mailto:syebadokpo@gmail.com'>syebadokpo@gmail.com</a> · <a href='https://linkedin.com/in/sidoineko' target='_blank'>LinkedIn</a> · <a href='https://huggingface.co/Sidoineko/portfolio' target='_blank'>Hugging Face</a><br>
    <span style='font-size:0.95em;'>🇫🇷 Application créée par Sidoine YEBADOKPO | 🇬🇧 App created by Sidoine YEBADOKPO</span>
</div>
""", unsafe_allow_html=True)
