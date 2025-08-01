---
title: AgriLens AI - Plant Disease Diagnosis
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: docker
sdk_version: "1.0.0"
app_file: src/streamlit_app_multilingual.py
pinned: false
license: mit
---

# 🌱 AgriLens AI

**Plant disease diagnosis for farmers using Google's Gemma 3n AI**

![AgriLens AI Logo](logo_app/logo_agrilesai.png)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-green.svg)](LICENSE)

## What is AgriLens AI?

AgriLens AI helps farmers identify plant diseases instantly using their smartphone camera. No internet needed after setup - works completely offline in the field.

**Key Features:**
- 📸 **Photo Analysis**: Take a photo, get instant diagnosis
- 🌐 **Bilingual**: French and English
- 📱 **Mobile-Friendly**: Works on any smartphone
- 💾 **Offline**: No internet required after initial setup
- 📄 **Export**: Save results as HTML or text

## 🚀 Try It Now

**Live Demo**: [AgriLens AI on Hugging Face](https://huggingface.co/spaces/sido1991/Agrilens_IAv1)

Open the link on your phone - the interface adapts automatically.

## How It Works

1. **Take Photo** of the diseased plant
2. **Specify Crop** (optional but recommended)
3. **Get Diagnosis** with treatment recommendations
4. **Export Results** if needed

## 📸 Examples

### Sample Plant Images
Here are examples of diseased plants you can analyze:

| **Maize Disease** | **Cassava Mosaic** | **Tomato Early Blight** |
|:---:|:---:|:---:|
| ![Maize Disease](sample%20image/mais_malade.jpg) | ![Cassava Mosaic](sample%20image/mosaique_manioc.jpg) | ![Tomato Early Blight](sample%20image/tomato_early_blight.jpg) |

### Demo Screenshots
See what the diagnosis results look like:

| **Interface Overview** | **Image Upload** | **Results Output** | **Mobile Mode** |
|:---:|:---:|:---:|:---:|
| ![Interface Overview](screenshots/Interface_overview.png) | ![Image Upload](screenshots/image_uploding.jpeg) | ![Results Output](screenshots/result_output.jpeg) | ![Mobile Mode](screenshots/mobile_mode1.jpg) |

## 🏗️ Architecture

![AgriLens AI Architecture](architecture_app.png)

**Technology Stack:**
- **AI Model**: Google Gemma 3n (multimodal)
- **Framework**: Streamlit
- **Languages**: Python, French, English

## 🌾 Real-World Usage

### The "Farm Laptop" Approach

AgriLens AI is designed for **offline use** in rural areas:

1. **Setup in Town**: Download once where internet is available
2. **Deploy to Farm**: Bring laptop to farm - no internet needed
3. **Daily Use**: Farmers take photos, transfer to laptop, get instant diagnosis

### Why This Works

- **70%+ of farmers** have smartphones
- **One laptop** serves entire community
- **No ongoing costs** - completely free
- **Instant results** vs weeks waiting for experts

### Deployment Options

**Option 1: Community Laptop**
- One laptop per village/cooperative
- Shared resource for all farmers
- Setup time: 30 minutes

**Option 2: Extension Workers**
- Technicians carry pre-loaded laptops
- Visit farms with diagnostic capability
- Professional on-site analysis

**Option 3: Individual Setup**
- Farmers with basic tech skills
- Personal diagnostic tool
- Complete independence

## 📊 Comparison

| Feature | AgriLens AI | Traditional Methods |
|---------|-------------|-------------------|
| **Cost** | Free | Expensive consultation |
| **Speed** | Instant | Days/weeks wait |
| **Availability** | 24/7 | Limited expert hours |
| **Language** | French + English | Often language barriers |
| **Internet** | Only for setup | Not required |

## 🔄 Usage Workflow

### Complete Flow Diagram

```mermaid
flowchart TD
    A[🚀 Start] --> B{📱 Device Type}
    B -->|💻 Laptop/Desktop| C[🖥️ Local Installation]
    B -->|📱 Mobile| D[🌐 Web Access]
    
    C --> E[📥 Clone Repository]
    E --> F[🐍 Install Python 3.11+]
    F --> G[📦 Install Dependencies]
    G --> H[🤖 Download AI Model]
    H --> I[⚡ Launch Application]
    
    D --> J[🔗 Open Hugging Face Link]
    J --> K[📱 Mobile Interface]
    
    I --> L[📸 Take/Upload Photo]
    K --> L
    L --> M[🌾 Specify Crop Type]
    M --> N[🔍 AI Analysis]
    N --> O[📋 Diagnosis + Recommendations]
    O --> P[💾 Export Results]
    P --> Q[✅ Diagnosis Complete]
    
    style A fill:#e1f5fe
    style Q fill:#c8e6c9
    style N fill:#fff3e0
    style O fill:#f3e5f5
```

### ⚠️ Important Tips

**🖥️ Sleep Prevention**

To avoid disruptions to the AI model during analysis, it is **strongly recommended** to:

#### On Laptop/Desktop:
- **Disable sleep mode** in system settings
- **Increase sleep delay** to at least 10 minutes
- **Disable screen saver** during use
- **Keep power connected** if possible

#### On Mobile:
- **Increase screen brightness**
- **Disable auto-rotation** of the screen
- **Close other applications** to save battery
- **Use "Do Not Disturb" mode** to avoid interruptions

#### Why This is Important:
- The AI model requires **2-20 minutes** for complete analysis (depending on RAM)
- Sleep mode can **interrupt the process** and corrupt results
- **System stability** ensures accurate diagnostics
- **Patience is crucial** - do not interrupt the process even if it seems slow

## 🛠️ Installation

### Quick Start

```bash
# Clone and setup
git clone https://github.com/Sidoine1991/Agrilens-AI.git
cd AgriLens-AI
pip install -r requirements.txt

# Run (requires internet for first model download)
streamlit run src/streamlit_app_multilingual.py
```

### 📥 Model Download for Offline Use

**Important**: For true offline functionality, you need to download the complete model files locally.

#### Model Information
- **Model**: `google/gemma-3n-E4B-it`
- **Size**: ~10GB+ (complete model files)
- **Location**: Hugging Face Hub

#### Download Methods

**Method 1: Automatic Download (First Run)**
```bash
# The app will download automatically on first run
streamlit run src/streamlit_app_multilingual.py
# This downloads ~10GB to your local cache
```

**Method 2: Manual Download**
```bash
# Download model files manually
python -c "
from transformers import AutoProcessor, AutoModelForCausalLM
model_name = 'google/gemma-3n-E4B-it'
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print('Model downloaded successfully!')
"
```

**Method 3: From Kaggle Notebook**
- Use the [Kaggle Notebook](https://www.kaggle.com/code/sidoineyebadokpo/agrilens-ai?scriptVersionId=253640926)
- Download model files from Kaggle environment
- Transfer to local machine

### 🚨 Critical First-Time Setup Process

**Important**: For offline-first usage, you need to download the complete model locally. This is a **one-time critical process** that requires stable internet connection.

#### Step-by-Step Process:

1. **Create Hugging Face Account**
   - Go to [Hugging Face](https://huggingface.co/join)
   - Create an account and verify your email
   - This step is mandatory for model access

2. **Access the Model**
   - Visit: `https://huggingface.co/google/gemma-3n-E4B-it`
   - Accept the model terms and conditions
   - This grants you download permissions

3. **Download Model Files**
   - **Model Size**: ~10GB (complete files)
   - **Critical**: Ensure stable internet connection
   - If download fails, you must restart from beginning
   - Download all model files individually if needed

4. **Organize Files**
   - Create folder: `model_gemma`
   - Place all downloaded files in this folder
   - Update path in code: `LOCAL_MODEL_PATH = "D:/Dev/model_gemma"`

#### ⚠️ Critical Requirements:

- **Stable Internet**: 10GB download requires reliable connection
- **Sufficient Storage**: 15GB+ free space recommended
- **Patience**: Download may take 30-60 minutes depending on connection
- **No Interruption**: Avoid system sleep or network disconnection

#### 🆚 Demo vs Offline Usage:

| Feature | Hugging Face Demo | Local Offline Setup |
|---------|------------------|-------------------|
| **Internet** | Required | Only for initial download |
| **Speed** | Depends on server | Instant local processing |
| **Reliability** | Subject to outages | Always available |
| **Setup Time** | Instant | 30-60 minutes one-time |
| **Model Access** | Pre-loaded | Downloaded locally |

#### 🔄 Alternative: Kaggle Download
If Hugging Face download fails:
1. Use our [Kaggle Notebook](https://www.kaggle.com/code/sidoineyebadokpo/agrilens-ai)
2. Download model from Kaggle environment
3. Transfer files to local `model_gemma` folder

#### Offline Setup Complete
Once downloaded, the model files are cached locally and the app works completely offline. No internet connection needed for diagnosis.

### Requirements
- Python 3.11+
- 8GB+ RAM (16GB recommended)
- 15GB+ free disk space (for model files)
- Internet connection (first time only)

### Docker

```bash
docker build -t agrilens-ai .
docker run -p 8501:7860 agrilens-ai
```

## 🎯 Performance

### Response Time by Hardware Configuration

| RAM Configuration | Expected Response Time | Notes |
|------------------|----------------------|-------|
| **16GB+ RAM** | < 30 seconds | Optimal performance |
| **8-12GB RAM** | 1-3 minutes | Good performance |
| **4-8GB RAM** | 5-10 minutes | Acceptable performance |
| **< 4GB RAM** | 10-20 minutes | **Maximum wait time** |

### ⚠️ Important Performance Notes

- **RAM is Critical**: The AI model requires significant memory for processing
- **First Run**: Initial model loading may take longer on all systems
- **Background Processes**: Close other applications to free up RAM
- **Patience Required**: On low-RAM devices, the diagnostic process can take up to **20 minutes maximum**
- **No Interruption**: Do not close the application during analysis, even if it seems slow

### Accuracy & Capabilities
- **Accuracy**: High precision with Gemma 3n
- **Memory**: Adaptive loading for different hardware
- **Supported Plants**: Vegetables, fruits, grains, ornamentals, herbs

## 🔧 Advanced Features

- **Memory Management**: Automatic optimization
- **Error Handling**: Graceful fallbacks
- **Export**: HTML and text reports
- **Mobile Mode**: Simulated offline interface

## 👨‍💻 Creator

**Sidoine Kolaolé YEBADOKPO**
- Location: Bohicon, Republic of Benin
- Email: syebadokpo@gmail.com
- [LinkedIn](https://linkedin.com/in/sidoineko) | [Portfolio](https://huggingface.co/spaces/Sidoineko/portfolio)

## 📄 License

**CC BY 4.0** - Free to use, modify, and distribute with attribution.

## 🔗 Links

- **Demo**: [Hugging Face Spaces](https://huggingface.co/spaces/sido1991/Agrilens_IAv1)
- **Notebook**: [Kaggle](https://www.kaggle.com/code/sidoineyebadokpo/agrilens-ai?scriptVersionId=253640926)
- **Code**: [GitHub](https://github.com/Sidoine1991/Agrilens-AI)

---

*AgriLens AI - Empowering farmers with AI-powered plant disease diagnosis* 🌱

**Version**: 1.0.0 | **Updated**: July 2025 