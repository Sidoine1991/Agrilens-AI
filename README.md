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

# 🌱 AgriLens AI - Plant Disease Diagnosis

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow.svg)](https://huggingface.co/spaces)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](https://huggingface.co/spaces/sido1991/Agrilens_IAv1)

## 🎯 Overview

AgriLens AI is an innovative **plant disease diagnosis application** using artificial intelligence to help farmers identify and treat problems in their crops. This first version was specifically developed to participate in the **Kaggle competition** and represents our expertise in AI applied to agriculture.

### 🌟 Key Features

- **📸 Image Analysis** : Visual disease diagnosis using AI
- **💬 Text Analysis** : Advice based on symptom descriptions
- **🌐 Multilingual Support** : French and English interfaces
- **📱 Mobile Responsive** : Optimized for smartphones and tablets
- **🔧 Practical Recommendations** : Concrete actions to take
- **⚡ Real-time Processing** : Fast AI-powered analysis
- **🧠 Advanced AI** : Gemini API integration for precise diagnosis
- **🔄 Adaptive Model Loading** : Smart model selection based on available resources
- **💾 Memory Optimization** : Efficient resource management for deployment

## 🚀 Live Demo

### 🌐 Online Version
- **Hugging Face Spaces** : [AgriLens AI Demo](https://huggingface.co/spaces/sido1991/Agrilens_IAv1)
- **Status** : ✅ Fully functional with multilingual support and Gemini AI
- **Performance** : Optimized for 16GB RAM environments

### 📱 Mobile Access
- Open the demo link on your smartphone
- Interface automatically adapts to mobile screens
- Touch-friendly controls and navigation

## 🏗️ Architecture

### System Design
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend        │    │   AI Models     │
│   (Streamlit)   │◄──►│   (Python)       │◄──►│   (Hugging Face)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │    │   Model Manager  │    │   Gemini API    │
│   (Image/Text)  │    │   (Adaptive)     │    │   (Enhancement) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Model Architecture
- **Primary Model** : Gemma 3B IT (Google) - Lightweight multimodal model
- **Fallback Models** : DialoGPT-medium, DistilBERT for resource-constrained environments
- **Enhancement** : Gemini 1.5 Flash API for improved diagnosis interpretation
- **Processing Pipeline** : Adaptive model loading based on available memory

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- Modern web browser
- Internet connection (for model loading)
- Google API Key (optional, for enhanced diagnosis)
- Minimum 4GB RAM (8GB+ recommended)

### Local Installation

```bash
# Clone the repository
git clone https://github.com/Sidoineko/AgriLensAI.git
cd AgriLensAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (optional)
echo "GOOGLE_API_KEY=your_api_key_here" > .env

# Run the application
streamlit run src/streamlit_app_multilingual.py
```

### Docker Installation

```bash
# Build the Docker image
docker build -t agrilens-ai .

# Run the container
docker run -p 8501:7860 agrilens-ai
```

### Quick Start Scripts
```bash
# Windows
lancer_app_local.bat

# Linux/Mac
./start.sh
```

## 📖 User Manuals

### 📚 Complete Documentation
- **[French Manual](docs/user_manual_fr.md)** : Manuel utilisateur complet en français
- **[English Manual](docs/user_manual_en.md)** : Complete user manual in English

### 🎯 Quick Start Guide

1. **Load the AI Model** : Click "Load AI Model" in settings
2. **Choose Language** : Select French or English
3. **Upload Image** : Take a photo of the diseased plant
4. **Get Diagnosis** : Receive AI-powered analysis and recommendations

## 🔬 Technology Stack

### Core Technologies
- **Framework** : [Streamlit](https://streamlit.io/) - Web application framework
- **AI Models** : 
  - [Gemma 3B IT](https://huggingface.co/google/gemma-3b-it) - Primary multimodal model
  - [DialoGPT-medium](https://huggingface.co/microsoft/DialoGPT-medium) - Lightweight fallback
  - [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) - Ultra-lightweight option
- **Advanced AI** : [Gemini API](https://ai.google.dev/) - Google's advanced AI for precise diagnosis
- **Deployment** : [Hugging Face Spaces](https://huggingface.co/spaces) - Cloud hosting platform
- **Languages** : Python, HTML, CSS

### Key Libraries
- **Transformers** : Hugging Face's AI model library
- **Google Generative AI** : Gemini API integration
- **PyTorch** : Deep learning framework
- **Pillow** : Image processing
- **Streamlit** : Web interface
- **Torch** : Tensor operations and model inference

## ⚙️ Model Configuration

### Adaptive Model Loading
The application automatically selects the best model based on available resources:

```python
# Memory-constrained environments (HF Spaces)
load_ultra_lightweight_for_hf_spaces()  # DistilBERT

# Standard environments
load_basic_pipeline()                   # Pipeline approach

# High-memory environments
load_gemma_full()                       # Full Gemma 3n E4B IT

# Conservative approach
load_conservative()                     # Gemma with CPU optimization
```

### Model Specifications
| Model | Size | Memory Usage | Use Case |
|-------|------|--------------|----------|
| Gemma 3n E4B IT | ~8GB | 16GB+ | Full-featured analysis |
| Gemma 3B IT | ~2GB | 4-8GB | Standard deployment |
| DialoGPT-medium | ~500MB | 2-4GB | Lightweight deployment |
| DistilBERT | ~250MB | 1-2GB | Ultra-lightweight |

## 🎯 Performance Metrics

### Accuracy & Speed
- **Image Recognition** : High accuracy in disease identification
- **Diagnostic Precision** : Enhanced with Gemini AI integration
- **Response Time** : < 30 seconds for complete analysis
- **Memory Efficiency** : Adaptive loading prevents OOM errors

### Supported Plant Types
- **Vegetables** : Tomatoes, peppers, cucumbers, lettuce, carrots
- **Fruits** : Apples, grapes, citrus, berries, stone fruits
- **Grains** : Corn, wheat, rice, barley, oats
- **Ornamentals** : Roses, flowers, shrubs, trees
- **Herbs** : Basil, mint, rosemary, thyme

### Disease Categories
- **Fungal Diseases** : Powdery mildew, rust, blight
- **Bacterial Diseases** : Bacterial spot, canker, wilt
- **Viral Diseases** : Mosaic virus, leaf curl
- **Nutritional Deficiencies** : Nitrogen, phosphorus, potassium
- **Environmental Stress** : Drought, heat, cold damage

## 🔧 Advanced Features

### Memory Management
- **Dynamic Model Loading** : Loads appropriate model based on available RAM
- **Cache Management** : Efficient model persistence and restoration
- **Resource Monitoring** : Real-time memory usage tracking

### Error Handling
- **Graceful Degradation** : Falls back to lighter models on memory issues
- **Timeout Protection** : Prevents infinite loading with signal handlers
- **User Feedback** : Clear error messages and recovery suggestions

### Multilingual Support
- **French Interface** : Complete localization
- **English Interface** : Full English support
- **Dynamic Translation** : Context-aware language switching

## 👨‍💻 Creator Information

### **Sidoine Kolaolé YEBADOKPO**
- **Location** : Bohicon, Republic of Benin
- **Phone** : +229 01 96 91 13 46
- **Email** : syebadokpo@gmail.com
- **LinkedIn** : [linkedin.com/in/sidoineko](https://linkedin.com/in/sidoineko)
- **Hugging Face Portfolio** : [Sidoineko/portfolio](https://huggingface.co/Sidoineko/portfolio)

### 🏆 Competition Version
This first version of AgriLens AI was specifically developed to participate in the **Kaggle competition**. It represents our initial public production and demonstrates our expertise in AI applied to agriculture.

## ⚙️ Configuration

### Environment Variables
```bash
# Required for enhanced diagnosis
GOOGLE_API_KEY=your_google_api_key_here

# Optional Hugging Face token
HF_TOKEN=your_huggingface_token_here

# Optional: Force specific model loading strategy
FORCE_MODEL_STRATEGY=ultra_lightweight
```

### Model Configuration
- **Primary Model** : Gemma 3B IT (Google)
- **Enhanced AI** : Gemini 1.5 Flash (Google)
- **Processing** : CPU optimized for deployment
- **Memory Management** : Adaptive based on environment

## 🗺️ Roadmap

### Version 1.0 (Current) ✅
- ✅ Basic image analysis
- ✅ Text-based diagnosis
- ✅ Multilingual support
- ✅ Mobile responsiveness
- ✅ Gemini AI integration
- ✅ Adaptive model loading
- ✅ Memory optimization
- ✅ Error handling

### Version 1.1 (In Progress) 🔄
- 🔄 Real-time video analysis
- 🔄 Offline mode support
- 🔄 More plant species
- 🔄 Community features
- 🔄 Expert consultation system

### Future Versions 🚀
- 🚀 Edge deployment support
- 🚀 IoT integration
- 🚀 Advanced analytics dashboard
- 🚀 API endpoints for third-party integration

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements_dev.txt

# Run tests
python -m pytest tests/

# Format code
black src/
isort src/
```

## 📄 License

This project is developed for the **Google - Gemma 3n Hackathon** and is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

**CC BY 4.0 License** : You are free to share and adapt this material for any purpose, even commercially, provided you give appropriate credit.

- **Attribution** : Sidoine Kolaolé YEBADOKPO
- **License Link** : https://creativecommons.org/licenses/by/4.0/
- **Competition Compliance** : This license is required for participation in the Google - Gemma 3n Hackathon

See the [LICENSE](LICENSE) file for full details.

## ⚠️ Disclaimer

This application is designed to assist farmers and gardeners in identifying plant diseases. While we strive for accuracy, the diagnosis should not replace professional agricultural advice. Always consult with local agricultural experts for critical decisions.

## 🐛 Known Issues & Solutions

### Memory Issues on HF Spaces
- **Problem** : Model loading fails due to 16GB RAM limit
- **Solution** : Application automatically uses ultra-lightweight models
- **Workaround** : Manual model selection in settings

### Model Loading Timeouts
- **Problem** : Long loading times on slow connections
- **Solution** : Implemented timeout protection with fallback models
- **Workaround** : Use local deployment for faster loading

## 🙏 Acknowledgments

- **Google** : For providing Gemma and Gemini AI models
- **Hugging Face** : For the deployment platform and transformers library
- **Streamlit** : For the web framework
- **Kaggle** : For hosting the competition that inspired this project
- **Open Source Community** : For the amazing tools and libraries

## 📊 Project Statistics

- **Lines of Code** : ~2,000+
- **Models Supported** : 4 different AI models
- **Languages** : 2 (French, English)
- **Deployment Platforms** : 3 (Local, Docker, HF Spaces)
- **Test Coverage** : Comprehensive error handling

---

*AgriLens AI - Intelligent plant disease diagnosis with AI* 🌱

**Last Updated** : July 2024  
**Version** : 1.0.0  
**Status** : Production Ready 