---
title: AgriLens AI - Plant Disease Diagnosis
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: docker
sdk_version: "1.0.0"
app_file: app.py
pinned: false
license: mit
---

# 🌱 AgriLens AI - Plant Disease Diagnosis

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow.svg)](https://huggingface.co/spaces)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

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

## 🚀 Live Demo

### 🌐 Online Version
- **Hugging Face Spaces** : [AgriLens AI Demo](https://huggingface.co/spaces/Sidoineko/AgriLensAI)
- **Status** : ✅ Fully functional with multilingual support and Gemini AI

### 📱 Mobile Access
- Open the demo link on your smartphone
- Interface automatically adapts to mobile screens
- Touch-friendly controls and navigation

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- Modern web browser
- Internet connection (for model loading)
- Google API Key (optional, for enhanced diagnosis)

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

## 📖 User Manuals

### 📚 Complete Documentation
- **[French Manual](docs/user_manual_fr.md)** : Manuel utilisateur complet en français
- **[English Manual](docs/user_manual_en.md)** : Complete user manual in English

### 🎯 Quick Start Guide

1. **Load the AI Model** : Click "Load Gemma 2B Model" in settings
2. **Choose Language** : Select French or English
3. **Upload Image** : Take a photo of the diseased plant
4. **Get Diagnosis** : Receive AI-powered analysis and recommendations

## 🔬 Technology Stack

### Core Technologies
- **Framework** : [Streamlit](https://streamlit.io/) - Web application framework
- **AI Model** : [Gemma 2B](https://huggingface.co/google/gemma-2b-it) - Google's lightweight language model
- **Advanced AI** : [Gemini API](https://ai.google.dev/) - Google's advanced AI for precise diagnosis
- **Deployment** : [Hugging Face Spaces](https://huggingface.co/spaces) - Cloud hosting platform
- **Languages** : Python, HTML, CSS

### Key Libraries
- **Transformers** : Hugging Face's AI model library
- **Google Generative AI** : Gemini API integration
- **PyTorch** : Deep learning framework
- **Pillow** : Image processing
- **Streamlit** : Web interface

## 🎯 Performance Metrics

### Accuracy
- **Image Recognition** : High accuracy in disease identification
- **Diagnostic Precision** : Enhanced with Gemini AI integration
- **Response Time** : < 30 seconds for complete analysis

### Supported Plant Types
- **Vegetables** : Tomatoes, peppers, cucumbers, lettuce
- **Fruits** : Apples, grapes, citrus
- **Grains** : Corn, wheat, rice
- **Ornamentals** : Roses, flowers, shrubs

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
```

### Model Configuration
- **Primary Model** : Gemma 2B (Google)
- **Enhanced AI** : Gemini 1.5 Flash (Google)
- **Processing** : CPU optimized for deployment

## 🗺️ Roadmap

### Version 1.0 (Current)
- ✅ Basic image analysis
- ✅ Text-based diagnosis
- ✅ Multilingual support
- ✅ Mobile responsiveness
- ✅ Gemini AI integration

### Future Versions
- 🔄 Real-time video analysis
- 🔄 Offline mode support
- 🔄 More plant species
- 🔄 Community features
- 🔄 Expert consultation system

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### How to Contribute
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This application is designed to assist farmers and gardeners in identifying plant diseases. While we strive for accuracy, the diagnosis should not replace professional agricultural advice. Always consult with local agricultural experts for critical decisions.

## 🙏 Acknowledgments

- **Google** : For providing Gemma and Gemini AI models
- **Hugging Face** : For the deployment platform and transformers library
- **Streamlit** : For the web framework
- **Kaggle** : For hosting the competition that inspired this project

---

*AgriLens AI - Intelligent plant disease diagnosis with AI* 🌱