---
title: AgriLens AI
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: docker
sdk_version: "1.0.0"
app_file: src/streamlit_app_local_models.py
pinned: false
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

## 🚀 Live Demo

### 🌐 Online Version
- **Hugging Face Spaces** : [AgriLens AI Demo](https://huggingface.co/spaces/Sidoineko/AgriLensAI)
- **Status** : ✅ Fully functional with multilingual support

### 📱 Mobile Access
- Open the demo link on your smartphone
- Interface automatically adapts to mobile screens
- Touch-friendly controls and navigation

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- Modern web browser
- Internet connection (for model loading)

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
- **Deployment** : [Hugging Face Spaces](https://huggingface.co/spaces) - Cloud hosting platform
- **Languages** : Python, HTML, CSS

### Key Libraries
- **Transformers** : Hugging Face's AI model library
- **PyTorch** : Deep learning framework
- **Pillow** : Image processing
- **Streamlit** : Web interface

## 🎨 Features in Detail

### 📸 Image Analysis
- **Supported Formats** : PNG, JPG, JPEG
- **Quality Optimization** : Automatic image processing
- **Smart Cropping** : Focus on diseased areas
- **Batch Processing** : Multiple images support

### 💬 Text Analysis
- **Symptom Description** : Natural language input
- **Context Awareness** : Plant type and conditions
- **Progressive Analysis** : Step-by-step diagnosis
- **Preventive Advice** : Long-term care recommendations

### 🌐 Multilingual Interface
- **French** : Interface complète en français
- **English** : Complete English interface
- **Dynamic Switching** : Real-time language change
- **Localized Content** : Culturally adapted responses

## 📊 Performance

### Model Performance
- **Loading Time** : ~30 seconds (first time)
- **Analysis Speed** : 2-5 seconds per image
- **Accuracy** : High precision for common diseases
- **Memory Usage** : Optimized for cloud deployment

### System Requirements
- **RAM** : 4GB minimum (8GB recommended)
- **Storage** : 2GB for model and dependencies
- **Network** : Stable internet for model loading

## 👨‍💻 Creator Information

### **Sidoine Kolaolé YEBADOKPO**
- 📍 **Location** : Bohicon, Benin Republic
- 📞 **Phone** : +229 01 96 91 13 46
- 📧 **Email** : syebadokpo@gmail.com
- 🔗 **LinkedIn** : [linkedin.com/in/sidoineko](https://linkedin.com/in/sidoineko)
- 📁 **Portfolio** : [Hugging Face Portfolio: Sidoineko/portfolio](https://huggingface.co/Sidoineko/portfolio)

### 🏆 Competition Version
This first version of AgriLens AI was specifically developed to participate in the **Kaggle competition**. It represents our first public production and demonstrates our expertise in AI applied to agriculture.

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Hugging Face token for private models
HF_TOKEN=your_token_here

# Optional: Custom model path
MODEL_PATH=/path/to/local/model
```

### Streamlit Configuration
```toml
# .streamlit/config.toml
[server]
port = 8501
headless = true
enableCORS = true
enableXsrfProtection = true

[theme]
primaryColor = "#00FF00"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

## 📈 Roadmap

### Version 1.1 (Planned)
- [ ] Additional language support (Spanish, Portuguese)
- [ ] Offline model capability
- [ ] Batch image processing
- [ ] Export results to PDF

### Version 1.2 (Future)
- [ ] Mobile app development
- [ ] Advanced disease database
- [ ] Treatment recommendation engine
- [ ] Community features

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/your-username/AgriLensAI.git

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Submit a pull request
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**Important Warning** : The results provided by AgriLens AI are for informational purposes only. For professional diagnosis, consult a qualified expert. The application is not a substitute for professional agricultural advice.

## 📞 Support

### Getting Help
1. **Documentation** : Check the user manuals above
2. **Application** : Use the "About" tab in the app
3. **Direct Contact** : Email syebadokpo@gmail.com
4. **Community** : Join agricultural forums

### Bug Reports
Please report bugs and issues on our [GitHub Issues](https://github.com/Sidoineko/AgriLensAI/issues) page.

## 🎉 Acknowledgments

- **Google** : For the Gemma 2B model
- **Hugging Face** : For the deployment platform
- **Streamlit** : For the web framework
- **Kaggle** : For hosting the competition

---

**Made with ❤️ by Sidoine Kolaolé YEBADOKPO**

*AgriLens AI - Intelligent plant diagnosis with AI* 🌱