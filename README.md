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
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](https://huggingface.co/spaces/sido1991/Agrilens_IAv1)

## 🎯 Overview

AgriLens AI is an innovative **plant disease diagnosis application** using artificial intelligence to help farmers identify and treat problems in their crops. This application leverages **Google's Gemma 3n multimodal model** for initial visual analysis, enhanced by **Google's Gemini AI** for precise diagnosis interpretation.

### 🌟 Key Features

- **📸 Image Analysis** : Visual disease diagnosis using Gemma 3n multimodal AI
- **💬 Text Analysis** : Advice based on symptom descriptions
- **🌐 Multilingual Support** : French and English interfaces
- **📱 Mobile Responsive** : Optimized for smartphones and tablets
- **🔧 Practical Recommendations** : Concrete actions to take
- **⚡ Real-time Processing** : Fast AI-powered analysis
- **🧠 Advanced AI** : Gemini API integration for enhanced diagnosis interpretation
- **🔄 Adaptive Model Loading** : Smart model selection based on available resources
- **💾 Memory Optimization** : Efficient resource management for deployment
- **📄 Export Functionality** : Export diagnostics in HTML or text format

## 🚀 Live Demo

### 🌐 Online Version
- **Hugging Face Spaces** : [AgriLens AI Demo](https://huggingface.co/spaces/sido1991/Agrilens_IAv1)
- **Status** : ✅ Fully functional with multilingual support and Gemini AI
- **Performance** : Optimized for 16GB RAM environments

### 📱 Mobile Access
- Open the demo link on your smartphone
- Interface automatically adapts to mobile screens
- Touch-friendly controls and navigation

## 🏗️ Project Architecture

![AgriLens AI Architecture](https://github.com/Sidoine1991/Agrilens-AI/blob/main/appdesign.png?raw=true)

### System Design
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend        │    │   AI Models     │
│   (Streamlit)   │◄──►│   (Python)       │◄──►│   (Gemma 3n)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │    │   Model Manager  │    │   Gemini API    │
│   (Image/Text)  │    │   (Adaptive)     │    │   (Enhancement) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### AI Architecture
- **Primary Model** : **Gemma 3n E4B IT** (Google) - Multimodal model for visual analysis
- **Enhancement Model** : **Gemini 1.5 Flash API** (Google) - For improved diagnosis interpretation
- **Processing Pipeline** : 
  1. Gemma 3n analyzes the image and provides initial diagnosis
  2. Gemini API enhances the interpretation with detailed recommendations
  3. Results are presented in a user-friendly format

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- Modern web browser
- Internet connection (for model loading)
- Google API Key (for Gemini enhancement)
- Minimum 8GB RAM (16GB+ recommended for full Gemma 3n)

### Local Installation

```bash
# Clone the repository
git clone https://github.com/Sidoine1991/Agrilens-AI.git
cd AgriLens-AI

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

## 📖 User Guide

### 🎯 Quick Start Guide

1. **Load the AI Model** : Click "Load AI Model" in settings
2. **Choose Language** : Select French or English
3. **Upload Image** : Take a photo of the diseased plant
4. **Specify Crop Type** : Enter the crop type for better accuracy
5. **Get Diagnosis** : Receive AI-powered analysis and recommendations
6. **Export Results** : Download diagnosis in HTML or text format

### 📸 Image Analysis Process
1. **Image Upload** : Upload or capture plant image
2. **Crop Specification** : Specify the crop type (optional but recommended)
3. **Gemma 3n Analysis** : Initial visual diagnosis by Gemma 3n
4. **Gemini Enhancement** : Detailed interpretation and recommendations
5. **Results Display** : Comprehensive diagnosis with treatment options

## 🔬 Technology Stack

### Core Technologies
- **Framework** : [Streamlit](https://streamlit.io/) - Web application framework
- **Primary AI Model** : [Gemma 3n E4B IT](https://huggingface.co/google/gemma-3n-E4B-it) - Google's multimodal model
- **Enhancement AI** : [Gemini 1.5 Flash API](https://ai.google.dev/) - Google's advanced AI for diagnosis interpretation
- **Deployment** : [Hugging Face Spaces](https://huggingface.co/spaces) - Cloud hosting platform
- **Languages** : Python, HTML, CSS, JavaScript

### Key Libraries
- **Transformers** : Hugging Face's AI model library
- **Google Generative AI** : Gemini API integration
- **PyTorch** : Deep learning framework
- **Pillow** : Image processing
- **Streamlit** : Web interface
- **Torch** : Tensor operations and model inference

## ⚙️ Model Configuration

### AI Model Architecture
The application uses a two-stage AI approach:

```python
# Stage 1: Gemma 3n for initial visual analysis
gemma_diagnosis = gemma3n_analyze_image(image, crop_type)

# Stage 2: Gemini for enhanced interpretation
enhanced_diagnosis = gemini_enhance_diagnosis(gemma_diagnosis)
```

### Model Specifications
| Model | Size | Memory Usage | Purpose |
|-------|------|--------------|---------|
| Gemma 3n E4B IT | ~8GB | 16GB+ | Primary visual analysis |
| Gemini 1.5 Flash | API | Minimal | Diagnosis enhancement |

### Memory Management
- **Dynamic Model Loading** : Loads Gemma 3n based on available RAM
- **Cache Management** : Efficient model persistence and restoration
- **Resource Monitoring** : Real-time memory usage tracking

## 🎯 Performance Metrics

### Accuracy & Speed
- **Image Recognition** : High accuracy in disease identification using Gemma 3n
- **Diagnostic Precision** : Enhanced with Gemini AI interpretation
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
- **Dynamic Model Loading** : Loads Gemma 3n based on available RAM
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

### Export Functionality
- **HTML Export** : Styled diagnostic reports
- **Text Export** : Plain text diagnostic summaries
- **Timestamped Files** : Organized export with timestamps

## 👨‍💻 Creator Information

### **Sidoine Kolaolé YEBADOKPO**
- **Location** : Bohicon, Republic of Benin
- **Phone** : +229 01 96 91 13 46
- **Email** : syebadokpo@gmail.com
- **LinkedIn** : [linkedin.com/in/sidoineko](https://linkedin.com/in/sidoineko)
- **Hugging Face Portfolio** : [Sidoineko/portfolio](https://huggingface.co/spaces/Sidoineko/portfolio)

### 🏆 Competition Version
This version of AgriLens AI was specifically developed for the **Google - Gemma 3n Hackathon**. It demonstrates advanced AI integration using Google's latest multimodal models for agricultural applications.

## ⚙️ Configuration

### Environment Variables
```bash
# Required for enhanced diagnosis
GOOGLE_API_KEY=your_google_api_key_here

# Optional Hugging Face token
HF_TOKEN=your_huggingface_token_here

# Optional: Force specific model loading strategy
FORCE_MODEL_STRATEGY=conservative
```

### Model Configuration
- **Primary Model** : Gemma 3n E4B IT (Google)
- **Enhancement AI** : Gemini 1.5 Flash (Google)
- **Processing** : CPU optimized for deployment
- **Memory Management** : Adaptive based on environment

## 🗺️ Roadmap

### Version 1.0 (Current) ✅
- ✅ Gemma 3n image analysis
- ✅ Gemini AI enhancement
- ✅ Text-based diagnosis
- ✅ Multilingual support
- ✅ Mobile responsiveness
- ✅ Export functionality
- ✅ Adaptive model loading
- ✅ Memory optimization
- ✅ Error handling

### Version 1.1 (Planned) 🔄
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

## 📄 License

This project is developed for the **Google - Gemma 3n Hackathon** and is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

**CC BY 4.0 License** : You are free to share and adapt this material for any purpose, even commercially, provided you give appropriate credit.

- **Attribution** : Sidoine Kolaolé YEBADOKPO
- **License Link** : https://creativecommons.org/licenses/by/4.0/
- **Competition Compliance** : This license is required for participation in the Google - Gemma 3n Hackathon

See the [LICENSE](LICENSE) file for full details.

## ⚠️ Disclaimer

This application is designed to assist farmers and gardeners in identifying plant diseases. While we strive for accuracy using advanced AI models (Gemma 3n + Gemini), the diagnosis should not replace professional agricultural advice. Always consult with local agricultural experts for critical decisions.

## 🐛 Known Issues & Solutions

### Memory Issues on HF Spaces
- **Problem** : Gemma 3n loading fails due to 16GB RAM limit
- **Solution** : Application automatically uses conservative loading strategy
- **Workaround** : Manual model selection in settings

### Model Loading Timeouts
- **Problem** : Long loading times on slow connections
- **Solution** : Implemented timeout protection with fallback strategies
- **Workaround** : Use local deployment for faster loading

## 🙏 Acknowledgments

- **Google** : For providing Gemma 3n and Gemini AI models
- **Hugging Face** : For the deployment platform and transformers library
- **Streamlit** : For the web framework
- **Google - Gemma 3n Hackathon** : For inspiring this project
- **Open Source Community** : For the amazing tools and libraries

## 📊 Project Statistics

- **Lines of Code** : ~2,000+
- **AI Models** : 2 (Gemma 3n + Gemini)
- **Languages** : 2 (French, English)
- **Deployment Platforms** : 3 (Local, Docker, HF Spaces)
- **Test Coverage** : Comprehensive error handling

## 🔗 Important Links

- **Live Demo** : [Hugging Face Spaces](https://huggingface.co/spaces/sido1991/Agrilens_IAv1)
- **Kaggle Notebook** : [AgriLens AI Notebook](https://www.kaggle.com/code/sidoineyebadokpo/agrilens-ai?scriptVersionId=253640926)
- **Portfolio** : [Hugging Face Portfolio](https://huggingface.co/spaces/Sidoineko/portfolio)
- **Repository** : [GitHub](https://github.com/Sidoine1991/Agrilens-AI)

---

*AgriLens AI - Intelligent plant disease diagnosis with Google's Gemma 3n and Gemini AI* 🌱

**Last Updated** : July 2025  
**Version** : 1.0.0  
**Status** : Production Ready 