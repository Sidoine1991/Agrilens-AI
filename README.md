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

AgriLens AI is an innovative **plant disease diagnosis application** using artificial intelligence to help farmers identify and treat problems in their crops. This application leverages **Google's Gemma 3n multimodal model** for visual analysis and precise diagnosis interpretation.

### 🌟 Key Features

- **📸 Image Analysis** : Visual disease diagnosis using Gemma 3n multimodal AI
- **💬 Text Analysis** : Advice based on symptom descriptions
- **🌐 Multilingual Support** : French and English interfaces
- **📱 Mobile Responsive** : Optimized for smartphones and tablets
- **🔧 Practical Recommendations** : Concrete actions to take
- **⚡ Real-time Processing** : Fast AI-powered analysis
- **🧠 Advanced AI** : Gemma 3n for precise diagnosis interpretation
- **🔄 Adaptive Model Loading** : Smart model selection based on available resources
- **💾 Memory Optimization** : Efficient resource management for deployment
- **📄 Export Functionality** : Export diagnostics in HTML or text format

## 🚀 Live Demo

### 🌐 Online Version
- **Hugging Face Spaces** : [AgriLens AI Demo](https://huggingface.co/spaces/sido1991/Agrilens_IAv1)
- **Status** : ✅ Fully functional with multilingual support
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
│   User Input    │    │   Model Manager  │    │   AI Analysis   │
│   (Image/Text)  │    │   (Adaptive)     │    │   (Gemma 3n)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### AI Architecture
- **Primary Model** : **Gemma 3n E4B IT** (Google) - Multimodal model for visual analysis and diagnosis
- **Processing Pipeline** : 
  1. Gemma 3n analyzes the image and provides comprehensive diagnosis
  2. Results are presented in a user-friendly format with detailed recommendations

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- Modern web browser
- Internet connection (for model loading)
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
# No additional API keys required

# Run the application
streamlit run src/streamlit_app_multilingual.py
```

## 🌾 Offline-First Architecture & Real-World Deployment

### **🏗️ Point 1: "Offline-First" Design Philosophy**

AgriLens AI was conceived from day one with a **"Offline-First" philosophy**. Our architecture is not that of a traditional web application dependent on remote APIs. Instead, the AI model and all processing logic are entirely contained within the application itself.

**Technical Proof in Our Code:**
- **Embedded Model**: Our script loads the Gemma 3n model directly into memory. There are no external API calls for diagnosis. Once loaded, cutting the internet connection has absolutely no impact on functionality.
- **Local Processing**: All operations – from image resizing to model inference – are performed locally by the CPU/GPU of the machine running the application.
- **Complete Independence**: The application only needs internet connection once during initial installation to download the model and libraries. After this step, it operates 100% autonomously.

### **💻 Point 2: The "Farm Laptop" Deployment Vision**

The question isn't asking farmers to use Hugging Face. The online version is simply a demonstration for the competition. Real deployment in the field is designed to be simple and robust.

**The Real-World Scenario:**
1. **One-Time Installation**: An agricultural technician, extension agent, or even a family member visits an area with internet connection (town, cybercafe...)
2. **The "AgriLens AI Kit"**: On a standard laptop (even a used model with 8GB RAM), they follow a simple installation procedure to download our application and the Gemma 3n model
3. **The "Farm Laptop"**: This computer is then brought to the farm or cooperative headquarters. It never needs internet connection again. It becomes the "Farm Laptop" – the community's diagnostic tool
4. **Daily Usage**: When a farmer has concerns about a plant, they don't connect to a website. They take a photo with their phone, transfer it to the "Farm Laptop" (via USB cable or SD card), and launch the locally installed AgriLens AI application. They get instant diagnosis, right in the field, kilometers from the nearest network tower.

### **📱 Point 3: Proof by Demonstration - Simulated "Offline" Mode**

Our Streamlit application is our strongest argument. We've integrated a "Mobile Offline Mode" directly into our demonstration.

**What You See in the Demo:**
- The mobile interface simulates exactly what the application would look like on a small device
- The "Mode: OFFLINE" badge isn't just decorative – it represents our commitment to building a solution that truly works without internet
- The code running on Hugging Face servers today is exactly the same code that would run on a simple laptop in Bohicon, Benin, without any connection

### **🌍 Real-World Deployment Strategies**

#### **Strategy 1: The "Farm Laptop" Model (Recommended)**
```bash
# One-time setup in town with internet
git clone https://github.com/Sidoine1991/Agrilens-AI.git
cd AgriLens-AI
pip install -r requirements.txt
streamlit run src/streamlit_app_multilingual.py
# Model downloads (~8GB) - then works offline forever

# Daily use on farm (no internet needed)
streamlit run src/streamlit_app_multilingual.py
# Instant diagnosis, completely offline
```

**Benefits:**
- ✅ **One-time setup** in town with internet
- ✅ **Lifetime offline use** on the farm
- ✅ **Shared resource** for entire community
- ✅ **No ongoing costs** or internet dependency
- ✅ **Works on any laptop** with 8GB+ RAM

#### **Strategy 2: Agricultural Extension Service**
- **Setup**: Extension workers carry pre-loaded laptops
- **Field Work**: Visit farms, analyze plants on-site without internet
- **Benefits**: Professional diagnosis anywhere, immediate recommendations

#### **Strategy 3: Cooperative Hub**
- **Setup**: One computer per agricultural cooperative
- **Shared Use**: Multiple farmers bring plant samples for analysis
- **Benefits**: Cost-effective, centralized expertise, community resource

### **📊 Offline vs Traditional Methods Comparison**

| Feature | AgriLens AI (Offline) | Traditional Methods | Online Solutions |
|---------|----------------------|-------------------|------------------|
| **Internet Required** | ❌ Only for initial setup | ❌ Requires expert visit | ✅ Always needed |
| **Cost per Analysis** | ✅ Free unlimited use | ❌ Expensive consultation | ❌ Potential API costs |
| **Data Privacy** | ✅ Data stays local | ✅ Expert confidentiality | ❌ Data sent to cloud |
| **Speed** | ✅ Instant local processing | ❌ Days/weeks wait time | ⚠️ Depends on connection |
| **Reliability** | ✅ Always available | ❌ Limited expert availability | ⚠️ Depends on internet |
| **Setup Complexity** | ⚠️ One-time installation | ✅ No setup needed | ✅ Just visit URL |

### **🎯 Why This Approach is Revolutionary for Farmers**

**The Traditional Problem:**
- Farmers wait weeks for agricultural experts
- Pay expensive consultation fees
- Limited expert availability in remote areas
- Language barriers with foreign experts
- No immediate treatment decisions

**The AgriLens AI Solution:**
- **Instant diagnosis** on the farm
- **Completely free** (open source)
- **Available 24/7** without scheduling
- **French and English** support
- **Immediate treatment decisions**

**Real Impact:**
- **Faster treatment** = reduced crop losses
- **Better yields** = increased income
- **Empowered farmers** = sustainable agriculture
- **Knowledge sharing** = community development

### **💡 Practical Implementation Guide**

#### **For Agricultural Technicians:**
1. **Download in Town**: Visit town with internet, download AgriLens AI
2. **Install on Laptop**: Simple installation on any laptop with 8GB+ RAM
3. **Deploy to Farm**: Bring laptop to farm - no internet needed
4. **Train Farmers**: Show farmers how to use the application
5. **Community Resource**: Laptop becomes shared diagnostic tool

#### **For Extension Services:**
1. **Pre-load Laptops**: Install AgriLens AI on extension worker laptops
2. **Field Deployment**: Workers visit farms with diagnostic capability
3. **On-site Analysis**: Immediate diagnosis and recommendations
4. **Knowledge Transfer**: Train farmers during visits

#### **For Cooperatives:**
1. **Central Setup**: Install on cooperative office computer
2. **Shared Access**: Multiple farmers can use the same system
3. **Record Keeping**: Save diagnoses for future reference
4. **Community Learning**: Share knowledge across members

### Quick Start (After Installation)

1. **Load the AI Model** : Click "Load AI Model" in the sidebar settings
2. **Choose Language** : Select French or English from the language selector
3. **Upload Image** : Take a photo or upload an image of the diseased plant
4. **Specify Crop Type** : Enter the crop type for better accuracy (optional but recommended)
5. **Get Diagnosis** : Receive AI-powered analysis and treatment recommendations
6. **Export Results** : Download diagnosis in HTML or text format if needed

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
3. **Gemma 3n Analysis** : Complete visual diagnosis by Gemma 3n
4. **Results Display** : Comprehensive diagnosis with treatment options

## 🔬 Technology Stack

### Core Technologies
- **Framework** : [Streamlit](https://streamlit.io/) - Web application framework
- **AI Model** : [Gemma 3n E4B IT](https://huggingface.co/google/gemma-3n-E4B-it) - Google's multimodal model for diagnosis
- **Deployment** : [Hugging Face Spaces](https://huggingface.co/spaces) - Cloud hosting platform
- **Languages** : Python, HTML, CSS, JavaScript

### Key Libraries
- **Transformers** : Hugging Face's AI model library
- **PyTorch** : Deep learning framework
- **Pillow** : Image processing
- **Streamlit** : Web interface

## ⚙️ Model Configuration

### AI Model Architecture
The application uses Gemma 3n for comprehensive visual analysis and diagnosis:

```python
# Gemma 3n for complete visual analysis and diagnosis
diagnosis = gemma3n_analyze_image(image, crop_type)
```

### Model Specifications
| Model | Size | Memory Usage | Purpose |
|-------|------|--------------|---------|
| Gemma 3n E4B IT | ~8GB | 16GB+ | Complete visual analysis and diagnosis |

### Memory Management
- **Dynamic Model Loading** : Loads Gemma 3n based on available RAM
- **Cache Management** : Efficient model persistence and restoration
- **Resource Monitoring** : Real-time memory usage tracking

## 🎯 Performance Metrics

### Accuracy & Speed
- **Image Recognition** : High accuracy in disease identification using Gemma 3n
- **Diagnostic Precision** : High accuracy with Gemma 3n analysis
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
# Optional Hugging Face token
HF_TOKEN=your_huggingface_token_here

# Optional: Force specific model loading strategy
FORCE_MODEL_STRATEGY=conservative
```

### Model Configuration
- **Primary Model** : Gemma 3n E4B IT (Google)
- **Processing** : CPU optimized for deployment
- **Memory Management** : Adaptive based on environment

## 🗺️ Roadmap

### Version 1.0 (Current) ✅
- ✅ Gemma 3n image analysis
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

This application is designed to assist farmers and gardeners in identifying plant diseases. While we strive for accuracy using advanced AI models (Gemma 3n), the diagnosis should not replace professional agricultural advice. Always consult with local agricultural experts for critical decisions.

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

- **Google** : For providing Gemma 3n model
- **Hugging Face** : For the deployment platform and transformers library
- **Streamlit** : For the web framework
- **Google - Gemma 3n Hackathon** : For inspiring this project
- **Open Source Community** : For the amazing tools and libraries

## 📊 Project Statistics

- **Lines of Code** : ~2,000+
- **AI Models** : 1 (Gemma 3n)
- **Languages** : 2 (French, English)
- **Deployment Platforms** : 3 (Local, Docker, HF Spaces)
- **Test Coverage** : Comprehensive error handling

## 🔗 Important Links

- **Live Demo** : [Hugging Face Spaces](https://huggingface.co/spaces/sido1991/Agrilens_IAv1)
- **Kaggle Notebook** : [AgriLens AI Notebook](https://www.kaggle.com/code/sidoineyebadokpo/agrilens-ai?scriptVersionId=253640926)
- **Portfolio** : [Hugging Face Portfolio](https://huggingface.co/spaces/Sidoineko/portfolio)
- **Repository** : [GitHub](https://github.com/Sidoine1991/Agrilens-AI)

---

*AgriLens AI - Intelligent plant disease diagnosis with Google's Gemma 3n* 🌱

**Last Updated** : July 2025  
**Version** : 1.0.0  
**Status** : Production Ready 