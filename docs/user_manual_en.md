# 🌱 AgriLens AI - User Manual

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [Installation and Configuration](#installation-and-configuration)
3. [User Interface](#user-interface)
4. [Image Analysis](#image-analysis)
5. [Text Analysis](#text-analysis)
6. [Result Interpretation](#result-interpretation)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [Support and Contact](#support-and-contact)
10. [Technical Information](#technical-information)

---

## 🎯 Introduction

### What is AgriLens AI?

AgriLens AI is an innovative plant disease diagnosis application using artificial intelligence. Developed specifically to participate in the Kaggle competition, this first version represents our expertise in AI applied to agriculture.

### Objectives

- **Quick diagnosis** : Identify plant diseases in seconds
- **Practical advice** : Provide concrete action recommendations
- **Accessibility** : Simple and intuitive interface for all farmers
- **Multilingual support** : Available in French and English

### Target Audience

- Professional and amateur farmers
- Gardeners and horticulturists
- Agronomy students
- Agricultural consultants
- Anyone interested in plant health

---

## ⚙️ Installation and Configuration

### Prerequisites

- Modern web browser (Chrome, Firefox, Safari, Edge)
- Stable internet connection
- Hugging Face account (for AI model access)

### Application Access

1. **Online version** : Access the application via Hugging Face Spaces
2. **Local version** : Clone the repository and run locally

### Initial Configuration

1. **Language selection** : Choose between French and English
2. **Model loading** : Click "Load Gemma 2B Model"
3. **Wait for loading** : The model downloads automatically

---

## 🖥️ User Interface

### General Structure

The application is organized into 4 main tabs:

1. **📸 Image Analysis** : Diagnosis by photography
2. **💬 Text Analysis** : Diagnosis by description
3. **📖 User Manual** : Usage guide
4. **ℹ️ About** : Application information

### Sidebar (Configuration)

- **Language selector** : French/English
- **Model loading** : Button to initialize AI
- **Model status** : System status indicator

---

## 📸 Image Analysis

### Analysis Process

1. **Image upload** : Drag and drop or select an image
2. **Verification** : Application displays image information
3. **Optional question** : Specify your concern
4. **AI analysis** : Model generates diagnosis
5. **Results** : Display of diagnosis and recommendations

### Accepted Formats

- **PNG** : Recommended format for quality
- **JPG/JPEG** : Common accepted formats
- **Minimum size** : 500x500 pixels recommended

### Tips for Better Results

- **Lighting** : Use natural and uniform lighting
- **Focus** : Center the image on the diseased area
- **Resolution** : Use good quality images
- **Multiple angles** : Take several photos if necessary

### Usage Example

```
1. Photograph a tomato leaf with brown spots
2. Upload the image to the application
3. Add the question: "What is this disease?"
4. Get a detailed diagnosis with recommendations
```

---

## 💬 Text Analysis

### When to Use Text Analysis

- No image available
- Detailed symptom description
- General questions about plant care
- Preventive advice

### Recommended Description Structure

```
1. Plant type: Tomato, Lettuce, etc.
2. Observed symptoms: Spots, discoloration, etc.
3. Location: Leaves, fruits, stems, etc.
4. Evolution: Since when, progression
5. Conditions: Watering, exposure, temperature
6. Actions already tried: Applied treatments
```

### Description Example

```
"My tomato plants have circular brown spots on the leaves 
for a week. The spots are getting bigger and some leaves 
are turning yellow. I reduced watering but it's getting worse. 
The plants are in full sun and I water in the morning."
```

---

## 🔍 Result Interpretation

### Result Structure

Each analysis produces:

1. **Diagnosis** : Probable disease identification
2. **Causes** : Factors that may have triggered the problem
3. **Symptoms** : Detailed description of signs
4. **Recommendations** : Concrete actions to take
5. **Prevention** : Measures to avoid recurrence

### Result Example

```
**Diagnosis:** Tomato blight (Phytophthora infestans)

**Possible causes:**
• Excessive humidity
• Watering on leaves
• Lack of air circulation

**Urgent recommendations:**
• Isolate diseased plants
• Remove affected leaves
• Apply appropriate fungicide
• Improve ventilation

**Prevention:**
• Water at the base of plants
• Space plants sufficiently
• Monitor humidity
```

---

## 💡 Best Practices

### For Image Analysis

- **Quality** : Use sharp and well-lit images
- **Framing** : Include the diseased area and some context
- **Scale** : Take photos at different distances
- **Series** : Photograph evolution over several days

### For Text Analysis

- **Precision** : Describe symptoms accurately
- **Context** : Mention growing conditions
- **History** : Indicate problem evolution
- **Actions** : List treatments already tried

### General

- **Regularity** : Monitor your plants regularly
- **Documentation** : Keep track of diagnoses
- **Consultation** : Consult an expert for complex cases
- **Prevention** : Apply preventive measures

---

## 🔧 Troubleshooting

### Common Problems

#### Model Loading Error
```
Symptom: "Model not loaded"
Solution: 
1. Check your internet connection
2. Reload the page
3. Click "Load Model" again
```

#### Image Upload Error
```
Symptom: "Upload error"
Solution:
1. Check format (PNG, JPG, JPEG)
2. Reduce image size
3. Try another browser
```

#### Imprecise Results
```
Symptom: Unreliable diagnosis
Solution:
1. Improve image quality
2. Add detailed description
3. Take several photos
4. Consult expert for confirmation
```

### Error Messages

- **"Model not loaded"** : Reload the model
- **"Analysis error"** : Check your input data
- **"Timeout"** : Wait and try again
- **"Unsupported format"** : Use PNG, JPG or JPEG

---

## 📞 Support and Contact

### Application Creator

**Sidoine Kolaolé YEBADOKPO**
- 📍 **Location** : Bohicon, Benin Republic
- 📞 **Phone** : +229 01 96 91 13 46
- 📧 **Email** : syebadokpo@gmail.com
- 🔗 **LinkedIn** : linkedin.com/in/sidoineko
- 📁 **Portfolio** : Hugging Face Portfolio: Sidoineko/portfolio

### Competition Version

This first version of AgriLens AI was specifically developed to participate in the Kaggle competition. It represents our first public production and demonstrates our expertise in AI applied to agriculture.

### Important Warning

⚠️ **The results provided are for informational purposes only. For professional diagnosis, consult a qualified expert.**

### How to Get Help

1. **Documentation** : Consult this user manual
2. **Interface** : Use the "About" tab in the application
3. **Direct contact** : Use the contact information above
4. **Community** : Join agricultural forums

---

## 🔬 Technical Information

### Architecture

- **Framework** : Streamlit
- **AI Model** : Gemma 2B (Google)
- **Deployment** : Hugging Face Spaces
- **Languages** : Python, HTML, CSS

### Technical Features

- **Image analysis** : Multimodal AI processing
- **Text analysis** : Contextual response generation
- **Responsive interface** : Mobile and desktop adapted
- **Multilingual support** : French and English
- **Smart caching** : Performance optimization

### Security and Privacy

- **Data** : No personal data collected
- **Images** : Processed locally, not stored
- **Model** : Executed on secure server
- **Connection** : HTTPS mandatory

---

## 📚 Additional Resources

### Technical Documentation

- **GitHub Repository** : Complete source code
- **API Documentation** : Technical specifications
- **Deployment Guide** : Installation instructions

### Agricultural Resources

- **Databases** : Disease repositories
- **Practical guides** : Treatment methods
- **Communities** : Farmer forums

### Training and Support

- **Video tutorials** : Practical demonstrations
- **Webinars** : Training sessions
- **Technical support** : Personalized assistance

---

## 🎉 Conclusion

AgriLens AI represents a significant advancement in the application of artificial intelligence to agriculture. This first version, developed for the Kaggle competition, demonstrates the potential of AI to help farmers in their daily work.

We hope this application will be useful to you and thank you for your trust. Don't hesitate to share your feedback and suggestions to improve future versions.

**Happy using AgriLens AI! 🌱**

---

*Document generated on: [Date]*
*Version: 1.0*
*Creator: Sidoine Kolaolé YEBADOKPO* 