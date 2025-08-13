# 🔧 Guide de Dépannage CUDA - BitsAndBytes

## 🚨 Problème Identifié
Votre Tesla P100 a une capacité de calcul 6.0, ce qui cause des problèmes avec bitsandbytes qui essaie d'utiliser CUDA 12.6 alors que votre environnement a CUDA 11.8.

## ✅ Solutions par Ordre de Priorité

### Solution 1 : Quantification 4-bit (Recommandée)
Utilisez le fichier `notebook_fixed_cell_4bit.py` qui :
- Force CUDA 11.8 avec `BITSANDBYTES_CUDA_VERSION=118`
- Utilise la quantification 4-bit au lieu de 8-bit
- Configure les variables d'environnement CUDA

### Solution 2 : Script Automatique
Exécutez le script de correction CUDA :
```bash
python fix_cuda_bitsandbytes.py
```

### Solution 3 : Installation Manuelle
```bash
# 1. Désinstaller bitsandbytes
pip uninstall bitsandbytes -y

# 2. Configurer l'environnement
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64
export BITSANDBYTES_CUDA_VERSION=118

# 3. Installer version compatible
pip install bitsandbytes==0.39.2
```

## 🔍 Diagnostic

### Vérifier votre configuration :
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Capacité: {torch.cuda.get_device_capability(0)}")
```

### Diagnostic bitsandbytes :
```bash
python -m bitsandbytes
```

## 🎯 Versions Compatibles Tesla P100

| Composant | Version | Compatibilité |
|-----------|---------|---------------|
| PyTorch | 2.1.2 | ✅ Stable |
| CUDA | 11.8 | ✅ Compatible P100 |
| bitsandbytes | 0.39.2 | ✅ Compatible CUDA 11.8 |
| Quantification | 4-bit | ✅ Recommandée pour P100 |

## 🚀 Configuration Optimale pour Tesla P100

```python
# Configuration environnement
import os
os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:/usr/local/nvidia/lib64"
os.environ["BITSANDBYTES_CUDA_VERSION"] = "118"

# Configuration quantification 4-bit
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit pour P100
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
```

## 🔧 Erreurs Courantes et Solutions

### Erreur : "CUDA detection failed"
**Solution :** Forcer CUDA 11.8
```bash
export BITSANDBYTES_CUDA_VERSION=118
```

### Erreur : "Compute capability < 7.5 detected"
**Solution :** Utiliser quantification 4-bit
```python
load_in_4bit=True  # Au lieu de load_in_8bit=True
```

### Erreur : "libcudart.so not found"
**Solution :** Configurer LD_LIBRARY_PATH
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64
```

## 📋 Checklist de Résolution

- [ ] Désinstaller bitsandbytes problématique
- [ ] Configurer variables d'environnement CUDA
- [ ] Installer bitsandbytes==0.39.2
- [ ] Utiliser quantification 4-bit
- [ ] Tester l'import de bitsandbytes
- [ ] Charger le modèle avec device_map="auto"

## 🆘 Si Rien Ne Fonctionne

### Option 1 : Sans Quantification
```python
model = Gemma3nForConditionalGeneration.from_pretrained(
    GEMMA_PATH,
    local_files_only=True,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16
)
```

### Option 2 : Compilation depuis Source
```bash
pip install git+https://github.com/TimDettmers/bitsandbytes.git
```

### Option 3 : Environnement Conda
```bash
conda install -c conda-forge bitsandbytes
```

---

**Note :** Le Tesla P100 a une capacité de calcul 6.0, ce qui limite les options de quantification. La quantification 4-bit est la meilleure option pour ce GPU. 