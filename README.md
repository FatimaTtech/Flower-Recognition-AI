# BloomAI — Lightweight Flower Classification Web Application

> **Tagline**: *Discover the flowers around you.*

BloomAI is an academic, production-ready, lightweight deep learning web application that classifies flower images into five botanical species with an **87.21% test accuracy** using a fine-tuned **ResNet-18** model.

---

## 1. Project Overview

BloomAI converts an academic machine learning study into a production-grade, decoupled web service. The project prioritizes simplicity, speed, small deployment footprint, and reproducible scientific metrics without unnecessary UI libraries or excessive dependencies.

### Supported Flower Classes
1. 🌼 **Daisy**
2. 🌾 **Dandelion**
3. 🌹 **Rose**
4. 🌻 **Sunflower**
5. 🌷 **Tulip**

---

## 2. Key Features

- **Transfer Learning Backbone**: Reconstructs the exact fine-tuned **ResNet-18** architecture trained on Colab.
- **Fast Single-Page Interface**: Clean botanical-themed React application with pastel green, cream/white backgrounds, and subtle pink accents.
- **Drag-and-Drop Classification**: Immediate image preview, client-side format & size validation (JPG/PNG/WEBP up to 10 MB).
- **Comprehensive Probability Breakdown**: Displays predicted species, top prediction confidence, and probability distributions for the top-3 candidates.
- **Decoupled Architecture**: Frontend static assets (Cloudflare Pages) communicate via REST API with the Python PyTorch backend (Hugging Face Spaces).
- **Memory-Efficient & Production-Ready**: Loads model into memory once upon startup; runs inference under `torch.inference_mode()`.

---

## 3. Technology Stack

- **Machine Learning & Backend**:
  - Python 3.11
  - PyTorch (CPU-optimized)
  - Torchvision
  - Flask & Flask-CORS
  - Gunicorn (WSGI Production Server)
  - Pillow & NumPy
- **Frontend**:
  - React 18
  - Vite
  - Plain Modular CSS (No heavy UI frameworks)

---

## 4. Model Architecture & Verified Evaluation Results

The primary model is a fine-tuned **ResNet-18** model with a customized classifier head:
- `fc.0`: `Linear(in_features=512, out_features=256)`
- `fc.1`: `ReLU(inplace=True)`
- `fc.2`: `Dropout(p=0.5)`
- `fc.3`: `Linear(in_features=256, out_features=5)`

### Preprocessing Pipeline:
- **Input Dimension**: `(3, 224, 224)`
- **Transforms**: `Resize(256)` → `CenterCrop(224)` → `ToTensor()` → `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

### Comparison Against Other Models on Test Set (735 samples):

| Model Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Checkpoint Size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MLP (Baseline)** | 48.98% | 49.25% | 49.12% | 47.90% | 24.63 MB |
| **Original CNN** | 52.65% | 52.78% | 52.55% | 51.84% | 98.37 MB |
| **Optimized CNN** | 67.21% | 67.36% | 67.04% | 66.97% | 4.63 MB |
| **ResNet-18 (Selected)** | **87.21%** | **87.25%** | **87.20%** | **87.20%** | **43.22 MB** |

---

## 5. Project Folder Structure

```text
Flower-Classification-App/
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.jsx            # Botanical single-page React UI
│   │   ├── main.jsx           # React DOM entrypoint
│   │   ├── api.js             # Configurable REST API client
│   │   └── style.css          # Lightweight custom botanical CSS
│   ├── index.html             # HTML shell with Google Fonts
│   ├── vite.config.js         # Vite bundler configuration
│   ├── package.json           # Frontend dependencies
│   └── .env.example           # Environment template (VITE_API_BASE_URL)
│
├── backend/
│   ├── app.py                 # Flask API endpoints (/api/health, /api/model_info, /api/predict)
│   ├── model.py               # ResNet-18 architecture reconstruction
│   ├── predictor.py           # Inference engine, transforms, and checkpoint discovery
│   ├── requirements.txt       # Production backend dependencies
│   └── Dockerfile             # CPU-optimized Docker container for Hugging Face Spaces
│
├── model/
│   └── model_config.json      # Verified classes, parameters, and evaluation metrics
│
├── docs/
│   └── model_performance.md   # Detailed evaluation report and architecture notes
│
├── .gitignore                 # Excludes .pth checkpoints, venv, and node_modules
└── README.md                  # Comprehensive project guide
```

---

## 6. Local Installation & Running

### Prerequisites
- Python 3.10+ (or [uv](https://docs.astral.sh/uv/))
- Node.js 18+ and npm

### 1. Run the Backend Locally

```bash
cd Flower-Classification-App/backend

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (CPU PyTorch is recommended for lightweight local serving)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Place your checkpoint or set environment variable
# If best_resnet_model.pth is in the parent backup folder, the predictor detects it automatically.
# Otherwise, set MODEL_CHECKPOINT_PATH:
# export MODEL_CHECKPOINT_PATH=/path/to/best_resnet_model.pth

python app.py
```
The backend starts at `http://localhost:7860`.

Verify:
- Health check: `curl http://localhost:7860/api/health`
- Model info: `curl http://localhost:7860/api/model_info`

### 2. Run the Frontend Locally

```bash
cd Flower-Classification-App/frontend

npm install
npm run dev
```
Open `http://localhost:3000` (or `http://localhost:5173`) in your browser.

---

## 7. Hugging Face Spaces Backend Deployment

1. **Create Space**:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces).
   - Click **Create new Space**.
   - Choose Space SDK: **Docker** (Blank template).
2. **Push Backend Files**:
   - Push `backend/Dockerfile`, `backend/app.py`, `backend/model.py`, `backend/predictor.py`, `backend/requirements.txt`, and `model/model_config.json`.
3. **Upload Checkpoint**:
   - Option A: Upload `best_resnet_model.pth` directly into the Space repo (checkpoint size is 43.2 MB, well within HF Git LFS limits).
   - Option B: Host it on Hugging Face Models repository and provide `MODEL_DOWNLOAD_URL` in Space Settings > Variables and secrets.
4. Note your public Space URL (e.g. `https://<your-username>-bloomai-backend.hf.space`).

---

## 8. Cloudflare Pages Frontend Deployment

1. **Build Verification**:
   ```bash
   cd Flower-Classification-App/frontend
   npm run build
   ```
   Output directory is `dist`.
2. **Deploy on Cloudflare Pages**:
   - Connect your GitHub repository to Cloudflare Pages.
   - Root directory: `Flower-Classification-App/frontend`
   - Build system: Version 2
   - Build command: `npm run build`
   - Build output directory: `dist`
   - Environment Variable:
     - `VITE_API_BASE_URL`: `https://<your-username>-bloomai-backend.hf.space`

---

## 9. Git Publishing Commands

```bash
# Initialize git repository inside the project directory
cd Flower-Classification-App
git init

# Stage tracked files (checkpoints, datasets, and node_modules are ignored via .gitignore)
git add .
git status

# Commit changes
git commit -m "feat: initial commit for BloomAI flower classification application"

# Add remote and push
git remote add origin https://github.com/<your-username>/BloomAI.git
git branch -M main
git push -u origin main
```

---

## 10. Known Limitations

1. **Fixed Class Domain**: The model is trained exclusively on 5 flower classes. Feeding images of animals, landscapes, or non-botanical items will still force a prediction across the 5 classes.
2. **Severe Occlusion**: Images where the flower is extremely small or heavily occluded may have reduced confidence.
3. **CPU Latency**: On free tier CPU instances (such as Hugging Face Spaces free tier), inference time is approximately 80–180ms per image.

