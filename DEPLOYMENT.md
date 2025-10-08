# Deployment Guide - Plant Disease Detection App

This guide shows you how to make your app accessible via a URL.

---

## Option 1: Gradio Share Link (Quickest - Already Configured!)

**Your app is already set up for this!**

### How it works:
When you run your app with `share=True`, Gradio creates a temporary public URL.

```bash
cd src
python app.py
```

**You'll see output like:**
```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://1234abcd5678.gradio.live
```

**Features**:
- ✅ **Free** and instant
- ✅ **Public URL** - share with anyone
- ✅ **No configuration** needed
- ❌ **Temporary** - expires after 72 hours
- ❌ **New URL** each time you restart

**Best for**: Quick demos, testing, sharing with friends

---

## Option 2: Hugging Face Spaces (Recommended for Free Hosting)

**Free, permanent URL with custom subdomain**

### Steps:

1. **Create Hugging Face account**: https://huggingface.co/join

2. **Create a new Space**:
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Name: `plant-disease-detection`
   - SDK: **Gradio**
   - Visibility: Public

3. **Prepare files**:
   ```bash
   # Create requirements.txt if not exists
   pip freeze > requirements.txt
   ```

4. **Create app.py in root** (Hugging Face looks for it there):
   ```python
   # Copy your src/app.py content but adjust paths
   import gradio as gr
   from src.predict import DiseasePredictor
   import os

   def create_interface():
       predictor = DiseasePredictor('best_model.pth', 'classes.json')
       # ... rest of your code

   if __name__ == "__main__":
       interface = create_interface()
       interface.launch()  # Remove share=True, server_port
   ```

5. **Upload to Hugging Face**:
   ```bash
   # Install Git LFS (for large files)
   git lfs install

   # Clone your space
   git clone https://huggingface.co/spaces/YOUR_USERNAME/plant-disease-detection
   cd plant-disease-detection

   # Copy files
   cp -r /path/to/project/src .
   cp /path/to/project/best_model.pth .
   cp /path/to/project/classes.json .
   cp /path/to/project/requirements.txt .

   # Track model file with Git LFS
   git lfs track "*.pth"

   # Commit and push
   git add .
   git commit -m "Initial deployment"
   git push
   ```

6. **Access your app**:
   - URL: `https://huggingface.co/spaces/YOUR_USERNAME/plant-disease-detection`
   - Or custom: `https://YOUR_USERNAME-plant-disease-detection.hf.space`

**Features**:
- ✅ **Free** forever
- ✅ **Permanent URL**
- ✅ **Custom subdomain**
- ✅ **Auto-rebuilds** on git push
- ⚠️ **CPU only** on free tier (slower inference)

---

## Option 3: Google Colab (Free, No Setup)

**Run in Google Colab with public URL**

### Steps:

1. **Create a Colab notebook**: https://colab.research.google.com

2. **Add setup cells**:
   ```python
   # Cell 1: Clone your repo
   !git clone https://github.com/YOUR_USERNAME/plant-disease-detection.git
   %cd plant-disease-detection

   # Cell 2: Install dependencies
   !pip install -q gradio torch torchvision pillow

   # Cell 3: Download model (if not in repo)
   # Upload best_model.pth and classes.json manually or from Drive

   # Cell 4: Run app
   %cd src
   !python app.py
   ```

3. **The Gradio share link will appear in output**

**Features**:
- ✅ **Free GPU** access
- ✅ **No installation** needed
- ✅ **Public URL** via Gradio
- ❌ **Session expires** after inactivity
- ❌ **Must keep browser open**

---

## Option 4: Render (Free Web Service)

**Free deployment with persistent URL**

### Steps:

1. **Create account**: https://render.com

2. **Create `render.yaml`** in project root:
   ```yaml
   services:
     - type: web
       name: plant-disease-detection
       env: python
       buildCommand: pip install -r requirements.txt
       startCommand: python src/app.py
   ```

3. **Update app.py** for Render:
   ```python
   # Change the launch line to:
   interface.launch(server_name="0.0.0.0", server_port=7860)
   ```

4. **Push to GitHub** (if not already)

5. **Deploy on Render**:
   - New Web Service → Connect GitHub repo
   - Select your repository
   - Render auto-detects settings
   - Click "Create Web Service"

6. **Access**: `https://plant-disease-detection.onrender.com`

**Features**:
- ✅ **Free tier** available
- ✅ **Persistent URL**
- ✅ **Auto-deploys** on git push
- ⚠️ **Sleeps after 15 min** inactivity (free tier)
- ⚠️ **Slow cold starts**

---

## Option 5: Railway (Easy Deployment)

**Simple deployment with custom domain**

### Steps:

1. **Create account**: https://railway.app

2. **Create `Procfile`**:
   ```
   web: python src/app.py
   ```

3. **Update app.py**:
   ```python
   import os
   port = int(os.environ.get("PORT", 7860))
   interface.launch(server_name="0.0.0.0", server_port=port)
   ```

4. **Deploy**:
   - New Project → Deploy from GitHub
   - Select repository
   - Railway auto-deploys

**Features**:
- ✅ **$5 free credit** monthly
- ✅ **Custom domains**
- ✅ **Fast deployment**
- ⚠️ **Paid after free credit**

---

## Option 6: Ngrok (Local with Public URL)

**Run locally but accessible via URL**

### Steps:

1. **Install ngrok**: https://ngrok.com/download
   ```bash
   brew install ngrok  # macOS
   # or download from website
   ```

2. **Sign up** at ngrok.com for auth token

3. **Configure**:
   ```bash
   ngrok config add-authtoken YOUR_TOKEN
   ```

4. **Run your app locally**:
   ```bash
   cd src
   python app.py  # Runs on port 7860
   ```

5. **In another terminal, create tunnel**:
   ```bash
   ngrok http 7860
   ```

6. **Access via ngrok URL**: `https://abc123.ngrok.io`

**Features**:
- ✅ **Runs on your machine**
- ✅ **Public URL**
- ✅ **Good for development**
- ❌ **Computer must stay on**
- ❌ **New URL** each time

---

## Comparison Table

| Option | Cost | Speed | Permanence | Setup Difficulty | Best For |
|--------|------|-------|------------|------------------|----------|
| **Gradio Share** | Free | Fast | Temporary (72h) | ⭐ Easiest | Quick demos |
| **Hugging Face** | Free | Medium (CPU) | Permanent | ⭐⭐ Easy | Public projects |
| **Google Colab** | Free | Fast (GPU) | Temporary | ⭐ Easiest | GPU testing |
| **Render** | Free tier | Slow (cold start) | Permanent | ⭐⭐⭐ Medium | Production |
| **Railway** | $5/month | Fast | Permanent | ⭐⭐ Easy | Production |
| **Ngrok** | Free tier | Fast | Temporary | ⭐⭐ Easy | Development |

---

## Recommended Setup for You

### For Quick Demo (Right Now):
```bash
# Already configured! Just run:
cd /Users/akankshagarg/Desktop/plant_disease_detection/src
python app.py

# Share the gradio.live URL that appears
```

### For Permanent Deployment (Best Option):
**Use Hugging Face Spaces** - Free, permanent, easy to maintain

1. Create Space on Hugging Face
2. Push your code
3. Share permanent URL: `https://huggingface.co/spaces/YOUR_USERNAME/plant-disease-detection`

---

## Important Notes

### Model File Size
Your `best_model.pth` might be large (100MB+). For deployment:

**Hugging Face/GitHub**: Use Git LFS
```bash
git lfs install
git lfs track "*.pth"
```

**Alternative**: Upload to Google Drive, download in startup script:
```python
# In app.py
import gdown
if not os.path.exists('best_model.pth'):
    gdown.download('GOOGLE_DRIVE_FILE_ID', 'best_model.pth', quiet=False)
```

### Environment Variables
For sensitive configs, use environment variables:
```python
import os
MODEL_PATH = os.getenv('MODEL_PATH', 'best_model.pth')
```

---

## Troubleshooting

### "Address already in use"
```bash
# Kill process on port 7860
lsof -ti:7860 | xargs kill -9
```

### "Model file not found" (in deployment)
- Ensure model is committed to git (with LFS)
- Or use download script on startup

### Slow inference (deployed)
- Free tiers use CPU
- Consider: Hugging Face paid tier, or AWS/GCP with GPU

---

## Next Steps

1. **Test locally** with Gradio share (already working)
2. **Choose deployment platform** based on needs
3. **Push to GitHub** (already done ✓)
4. **Deploy** using chosen platform
5. **Share URL** with users

---

**Quick Start Command**:
```bash
cd /Users/akankshagarg/Desktop/plant_disease_detection/src
python app.py
# Look for: "Running on public URL: https://xxxxx.gradio.live"
```

That URL is immediately shareable! 🚀
