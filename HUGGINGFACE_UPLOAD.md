# Hugging Face Upload Guide (Step-by-Step with Screenshots)

Since you can't see the "Add files" option, here's the alternative method using Git.

---

## Method 1: Using Git (Recommended)

### Step 1: Install Git LFS
```bash
# On macOS
brew install git-lfs

# Initialize Git LFS
git lfs install
```

### Step 2: Create Your Space on Hugging Face

1. Go to https://huggingface.co/spaces
2. Click the **"+"** button (top right, next to your profile)
3. Fill in:
   - **Owner**: Your username
   - **Space name**: `plant-disease-detection`
   - **License**: MIT
   - **Select the SDK**: Choose **Gradio**
   - **Space hardware**: CPU basic (free)
4. Click **"Create Space"**

### Step 3: Copy the Git Repository URL

After creating the Space, you'll see a URL like:
```
https://huggingface.co/spaces/YOUR_USERNAME/plant-disease-detection
```

The Git URL will be:
```
https://huggingface.co/spaces/YOUR_USERNAME/plant-disease-detection.git
```

### Step 4: Clone Your Empty Space

```bash
# Navigate to a temporary directory
cd ~/Desktop

# Clone your Space (replace YOUR_USERNAME)
git clone https://huggingface.co/spaces/YOUR_USERNAME/plant-disease-detection.git

# Enter the directory
cd plant-disease-detection
```

### Step 5: Copy Files from Your Project

```bash
# Copy all necessary files
cp /Users/akankshagarg/Desktop/plant_disease_detection/app.py .
cp /Users/akankshagarg/Desktop/plant_disease_detection/requirements.txt .
cp /Users/akankshagarg/Desktop/plant_disease_detection/src/best_model.pth .
cp /Users/akankshagarg/Desktop/plant_disease_detection/src/classes.json .

# Copy src folder
cp -r /Users/akankshagarg/Desktop/plant_disease_detection/src .

# Optional: Copy README
cp /Users/akankshagarg/Desktop/plant_disease_detection/README.md .
```

### Step 6: Configure Git LFS for Large Files

```bash
# Track the model file with Git LFS
git lfs track "*.pth"

# Add .gitattributes
git add .gitattributes
```

### Step 7: Commit and Push

```bash
# Add all files
git add .

# Commit
git commit -m "Deploy plant disease detection app"

# Push to Hugging Face
git push
```

**Enter your Hugging Face credentials when prompted:**
- Username: Your Hugging Face username
- Password: Use your **Access Token** (not password)

### Step 8: Get Your Access Token

If you don't have an access token:
1. Go to https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Name: `git-access`
4. Role: **write**
5. Click **"Generate"**
6. Copy the token and use it as password when pushing

### Step 9: Wait for Deployment

- Go to your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/plant-disease-detection`
- Click on **"App"** tab
- You'll see "Building..." - this takes 2-5 minutes
- When ready, your app will appear!

---

## Method 2: Using Hugging Face CLI (Alternative)

### Install CLI
```bash
pip install huggingface_hub
```

### Login
```bash
huggingface-cli login
# Paste your access token
```

### Upload Files
```bash
# Navigate to your project
cd /Users/akankshagarg/Desktop/plant_disease_detection

# Upload all files
huggingface-cli upload YOUR_USERNAME/plant-disease-detection . --repo-type=space
```

---

## Method 3: Manual Upload via Web (If Available)

If you see the interface with upload options:

1. **Go to your Space page**
2. Look for these options:
   - **Files and versions** tab (top of page)
   - **+ Add file** button (might be in Files tab)
   - Or **three dots menu** (⋮) → **Add file**

3. **Upload each file:**
   - Drag and drop OR click browse
   - Upload: `app.py`, `requirements.txt`, `classes.json`, `best_model.pth`
   - Create `src/` folder and upload `predict.py`, `model.py`, `dataset.py`, `config.py`, `prepare_data.py`, `train.py`

---

## Files You Need to Upload

Here's the checklist:

### Root Directory Files:
- ✅ `app.py` (created at project root)
- ✅ `requirements.txt`
- ✅ `best_model.pth` (from src/)
- ✅ `classes.json` (from src/)
- ✅ `README.md` (optional)

### src/ Folder:
- ✅ `src/predict.py`
- ✅ `src/model.py`
- ✅ `src/dataset.py`
- ✅ `src/config.py`
- ✅ `src/prepare_data.py`
- ✅ `src/train.py`

---

## Troubleshooting

### "Authentication failed"
- Make sure you're using **Access Token** as password, not your account password
- Get token from: https://huggingface.co/settings/tokens

### "Large file detected"
```bash
# Make sure Git LFS is tracking it
git lfs track "*.pth"
git add .gitattributes
git add best_model.pth
git commit -m "Add model with LFS"
git push
```

### "Module not found" in deployment
- Check `requirements.txt` has all dependencies:
```
torch
torchvision
Pillow
gradio
pandas
scikit-learn
```

### Can't find "Add files" button
- You might need to use Git method (Method 1 above)
- Or use CLI method (Method 2 above)

---

## Quick Command Summary

```bash
# 1. Install Git LFS
brew install git-lfs
git lfs install

# 2. Clone your Space
cd ~/Desktop
git clone https://huggingface.co/spaces/YOUR_USERNAME/plant-disease-detection.git
cd plant-disease-detection

# 3. Copy files
cp /Users/akankshagarg/Desktop/plant_disease_detection/app.py .
cp /Users/akankshagarg/Desktop/plant_disease_detection/requirements.txt .
cp /Users/akankshagarg/Desktop/plant_disease_detection/src/best_model.pth .
cp /Users/akankshagarg/Desktop/plant_disease_detection/src/classes.json .
cp -r /Users/akankshagarg/Desktop/plant_disease_detection/src .

# 4. Setup Git LFS
git lfs track "*.pth"

# 5. Commit and push
git add .
git commit -m "Deploy plant disease detection app"
git push

# 6. Visit your Space
# https://huggingface.co/spaces/YOUR_USERNAME/plant-disease-detection
```

---

## After Deployment

Your app will be live at:
- **Main URL**: `https://huggingface.co/spaces/YOUR_USERNAME/plant-disease-detection`
- **Direct app**: `https://YOUR_USERNAME-plant-disease-detection.hf.space`

**Features:**
- ✅ Always online (24/7)
- ✅ No need to keep computer on
- ✅ Free forever
- ✅ Shareable permanent link

---

## Need Help?

If you're still stuck:
1. Show me a screenshot of your Hugging Face Space page
2. Or tell me what you see when you open your Space
3. I'll guide you through the exact steps
