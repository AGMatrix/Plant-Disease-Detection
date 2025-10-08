# How to Run Your App

## The Problem You're Seeing

When you get "no interface running" on the public URL, it means your local app isn't running. The Gradio share link only works when your app is actively running on your computer.

---

## Quick Fix - Run the App

### Step 1: Activate Virtual Environment

```bash
cd /Users/akankshagarg/Desktop/plant_disease_detection

# Activate virtual environment
source venv/bin/activate
```

You'll see `(venv)` appear in your terminal prompt.

### Step 2: Navigate to src folder

```bash
cd src
```

### Step 3: Run the app

```bash
python app.py
```

### Step 4: Get Your Public URL

You'll see output like:
```
Launching Plant Disease Detection Interface...
Loading DINOv2 model...
DINOv2 predictor ready on cpu

Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://abc123xyz.gradio.live

To create a public link, set `share=True` in `launch()`.
```

**The public URL (https://abc123xyz.gradio.live) is what you share with others!**

---

## One-Line Command

```bash
cd /Users/akankshagarg/Desktop/plant_disease_detection && source venv/bin/activate && cd src && python app.py
```

---

## Important Notes

### ⚠️ Keep Terminal Open
- The public URL **only works while your terminal is running**
- If you close the terminal, the URL stops working
- Press `Ctrl+C` to stop the app

### ⚠️ URL Changes Each Time
- Every time you restart the app, you get a **new public URL**
- The old URL won't work anymore

### ⚠️ 72-Hour Limit
- Gradio share links expire after 72 hours maximum
- You'll need to restart for a new URL

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'gradio'"

**Problem**: Virtual environment not activated

**Solution**:
```bash
source /Users/akankshagarg/Desktop/plant_disease_detection/venv/bin/activate
```

### "Address already in use"

**Problem**: Port 7860 is already being used

**Solution**:
```bash
# Kill existing process
lsof -ti:7860 | xargs kill -9

# Or change port in app.py
interface.launch(share=True, server_port=7861)  # Use different port
```

### "Model file not found"

**Problem**: Not in correct directory

**Solution**: Make sure you're in the `src` folder:
```bash
cd /Users/akankshagarg/Desktop/plant_disease_detection/src
ls -la  # Should see best_model.pth and classes.json
```

---

## Alternative: Run Without Virtual Environment Issues

If you don't want to deal with virtual environment:

### Option 1: Install globally (not recommended)
```bash
pip3 install gradio torch torchvision pillow pandas scikit-learn
cd /Users/akankshagarg/Desktop/plant_disease_detection/src
python3 app.py
```

### Option 2: Use absolute path to venv python
```bash
cd /Users/akankshagarg/Desktop/plant_disease_detection/src
../venv/bin/python app.py
```

---

## Better Solution: Create a Run Script

Create a script to make it easier:

```bash
# Create run.sh in project root
cat > /Users/akankshagarg/Desktop/plant_disease_detection/run.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
cd src
python app.py
EOF

# Make it executable
chmod +x /Users/akankshagarg/Desktop/plant_disease_detection/run.sh
```

Then just run:
```bash
./run.sh
```

---

## For Permanent URL (No Need to Keep Terminal Open)

If you want a URL that works without keeping your computer on:

1. **Hugging Face Spaces** (Free, recommended)
   - See DEPLOYMENT.md for full guide
   - URL stays active 24/7
   - No need to keep your computer on

2. **Google Colab**
   - Upload notebook and run there
   - Get public URL
   - Free GPU available

---

## Quick Reference

```bash
# Start app (full command)
cd /Users/akankshagarg/Desktop/plant_disease_detection && \
source venv/bin/activate && \
cd src && \
python app.py

# Stop app
# Press Ctrl+C in terminal

# Check if running
lsof -i :7860
```

---

## Summary

1. **Activate venv**: `source venv/bin/activate`
2. **Go to src**: `cd src`
3. **Run app**: `python app.py`
4. **Copy public URL** from terminal output
5. **Share URL** - it works while terminal is open
6. **Keep terminal open** to keep URL active

---

**Need help?** The public URL only works when:
- ✅ App is running in terminal
- ✅ Virtual environment is activated
- ✅ Terminal window stays open
- ✅ Within 72-hour limit
