# 🚗 Car Price Predictor — Flask + ML

A machine learning web app that predicts car prices using Gradient Boosting.

## Model Performance
- **Algorithm**: Gradient Boosting Regressor  
- **R² Score**: 0.72  
- **MAE**: ~$5,409  
- **Dataset**: 19,237 car listings

## Files
| File | Purpose |
|------|---------|
| `app.py` | Flask web application |
| `model.pkl` | Trained ML model |
| `encoders.pkl` | Label encoders for categorical features |
| `unique_vals.json` | Valid dropdown values |
| `requirements.txt` | Python dependencies |
| `render.yaml` | Render deployment config |
| `Procfile` | Gunicorn start command |

## Deploy to Render (Step-by-Step)

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit - Car Price Predictor"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/car-price-predictor.git
git push -u origin main
```

### 2. Create Render Account
- Go to https://render.com and sign up (free)
- Connect your GitHub account

### 3. Deploy on Render
1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub repo
3. Set these settings:
   - **Name**: `car-price-predictor`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
4. Click **"Create Web Service"**
5. Wait ~3 minutes for deployment
6. Your live URL will be: `https://car-price-predictor.onrender.com`

## Local Run
```bash
pip install -r requirements.txt
python app.py
# Visit http://localhost:5000
```
