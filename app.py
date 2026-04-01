from flask import Flask, request, jsonify, render_template_string
import joblib, json, numpy as np, os

app = Flask(__name__)

BASE = os.path.dirname(__file__)
model    = joblib.load(os.path.join(BASE, 'model.pkl'))
encoders = joblib.load(os.path.join(BASE, 'encoders.pkl'))
with open(os.path.join(BASE, 'unique_vals.json')) as f:
    unique_vals = json.load(f)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>🚗 Car Price Predictor</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px}
  .card{background:rgba(255,255,255,0.05);backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,0.1);border-radius:24px;padding:40px;max-width:800px;width:100%;color:#fff;box-shadow:0 25px 50px rgba(0,0,0,0.5)}
  h1{text-align:center;font-size:2em;margin-bottom:6px;background:linear-gradient(90deg,#e94560,#f5a623);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
  .subtitle{text-align:center;color:#aaa;margin-bottom:32px;font-size:.95em}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  @media(max-width:600px){.grid{grid-template-columns:1fr}}
  .field{display:flex;flex-direction:column;gap:6px}
  label{font-size:.82em;color:#ccc;text-transform:uppercase;letter-spacing:.05em}
  input,select{background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);border-radius:10px;color:#fff;padding:10px 14px;font-size:.95em;outline:none;transition:.2s}
  input:focus,select:focus{border-color:#e94560;background:rgba(255,255,255,0.12)}
  select option{background:#1a1a2e;color:#fff}
  .btn{grid-column:1/-1;background:linear-gradient(90deg,#e94560,#f5a623);border:none;border-radius:12px;color:#fff;font-size:1.1em;font-weight:700;padding:14px;cursor:pointer;margin-top:8px;transition:.2s;letter-spacing:.04em}
  .btn:hover{opacity:.9;transform:translateY(-1px)}
  .result{grid-column:1/-1;background:rgba(233,69,96,0.15);border:1px solid rgba(233,69,96,0.4);border-radius:14px;padding:20px;text-align:center;display:none}
  .result .price{font-size:2.2em;font-weight:800;color:#f5a623}
  .result .label{font-size:.85em;color:#ccc;margin-top:4px}
  .loader{display:none;text-align:center;grid-column:1/-1;color:#aaa}
  .error{grid-column:1/-1;color:#ff6b6b;background:rgba(255,107,107,0.1);border:1px solid rgba(255,107,107,0.3);border-radius:10px;padding:12px;display:none;text-align:center}
  .badge{display:inline-block;background:rgba(245,166,35,0.2);color:#f5a623;border-radius:20px;padding:2px 10px;font-size:.8em;margin-left:6px;vertical-align:middle}
</style>
</head>
<body>
<div class="card">
  <h1>🚗 Car Price Predictor</h1>
  <p class="subtitle">ML-powered · Gradient Boosting · R² = 0.72 <span class="badge">Live</span></p>
  <div class="grid">
    <div class="field">
      <label>Manufacturer</label>
      <select id="Manufacturer">{{ manufacturer_options }}</select>
    </div>
    <div class="field">
      <label>Category</label>
      <select id="Category">{{ category_options }}</select>
    </div>
    <div class="field">
      <label>Production Year</label>
      <input type="number" id="prod_year" value="2018" min="1990" max="2025"/>
    </div>
    <div class="field">
      <label>Fuel Type</label>
      <select id="Fuel type">{{ fuel_options }}</select>
    </div>
    <div class="field">
      <label>Engine Volume (L)</label>
      <input type="number" id="Engine volume" value="2.0" step="0.1" min="0.5" max="8.0"/>
    </div>
    <div class="field">
      <label>Mileage (km)</label>
      <input type="number" id="Mileage" value="50000" step="1000"/>
    </div>
    <div class="field">
      <label>Cylinders</label>
      <input type="number" id="Cylinders" value="4" min="2" max="16"/>
    </div>
    <div class="field">
      <label>Gear Box Type</label>
      <select id="Gear box type">{{ gear_options }}</select>
    </div>
    <div class="field">
      <label>Drive Wheels</label>
      <select id="Drive wheels">{{ drive_options }}</select>
    </div>
    <div class="field">
      <label>Doors</label>
      <input type="number" id="Doors" value="4" min="2" max="6"/>
    </div>
    <div class="field">
      <label>Wheel</label>
      <select id="Wheel">{{ wheel_options }}</select>
    </div>
    <div class="field">
      <label>Color</label>
      <select id="Color">{{ color_options }}</select>
    </div>
    <div class="field">
      <label>Airbags</label>
      <input type="number" id="Airbags" value="4" min="0" max="16"/>
    </div>
    <div class="field">
      <label>Levy ($)</label>
      <input type="number" id="Levy" value="500" min="0"/>
    </div>
    <div class="field">
      <label>Leather Interior</label>
      <select id="Leather interior">
        <option value="Yes">Yes</option>
        <option value="No">No</option>
      </select>
    </div>
    <button class="btn" onclick="predict()">⚡ Predict Price</button>
    <div class="loader" id="loader">Calculating...</div>
    <div class="error" id="error"></div>
    <div class="result" id="result">
      <div class="label">Estimated Market Price</div>
      <div class="price" id="price-val"></div>
      <div class="label" id="price-range"></div>
    </div>
  </div>
</div>
<script>
async function predict(){
  document.getElementById('result').style.display='none';
  document.getElementById('error').style.display='none';
  document.getElementById('loader').style.display='block';
  const payload={
    Levy:+document.getElementById('Levy').value,
    Manufacturer:document.getElementById('Manufacturer').value,
    Category:document.getElementById('Category').value,
    'Fuel type':document.getElementById('Fuel type').value,
    'Engine volume':+document.getElementById('Engine volume').value,
    Mileage:+document.getElementById('Mileage').value,
    Cylinders:+document.getElementById('Cylinders').value,
    'Gear box type':document.getElementById('Gear box type').value,
    'Drive wheels':document.getElementById('Drive wheels').value,
    Doors:+document.getElementById('Doors').value,
    Wheel:document.getElementById('Wheel').value,
    Color:document.getElementById('Color').value,
    Airbags:+document.getElementById('Airbags').value,
    'Prod. year':+document.getElementById('prod_year').value,
    'Leather interior':document.getElementById('Leather interior').value
  };
  try{
    const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const d=await r.json();
    document.getElementById('loader').style.display='none';
    if(d.error){document.getElementById('error').textContent=d.error;document.getElementById('error').style.display='block';return;}
    document.getElementById('price-val').textContent='$'+d.price.toLocaleString();
    document.getElementById('price-range').textContent='Range: $'+d.low.toLocaleString()+' – $'+d.high.toLocaleString();
    document.getElementById('result').style.display='block';
  }catch(e){
    document.getElementById('loader').style.display='none';
    document.getElementById('error').textContent='Server error. Please try again.';
    document.getElementById('error').style.display='block';
  }
}
</script>
</body>
</html>
"""

def make_options(key):
    return "\n".join(f'<option value="{v}">{v}</option>' for v in unique_vals[key])

@app.route('/')
def index():
    html = HTML.replace('{{ manufacturer_options }}', make_options('Manufacturer'))
    html = html.replace('{{ category_options }}',     make_options('Category'))
    html = html.replace('{{ fuel_options }}',          make_options('Fuel type'))
    html = html.replace('{{ gear_options }}',          make_options('Gear box type'))
    html = html.replace('{{ drive_options }}',         make_options('Drive wheels'))
    html = html.replace('{{ wheel_options }}',         make_options('Wheel'))
    html = html.replace('{{ color_options }}',         make_options('Color'))
    return render_template_string(html)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        cat_cols = ['Manufacturer','Category','Fuel type','Gear box type',
                    'Drive wheels','Wheel','Color','Leather interior']
        for c in cat_cols:
            val = str(data[c])
            le  = encoders[c]
            if val not in le.classes_:
                val = le.classes_[0]
            data[c] = int(le.transform([val])[0])

        features = ['Levy','Manufacturer','Category','Fuel type','Engine volume','Mileage',
                    'Cylinders','Gear box type','Drive wheels','Doors','Wheel','Color',
                    'Airbags','Prod. year','Leather interior']
        X = np.array([[data[f] for f in features]])
        price = float(model.predict(X)[0])
        price = max(500, price)
        margin = price * 0.15
        return jsonify(price=round(price,2), low=round(price-margin,2), high=round(price+margin,2))
    except Exception as e:
        return jsonify(error=str(e)), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
