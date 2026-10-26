# 📱 Mobile Price Estimator

A small Flask website that estimates a smartphone's price from its specifications using
**Linear Regression** (scikit-learn).

## Project structure

```
├── app.py            # Flask app (routes: /, /predict, /health)
├── train_model.py    # Trains the Linear Regression pipeline and writes model.pkl
├── model.pkl         # Trained model (generated)
├── requirements.txt
├── .gitignore
├── Dockerfile
├── README.md
├── templates/
│   └── index.html    # Web UI
├── static/
│   └── style.css
└── tests/
    └── test_app.py   # pytest tests
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python train_model.py            # optional: model.pkl is already included
python app.py                    # open http://localhost:5000
```

## Features used

| Feature | Meaning |
|---|---|
| `ram_gb` | RAM in GB |
| `storage_gb` | Internal storage in GB |
| `battery_mah` | Battery capacity in mAh |
| `camera_mp` | Main camera resolution in MP |
| `screen_inch` | Screen size in inches |
| `refresh_rate` | Display refresh rate in Hz |
| `has_5g` | 1 if 5G is supported, else 0 |
| `brand_tier` | 1 = budget, 2 = mid-range, 3 = premium |

## Use your own data

The bundled model is trained on a **synthetic** dataset, so prices are illustrative only.
To train on real data, provide a CSV with the columns above plus a `price` column:

```bash
python train_model.py data/phones.csv
```

## API

`POST /predict`

```json
{ "ram_gb": 8, "storage_gb": 128, "battery_mah": 5000, "camera_mp": 50,
  "screen_inch": 6.5, "refresh_rate": 120, "has_5g": 1, "brand_tier": 2 }
```

Response:

```json
{ "price": 512.4, "range": [461.16, 563.64] }
```

## Tests

```bash
pytest -v
```

## Docker

```bash
docker build -t mobile-price-estimator .
docker run -p 5000:5000 mobile-price-estimator
```

## How it works

1. Features are standardised with `StandardScaler`.
2. `LinearRegression` fits price = b0 + b1·x1 + ... + bn·xn by least squares.
3. The pipeline is saved with `joblib` to `model.pkl` and loaded by Flask.
4. The UI shows the estimate with a ±10% range.
