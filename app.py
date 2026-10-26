"""Flask web app that estimates a smartphone's price from its specs using Linear Regression."""
import os
import re

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from train_model import FEATURES

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
INR_PER_MODEL_USD = 83
BRAND_TIERS = {
    "Samsung": 3,
    "Apple": 3,
    "Google": 3,
    "OnePlus": 3,
    "Honor": 2,
    "Huawei": 2,
    "iQOO": 2,
    "Motorola": 2,
    "Nothing": 2,
    "Oppo": 2,
    "POCO": 2,
    "Realme": 2,
    "Vivo": 2,
    "Xiaomi": 2,
    "Infinix": 1,
    "Nokia": 1,
    "Redmi": 1,
    "Tecno": 1,
}
PHONE_MODELS = {
    "Samsung": [
        "Galaxy S25 Ultra", "Galaxy S25+", "Galaxy S25", "Galaxy S24 Ultra",
        "Galaxy S24+", "Galaxy S24", "Galaxy S24 FE", "Galaxy Z Fold6",
        "Galaxy Z Flip6", "Galaxy Z Fold5", "Galaxy Z Flip5", "Galaxy A56",
        "Galaxy A55", "Galaxy A54", "Galaxy A36", "Galaxy A35", "Galaxy A26",
        "Galaxy A25", "Galaxy M55", "Galaxy M35",
    ],
    "Apple": [
        "iPhone 16 Pro Max", "iPhone 16 Pro", "iPhone 16 Plus", "iPhone 16",
        "iPhone 16e", "iPhone 15 Pro Max", "iPhone 15 Pro", "iPhone 15 Plus",
        "iPhone 15", "iPhone 14 Pro Max", "iPhone 14 Pro", "iPhone 14 Plus",
        "iPhone 14", "iPhone 13 Pro Max", "iPhone 13 Pro", "iPhone 13",
        "iPhone 13 mini", "iPhone 12 Pro Max", "iPhone 12 Pro", "iPhone 12",
    ],
    "Google": [
        "Pixel 9 Pro Fold", "Pixel 9 Pro XL", "Pixel 9 Pro", "Pixel 9",
        "Pixel 9a", "Pixel 8 Pro", "Pixel 8", "Pixel 8a", "Pixel 7 Pro",
        "Pixel 7", "Pixel 7a", "Pixel 6 Pro", "Pixel 6", "Pixel 6a",
        "Pixel 5", "Pixel 5a", "Pixel 4 XL", "Pixel 4", "Pixel 4a 5G",
        "Pixel 4a",
    ],
    "OnePlus": [
        "OnePlus 13", "OnePlus 13R", "OnePlus 12", "OnePlus 12R",
        "OnePlus 11", "OnePlus 11R", "OnePlus 10 Pro", "OnePlus 10T",
        "OnePlus 10R", "OnePlus 9 Pro", "OnePlus 9", "OnePlus 9R",
        "OnePlus 8T", "OnePlus 8 Pro", "OnePlus 8", "OnePlus 7T Pro",
        "OnePlus 7T", "OnePlus Nord 4", "OnePlus Nord 3", "OnePlus Nord CE4",
    ],
    "Honor": [
        "Honor Magic7 Pro", "Honor Magic7", "Honor Magic6 Pro", "Honor Magic6",
        "Honor Magic V3", "Honor Magic V2", "Honor 200 Pro", "Honor 200",
        "Honor 200 Lite", "Honor 100 Pro", "Honor 100", "Honor 90",
        "Honor 90 Lite", "Honor X9c", "Honor X9b", "Honor X9a", "Honor X8b",
        "Honor X8a", "Honor X7b", "Honor X6b",
    ],
    "Huawei": [
        "Huawei Pura 70 Ultra", "Huawei Pura 70 Pro", "Huawei Pura 70",
        "Huawei Mate XT", "Huawei Mate X6", "Huawei Mate X5", "Huawei Mate X3",
        "Huawei Mate 60 Pro+", "Huawei Mate 60 Pro", "Huawei Mate 60",
        "Huawei P60 Pro", "Huawei P60", "Huawei nova 13 Pro", "Huawei nova 13",
        "Huawei nova 12 Pro", "Huawei nova 12", "Huawei nova 11 Pro",
        "Huawei nova 11", "Huawei nova Y91", "Huawei nova Y72",
    ],
    "iQOO": [
        "iQOO 13", "iQOO 12", "iQOO 12 Pro", "iQOO 11", "iQOO 11 Pro",
        "iQOO 11S", "iQOO 10", "iQOO 10 Pro", "iQOO 9", "iQOO 9 Pro",
        "iQOO 9T",         "iQOO Neo 10 Pro", "iQOO Neo 10", "iQOO Neo 9 Pro",
        "iQOO Neo 7 Pro", "iQOO Neo 7", "iQOO Neo 6", "iQOO Z9 Turbo",
        "iQOO Z9", "iQOO Z7 Pro",
    ],
    "Motorola": [
        "Motorola Edge 50 Ultra", "Motorola Edge 50 Pro", "Motorola Edge 50",
        "Motorola Edge 50 Fusion", "Motorola Edge 40 Pro", "Motorola Edge 40",
        "Motorola Edge 30 Ultra", "Motorola Edge 30 Pro", "Motorola Edge 30",
        "Motorola Razr 50 Ultra", "Motorola Razr 50", "Motorola Razr 40 Ultra",
        "Motorola Razr 40", "Moto G85", "Moto G75", "Moto G64", "Moto G55",
        "Moto G54", "Moto G34", "Moto G24",
    ],
    "Nothing": [
        "Nothing Phone (3)", "Nothing Phone (3a) Pro", "Nothing Phone (3a)",
        "Nothing Phone (2)", "Nothing Phone (2a) Plus", "Nothing Phone (2a)",
        "Nothing Phone (1)", "CMF Phone 2 Pro", "CMF Phone 1",
        "Nothing Phone (2a) Community Edition", "Nothing Phone (2a) Special Edition",
        "Nothing Phone (2a) Blue Edition", "Nothing Phone (2a) Black Edition",
        "Nothing Phone (2a) White Edition", "Nothing Phone (2a) 5G",
        "Nothing Phone (2) 5G", "Nothing Phone (1) 5G",
        "CMF Phone 1 Orange Edition", "CMF Phone 1 Black Edition",
        "CMF Phone 1 Blue Edition",
    ],
    "Oppo": [
        "OPPO Find X8 Pro", "OPPO Find X8", "OPPO Find X7 Ultra", "OPPO Find X7",
        "OPPO Find N5", "OPPO Find N3", "OPPO Find N3 Flip", "OPPO Reno13 Pro",
        "OPPO Reno13", "OPPO Reno12 Pro", "OPPO Reno12", "OPPO Reno11 Pro",
        "OPPO Reno11", "OPPO Reno10 Pro+", "OPPO Reno10 Pro", "OPPO Reno10",
        "OPPO F27 Pro+", "OPPO F27", "OPPO A3 Pro", "OPPO A79 5G",
    ],
    "POCO": [
        "POCO F7 Ultra", "POCO F7 Pro", "POCO F6 Pro", "POCO F6",
        "POCO F5 Pro", "POCO F5", "POCO F4 GT", "POCO F4", "POCO X7 Pro",
        "POCO X7", "POCO X6 Pro", "POCO X6", "POCO X5 Pro", "POCO X5",
        "POCO X4 Pro 5G", "POCO X4 GT", "POCO M7 Pro 5G", "POCO M6 Pro",
        "POCO M6 5G", "POCO C75",
    ],
    "Realme": [
        "realme GT 7 Pro", "realme GT 6", "realme GT 6T", "realme GT 5 Pro",
        "realme GT 5", "realme GT Neo 6", "realme GT Neo 5", "realme 14 Pro+",
        "realme 14 Pro", "realme 13 Pro+", "realme 13 Pro", "realme 13+",
        "realme 12 Pro+", "realme 12 Pro", "realme 12+", "realme 12",
        "realme 11 Pro+", "realme 11 Pro", "realme Narzo 70 Pro",
        "realme Narzo 70",
    ],
    "Vivo": [
        "vivo X200 Pro", "vivo X200", "vivo X100 Ultra", "vivo X100 Pro",
        "vivo X100", "vivo X90 Pro+", "vivo X90 Pro", "vivo X90",
        "vivo V40 Pro", "vivo V40", "vivo V30 Pro", "vivo V30",
        "vivo V29 Pro", "vivo V29", "vivo V27 Pro", "vivo V27",
        "vivo T3 Ultra", "vivo T3 Pro", "vivo T3", "vivo Y200 Pro",
    ],
    "Xiaomi": [
        "Xiaomi 15 Ultra", "Xiaomi 15", "Xiaomi 14 Ultra", "Xiaomi 14 Pro",
        "Xiaomi 14", "Xiaomi 13 Ultra", "Xiaomi 13 Pro", "Xiaomi 13",
        "Xiaomi 13T Pro", "Xiaomi 13T", "Xiaomi 12 Pro", "Xiaomi 12",
        "Xiaomi 12T Pro", "Xiaomi 12T", "Xiaomi 11T Pro", "Xiaomi 11T",
        "Xiaomi MIX Flip", "Xiaomi MIX Fold 4", "Xiaomi Civi 4 Pro",
        "Xiaomi Civi 3",
    ],
    "Infinix": [
        "Infinix ZERO 40", "Infinix ZERO 40 5G", "Infinix ZERO 30 5G",
        "Infinix ZERO 30", "Infinix ZERO 20", "Infinix NOTE 50 Pro+ 5G",
        "Infinix NOTE 50 Pro", "Infinix NOTE 50", "Infinix NOTE 40 Pro+ 5G",
        "Infinix NOTE 40 Pro 5G", "Infinix NOTE 40 Pro", "Infinix NOTE 40",
        "Infinix NOTE 30 VIP", "Infinix NOTE 30 Pro", "Infinix NOTE 30",
        "Infinix HOT 50 Pro+", "Infinix HOT 50 Pro", "Infinix HOT 50",
        "Infinix HOT 40 Pro", "Infinix HOT 40i",
    ],
    "Nokia": [
        "Nokia X30 5G", "Nokia X20", "Nokia X10", "Nokia XR21", "Nokia XR20",
        "Nokia G60 5G", "Nokia G50", "Nokia G42 5G", "Nokia G41", "Nokia G40",
        "Nokia G22", "Nokia G21", "Nokia G20", "Nokia G11 Plus", "Nokia G11",
        "Nokia G10", "Nokia C32", "Nokia C31", "Nokia C30", "Nokia C22",
    ],
    "Redmi": [
        "Redmi Note 14 Pro+ 5G", "Redmi Note 14 Pro 5G", "Redmi Note 14",
        "Redmi Note 13 Pro+ 5G", "Redmi Note 13 Pro 5G", "Redmi Note 13 Pro",
        "Redmi Note 13 5G", "Redmi Note 13", "Redmi Note 12 Pro+ 5G",
        "Redmi Note 12 Pro 5G", "Redmi Note 12 5G", "Redmi Note 12",
        "Redmi Note 11 Pro+ 5G", "Redmi Note 11 Pro", "Redmi Note 11S",
        "Redmi Note 11", "Redmi 13 5G", "Redmi 13C 5G", "Redmi 13C",
        "Redmi A3",
    ],
    "Tecno": [
        "TECNO Phantom V Fold2", "TECNO Phantom V Flip2", "TECNO Phantom V Fold",
        "TECNO Phantom V Flip", "TECNO Phantom X2 Pro", "TECNO Phantom X2",
        "TECNO Camon 30 Premier", "TECNO Camon 30 Pro", "TECNO Camon 30",
        "TECNO Camon 20 Premier", "TECNO Camon 20 Pro", "TECNO Camon 20",
        "TECNO Pova 6 Pro", "TECNO Pova 6", "TECNO Pova 5 Pro",
        "TECNO Pova 5", "TECNO Spark 30 Pro", "TECNO Spark 30",
        "TECNO Spark 20 Pro+", "TECNO Spark 20",
    ],
}

# Allowed (min, max) per feature
LIMITS = {
    "ram_gb": (1, 24),
    "storage_gb": (8, 2048),
    "battery_mah": (1000, 10000),
    "camera_mp": (2, 250),
    "screen_inch": (3.5, 8.5),
    "refresh_rate": (30, 240),
    "has_5g": (0, 1),
    "brand_tier": (1, 3),
}


def load_model():
    if not os.path.exists(MODEL_PATH):
        import train_model
        df = train_model.load_dataset()
        model = train_model.build_pipeline().fit(df[FEATURES], df[train_model.TARGET])
        joblib.dump(model, MODEL_PATH)
        return model
    return joblib.load(MODEL_PATH)


def default_memory_for_model(brand, model):
    """Return a representative RAM/storage configuration; variants may differ."""
    name = model.casefold()
    if brand == "Apple":
        generation_match = re.search(r"\d+", name)
        generation = int(generation_match.group()) if generation_match else 16
        ram = 8 if generation >= 15 else 6 if generation >= 13 else 4
        storage = 256 if "pro max" in name else 128 if generation >= 13 else 64
    elif brand == "Samsung":
        if any(term in name for term in ("ultra", "fold")):
            ram, storage = 12, 256
        elif name.startswith("galaxy s"):
            ram, storage = 8, 256 if "+" in model else 128
        elif any(term in name for term in ("a5", "m55")):
            ram, storage = 8, 128
        else:
            ram, storage = 6, 128
    elif brand == "Google":
        ram = 12 if any(term in name for term in ("pro", "fold")) else 8
        storage = 256 if any(term in name for term in ("pro", "fold")) else 128
    elif brand in {"OnePlus", "iQOO", "Xiaomi"}:
        premium = any(term in name for term in ("pro", "ultra", "fold", "plus"))
        ram, storage = (12, 256) if premium else (8, 128)
        if brand == "OnePlus" and "nord" in name:
            ram, storage = 8, 256
    elif brand == "Honor":
        ram, storage = (12, 256) if "magic" in name or "pro" in name else (8, 128)
    elif brand == "Huawei":
        ram, storage = (12, 256) if any(term in name for term in ("pura", "mate", "pro")) else (8, 128)
    elif brand == "Motorola":
        ram, storage = (12, 256) if any(term in name for term in ("ultra", "pro", "razr")) else (8, 128)
        if "g24" in name or "g34" in name:
            ram = 4
    elif brand == "Nothing":
        ram, storage = (12, 256) if "phone (3)" in name or "phone (2)" in name else (8, 128)
    elif brand == "Oppo":
        ram, storage = (12, 256) if any(term in name for term in ("find", "pro")) else (8, 128)
    elif brand in {"POCO", "Redmi"}:
        ram, storage = (8, 256) if any(term in name for term in ("pro", "ultra", "f7", "f6")) else (6, 128)
    elif brand == "Realme":
        ram, storage = (12, 256) if "gt" in name or "pro+" in name else (8, 128)
    elif brand == "Vivo":
        ram, storage = (12, 256) if name.startswith("vivo x") else (8, 128)
        if "pro" in name:
            ram, storage = 8, 256
    elif brand == "Infinix":
        ram, storage = (8, 256) if "note" in name or "zero" in name else (8, 128)
    elif brand == "Nokia":
        ram, storage = (8, 128) if "x" in name or "xr" in name else (6, 128)
        if "c" in name:
            ram, storage = 4, 64
    elif brand == "Tecno":
        ram, storage = (12, 256) if "phantom" in name else (8, 256) if "camon" in name else (8, 128)
    else:
        ram, storage = 8, 128
    return ram, storage


def estimate_catalog_starting_price(brand, model):
    """Return an indicative India starting price, not a verified launch price."""
    numbers = [int(value) for value in re.findall(r"\d+", model)]
    generation = numbers[0] if numbers else 0
    name = model.casefold()

    if brand == "Huawei":
        return None
    if brand == "Apple":
        base = {16: 79900, 15: 79900, 14: 69900, 13: 59900, 12: 64900}.get(generation, 49900)
        if "pro max" in name:
            return base + 65000
        if "pro" in name:
            return base + 40000
        if "plus" in name:
            return base + 10000
        if "mini" in name or name.endswith("e"):
            return base - 10000
        return base
    if brand == "Samsung":
        if "fold" in name:
            return max(139999, 174999 - max(generation - 4, 0) * 10000)
        if "flip" in name:
            return max(89999, 119999 - max(generation - 4, 0) * 5000)
        if name.startswith("galaxy s"):
            base = {
                25: 79999, 24: 79999, 23: 79999, 22: 72999,
                21: 69999, 20: 66999,
            }.get(generation, 69999)
            return base + (50000 if "ultra" in name else 20000 if "+" in model else 0)
        if "galaxy a" in name:
            return 39999 if generation >= 50 else 29999 if generation >= 30 else 21999
        return 24999 if "m55" in name else 17999
    if brand == "Google":
        base = {9: 79999, 8: 75999, 7: 59999, 6: 59999, 5: 69999, 4: 69999}.get(generation, 49999)
        return base + (80000 if "fold" in name else 25000 if "pro" in name else -35000 if "a" in name else 0)
    if brand == "OnePlus":
        if "nord" in name:
            return 24999 if "ce" in name else 29999 if generation >= 4 else 33999
        base = {13: 69999, 12: 64999, 11: 56999, 10: 49999, 9: 49999, 8: 41999, 7: 34999}.get(generation, 39999)
        return base - 18000 if re.search(r"\dR\b", model, re.IGNORECASE) else base
    if brand == "Honor":
        if "magic" in name:
            return 89999 if "pro" in name or "fold" in name else 69999
        return 39999 if "pro" in name else 29999 if "200" in name else 24999
    if brand == "iQOO":
        if "neo" in name:
            return 39999 if "pro" in name else 34999
        if "z" in name:
            return 24999
        return {13: 54999, 12: 52999, 11: 44999, 10: 39999, 9: 33999}.get(generation, 39999)
    if brand == "Motorola":
        if "razr" in name:
            return 99999 if "ultra" in name else 69999
        if "edge" in name:
            return 59999 if "ultra" in name else 34999 if "pro" in name else 27999
        return 22999 if generation >= 80 else 17999 if generation >= 50 else 12999
    if brand == "Nothing":
        if "cmf" in name:
            return 17999
        return 79999 if "(3)" in name else 44999 if "(2)" in name else 32999
    if brand == "Oppo":
        if "find n" in name:
            return 139999
        if "find x" in name:
            return 99999 if "pro" in name or "ultra" in name else 69999
        if "reno" in name:
            return 44999 if "pro" in name else 32999
        return 27999 if "f" in name else 17999
    if brand == "POCO":
        if "f" in name:
            return 34999 if "pro" in name else 29999
        if "x" in name:
            return 24999 if "pro" in name else 18999
        if "m" in name:
            return 13999
        return 8999
    if brand == "Realme":
        if "gt" in name:
            return 59999 if "pro" in name else 39999
        if "pro+" in name:
            return 34999
        if "pro" in name:
            return 29999
        if "narzo" in name:
            return 19999
        return 19999
    if brand == "Vivo":
        if re.search(r"\bx\d", name):
            return 99999 if "pro" in name or "ultra" in name else 79999
        if re.search(r"\bv\d", name):
            return 49999 if "pro" in name else 34999
        if re.search(r"\bt\d", name):
            return 24999
        return 17999
    if brand == "Xiaomi":
        if "mix fold" in name:
            return 99999
        if "mix flip" in name:
            return 69999
        if "civi" in name:
            return 39999
        if "t pro" in name:
            return 49999
        return 99999 if "ultra" in name else 79999 if "pro" in name else 69999
    if brand == "Infinix":
        if "zero" in name:
            return 29999
        if "note" in name:
            return 19999 if "pro" in name else 14999
        return 12999 if "pro" in name else 9999
    if brand == "Nokia":
        if "x" in name or "xr" in name:
            return 39999
        if "g" in name:
            return 17999 if generation >= 40 else 12999
        return 9999
    if brand == "Redmi":
        if "note" in name:
            return 32999 if "pro+" in name else 24999 if "pro" in name else 17999
        if name.startswith("redmi a"):
            return 7999
        return 11999
    if brand == "Tecno":
        if "phantom" in name:
            return 89999 if "fold" in name else 69999
        if "camon" in name:
            return 29999 if "premier" in name or "pro" in name else 22999
        if "pova" in name:
            return 17999
        return 12999 if "pro" in name else 9999
    return None


def parse_features(data):
    """Return (values, error). Validates presence, type and range of every feature."""
    if "brand" in data:
        brand = str(data.get("brand", "")).strip().casefold()
        brand_tiers = {name.casefold(): tier for name, tier in BRAND_TIERS.items()}
        if brand not in brand_tiers:
            return None, "'brand' must be a supported mobile company."
        data = data.copy()
        data["brand_tier"] = brand_tiers[brand]

    values = {}
    for name in FEATURES:
        raw = data.get(name)
        if raw is None or str(raw).strip() == "":
            return None, f"Missing value for '{name}'."
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None, f"'{name}' must be a number."
        low, high = LIMITS[name]
        if not low <= value <= high:
            return None, f"'{name}' must be between {low} and {high}."
        values[name] = value
    return values, None


def create_app():
    app = Flask(__name__)
    model = load_model()

    @app.route("/")
    def index():
        model_prices = {
            brand: {
                model: estimate_catalog_starting_price(brand, model)
                for model in models
            }
            for brand, models in PHONE_MODELS.items()
        }
        model_memory = {
            brand: {
                model: default_memory_for_model(brand, model)
                for model in models
            }
            for brand, models in PHONE_MODELS.items()
        }
        return render_template(
            "index.html", limits=LIMITS, brands=BRAND_TIERS,
            phone_models=PHONE_MODELS, model_prices=model_prices,
            model_memory=model_memory
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json(silent=True) or request.form
        values, error = parse_features(data)
        if error:
            return jsonify({"error": error}), 400
        price = float(model.predict(pd.DataFrame([values], columns=FEATURES))[0])
        price = max(price, 0.0) * INR_PER_MODEL_USD
        return jsonify({
            "price": round(price, 2),
            "range": [round(price * 0.9, 2), round(price * 1.1, 2)],
        })

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
