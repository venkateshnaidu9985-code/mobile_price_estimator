import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (  # noqa: E402
    PHONE_MODELS,
    create_app,
    default_memory_for_model,
    estimate_catalog_starting_price,
)

BASE = {
    "ram_gb": 8, "storage_gb": 128, "battery_mah": 5000, "camera_mp": 50,
    "screen_inch": 6.5, "refresh_rate": 120, "has_5g": 1, "brand_tier": 2,
}


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_index_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Mobile Price Estimator" in res.data
    assert b'name="phone_model"' in res.data
    assert b'<optgroup label="Samsung" data-brand="Samsung">' in res.data
    assert b'data-launch-price="129999"' in res.data
    assert b'data-default-ram="12"' in res.data
    assert b'data-default-storage="256"' in res.data
    assert b'group.hidden = group.dataset.brand !== brandSelect.value' in res.data
    assert b"Selecting a model fills suggested RAM and storage defaults" in res.data
    assert set(PHONE_MODELS) == {
        "Samsung", "Apple", "Google", "OnePlus", "Honor", "Huawei", "iQOO",
        "Motorola", "Nothing", "Oppo", "POCO", "Realme", "Vivo", "Xiaomi",
        "Infinix", "Nokia", "Redmi", "Tecno",
    }


@pytest.mark.parametrize("brand,models", PHONE_MODELS.items())
def test_each_company_has_20_distinct_phone_models(brand, models):
    assert len(models) == 20, brand
    assert len(set(models)) == 20, brand


def test_catalog_price_guide_is_labeled_approximate():
    assert estimate_catalog_starting_price("Samsung", "Galaxy S25 Ultra") == 129999
    assert estimate_catalog_starting_price("Apple", "iPhone 16 Pro Max") == 144900
    assert estimate_catalog_starting_price("OnePlus", "OnePlus 13R") == 51999
    assert estimate_catalog_starting_price("OnePlus", "OnePlus 13") == 69999
    assert estimate_catalog_starting_price("Huawei", "Huawei Pura 70") is None


def test_model_selection_memory_defaults():
    assert default_memory_for_model("Samsung", "Galaxy S25 Ultra") == (12, 256)
    assert default_memory_for_model("Apple", "iPhone 16 Pro Max") == (8, 256)
    assert default_memory_for_model("Nokia", "Nokia C32") == (4, 64)


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.get_json() == {"status": "ok"}


def test_predict_returns_price(client):
    res = client.post("/predict", json=BASE)
    data = res.get_json()
    assert res.status_code == 200
    assert data["price"] > 1000
    assert data["range"][0] < data["price"] < data["range"][1]


def test_predict_accepts_mobile_company(client):
    res = client.post("/predict", json={**BASE, "brand": "Apple"})
    assert res.status_code == 200
    assert res.get_json()["price"] > 1000


def test_predict_converts_price_to_inr(monkeypatch):
    class FixedModel:
        def predict(self, features):
            return [100]

    monkeypatch.setattr("app.load_model", lambda: FixedModel())
    client = create_app().test_client()
    res = client.post("/predict", json={**BASE, "brand": "Apple"})

    assert res.status_code == 200
    assert res.get_json() == {"price": 8300, "range": [7470, 9130]}


def test_predict_rejects_unsupported_mobile_company(client):
    res = client.post("/predict", json={**BASE, "brand": "Unsupported"})
    assert res.status_code == 400
    assert "brand" in res.get_json()["error"]


def test_better_specs_cost_more(client):
    low = client.post("/predict", json={**BASE, "ram_gb": 4, "storage_gb": 64, "brand_tier": 1}).get_json()
    high = client.post("/predict", json={**BASE, "ram_gb": 16, "storage_gb": 512, "brand_tier": 3}).get_json()
    assert high["price"] > low["price"]


def test_missing_field(client):
    payload = {k: v for k, v in BASE.items() if k != "ram_gb"}
    res = client.post("/predict", json=payload)
    assert res.status_code == 400
    assert "ram_gb" in res.get_json()["error"]


def test_non_numeric_value(client):
    res = client.post("/predict", json={**BASE, "battery_mah": "abc"})
    assert res.status_code == 400


def test_out_of_range_value(client):
    res = client.post("/predict", json={**BASE, "ram_gb": 999})
    assert res.status_code == 400
