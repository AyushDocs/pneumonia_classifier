
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

def test_read_main():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "PnuemoCheck AI" in response.text

def test_history_endpoint():
    with TestClient(app) as client:
        response = client.get("/history/TEST-ID")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "history" in data

def test_invalid_upload():
    # Test without file
    with TestClient(app) as client:
        response = client.post("/predict", data={"patient_id": "TEST-ID"})
        assert response.json()["status"] == "error"

def test_ood_detection():
    # Create a dummy non-medical image (solid black)
    import io

    from PIL import Image
    img = Image.new('RGB', (224, 224), color='black')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", img_byte_arr, "image/jpeg")},
            data={"patient_id": "TEST-ID"}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "error"
        assert "lacks sufficient contrast" in response.json()["message"]
        assert response.status_code == 200
        assert response.json()["status"] == "error"
        assert "lacks sufficient contrast" in response.json()["message"]
