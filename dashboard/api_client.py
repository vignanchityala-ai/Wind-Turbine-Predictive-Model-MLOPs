import requests
import os

class WindTurbineAPIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.api_key = os.environ.get("API_KEY", "")
        
    def _get_headers(self):
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers
    
    def predict(self, dataset_name, readings):
        return requests.post(f"{self.base_url}/predict/{dataset_name}", json={"readings": readings}, headers=self._get_headers())
    
    def batch_predict(self, dataset_name, csv_file_bytes):
        # We need to send file bytes via requests
        files = {"file": ("data.csv", csv_file_bytes, "text/csv")}
        return requests.post(f"{self.base_url}/batch_predict/{dataset_name}", files=files, headers=self._get_headers())
    
    def get_farms(self):
        resp = requests.get(f"{self.base_url}/farms", headers=self._get_headers())
        if resp.status_code == 200:
            return resp.json()
        return {"farms": {}}
    
    def get_model_info(self, name):
        resp = requests.get(f"{self.base_url}/model/info/{name}", headers=self._get_headers())
        if resp.status_code == 200:
            return resp.json()
        return {}
    
    def get_health(self):
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=2)
            if resp.status_code == 200:
                return resp.json()
        except requests.RequestException:
            pass
        return {"status": "offline", "models_available": 0}
