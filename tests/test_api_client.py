# tests/test_api_client.py
import requests
import json

API_BASE_URL = "http://localhost:8000/api"

def test_optimize_team():
    """Test team optimization endpoint."""
    url = f"{API_BASE_URL}/optimize/team"
    params = {
        "budget": 100.0,
        "gameweek": None  # Use current gameweek
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Optimized team for gameweek {data.get('gameweek', 'current')}:")
        print(f"Formation: {data['data']['formation']}")
        print(f"Captain: {data['data']['captain']['player_name']}")
        print(f"Total predicted points: {data['data']['total_points']}")
        return data
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

if __name__ == "__main__":
    test_optimize_team()