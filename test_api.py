#!/usr/bin/env python3
"""
Simple script to test the Flask API like curl
"""
import requests
import sys

def test_forge_api(token="ETH"):
    """Test the forge API endpoint"""
    try:
        url = f"http://127.0.0.1:9000/forge/{token}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print(response.text.strip())
            return True
        else:
            print(f"Error: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
        return False

if __name__ == "__main__":
    token = sys.argv[1] if len(sys.argv) > 1 else "ETH"
    test_forge_api(token)