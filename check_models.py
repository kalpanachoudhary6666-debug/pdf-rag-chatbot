"""
Run this once to find which Gemini models are available for your API key.
Usage: python check_models.py
"""
import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("ERROR: GOOGLE_API_KEY not found in .env file")
    exit(1)

print(f"Checking models for your API key...\n")

# Try v1 first
for version in ["v1", "v1beta"]:
    url = f"https://generativelanguage.googleapis.com/{version}/models?key={api_key}"
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"{version}: Failed ({resp.status_code})")
        continue

    models = resp.json().get("models", [])
    flash_models = [
        m["name"] for m in models
        if "generateContent" in m.get("supportedGenerationMethods", [])
        and "flash" in m["name"].lower() or "pro" in m["name"].lower()
    ]

    print(f"✅ {version} — models that support generateContent:")
    for m in models:
        if "generateContent" in m.get("supportedGenerationMethods", []):
            print(f"   {m['name']}")
    print()
