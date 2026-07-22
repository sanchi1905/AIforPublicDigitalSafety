"""
start_ngrok.py
==============
Starts a pyngrok tunnel on port 8000 and checks if the FastAPI server
is online by hitting the /health endpoint.
"""

import sys
import io
import time
import urllib.request
import urllib.error
import json

# Force UTF-8 output so Windows cp1252 terminal doesn't crash
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pyngrok import ngrok, conf

# ── 1. Open an ngrok tunnel on port 8000 ─────────────────────────────────────
print("Starting ngrok tunnel on port 8000...")
tunnel = ngrok.connect(8000, "http")
public_url = tunnel.public_url

print("\n" + "="*60)
print(f"  [*]  NEW NGROK URL  ->  {public_url}")
print("="*60)
print(f"\n  Share this URL with your teammate:")
print(f"  {public_url}/predict   (POST - upload image)")
print(f"  {public_url}/health    (GET  - health check)")
print(f"  {public_url}/          (GET  - API info)")
print("="*60 + "\n")

# ── 2. Wait a moment for the server to be ready ───────────────────────────────
time.sleep(2)

# ── 3. Health check via the ngrok URL ─────────────────────────────────────────
health_url = f"{public_url}/health"
print(f"Checking terminal status via: {health_url}")

try:
    req = urllib.request.Request(health_url, headers={"ngrok-skip-browser-warning": "true"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())
        print(f"\n  [OK]  SERVER IS ONLINE")
        print(f"  Response: {data}")
except urllib.error.URLError as e:
    print(f"\n  [!!]  SERVER IS OFFLINE or unreachable")
    print(f"  Error: {e.reason}")
except Exception as e:
    print(f"\n  [!!]  Unexpected error: {e}")

print("\nPress Ctrl+C to stop the ngrok tunnel.\n")

# ── 4. Keep tunnel alive ───────────────────────────────────────────────────────
try:
    ngrok.get_ngrok_process().proc.wait()
except KeyboardInterrupt:
    print("\nStopping ngrok tunnel...")
    ngrok.kill()
