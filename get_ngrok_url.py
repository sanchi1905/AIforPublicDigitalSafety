import sys, io, time, urllib.request, urllib.error, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pyngrok import ngrok

# Open tunnel
tunnel = ngrok.connect(8000, "http")
url = tunnel.public_url

print("=" * 60)
print("  NEW NGROK URL:", url)
print("=" * 60)
print("  /predict endpoint ->", url + "/predict")
print("  /health  endpoint ->", url + "/health")
print("=" * 60)

# Health check
time.sleep(1)
try:
    req = urllib.request.Request(url + "/health",
          headers={"ngrok-skip-browser-warning": "true"})
    with urllib.request.urlopen(req, timeout=10) as r:
        print("\n  [STATUS] SERVER IS ONLINE -", json.loads(r.read()))
except Exception as e:
    print("\n  [STATUS] SERVER IS OFFLINE -", e)

# Disconnect so script exits cleanly
ngrok.disconnect(url)
print("\nDone. Copy the URL above and share with your teammate.")
