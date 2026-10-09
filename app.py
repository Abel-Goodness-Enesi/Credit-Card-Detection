import json, os
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

BASE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, "artifacts")

META = json.load(open(os.path.join(ART, "meta.json")))
W = np.load(os.path.join(ART, "weights.npz"))
LAYERS = [(W[f"W{i}"], W[f"b{i}"], a) for i, a in enumerate(META["activations"])]
MEAN = np.array(META["scaler"]["mean"])
SCALE = np.array(META["scaler"]["scale"])
N_SCALER = META["scaler"]["n"]
THRESHOLD = META["threshold"]
SAMPLES = META["samples"]

app = Flask(__name__, static_folder="static")


def act(name, z):
    if name == "relu":
        return np.maximum(z, 0)
    if name == "sigmoid":
        return 1 / (1 + np.exp(-z))
    if name == "tanh":
        return np.tanh(z)
    return z


def scale(x):
    if N_SCALER == len(x):
        return (x - MEAN) / SCALE
    out = x.copy()
    out[-1] = (x[-1] - MEAN[0]) / SCALE[0]  # scaler was fit on Amount only
    return out


def forward(x):
    a = scale(x)
    trace = [a]
    for Wm, b, name in LAYERS:
        a = act(name, a @ Wm + b)
        trace.append(a)
    return float(a.ravel()[0]), trace


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/meta")
def meta():
    return jsonify(
        features=META["features"],
        sizes=META["sizes"],
        threshold=THRESHOLD,
        metrics=META["metrics"],
        n=len(SAMPLES),
        frauds=sum(1 for s in SAMPLES if s["y"] == 1),
    )


@app.get("/api/feed")
def feed():
    """Run one stored test transaction through the network."""
    try:
        s = SAMPLES[int(request.args.get("i", 0)) % len(SAMPLES)]
    except (ValueError, ZeroDivisionError):
        return jsonify(error="bad index"), 400
    x = np.array(s["x"], dtype=float)
    p, trace = forward(x)
    return jsonify(
        features=s["x"],
        label=s["y"],
        probability=p,
        flagged=bool(p >= THRESHOLD),
        layers=[[round(float(v), 4) for v in t.ravel()] for t in trace],
    )


@app.get("/health")
def health():
    return "ok"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
