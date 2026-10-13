from datetime import datetime, timezone
import numpy as np
from flask import Flask, render_template, jsonify
from flask.json.provider import DefaultJSONProvider
import technical

class NpProvider(DefaultJSONProvider):
    """Convertit les types numpy (int64, float64, bool_) en types JSON natifs."""
    @staticmethod
    def default(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.bool_): return bool(o)
        return DefaultJSONProvider.default(o)

app = Flask(__name__)
app.json = NpProvider(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/analysis")
def api_analysis():
    try:
        d = technical.run()
        d["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return jsonify(d)
    except Exception as e:
        return jsonify(error=str(e)), 502

@app.route("/health")
def health():
    return "ok"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
