from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "customer_churn_dataset-training-master.csv"
STATIC_DIR = ROOT / "static"

NUMERIC_FIELDS = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
]
CATEGORICAL_FIELDS = ["Gender", "Subscription Type", "Contract Length"]
FEATURE_NAMES = [
    "age",
    "tenure",
    "usage_frequency",
    "support_calls",
    "payment_delay",
    "total_spend",
    "last_interaction",
    "gender_male",
    "subscription_standard",
    "subscription_premium",
    "contract_quarterly",
    "contract_annual",
]


@dataclass
class RetentionModel:
    weights: np.ndarray
    bias: float
    means: np.ndarray
    stds: np.ndarray
    training_rows: int
    sample_rows: int
    churn_rate: float
    accuracy: float
    trained_at: float


def _as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "").strip()
    if not value:
        return None
    return float(value)


def row_to_features(row: dict[str, Any]) -> np.ndarray:
    gender = str(row.get("Gender", "")).lower()
    subscription = str(row.get("Subscription Type", "")).lower()
    contract = str(row.get("Contract Length", "")).lower()

    return np.array(
        [
            float(row.get("Age", 0)),
            float(row.get("Tenure", 0)),
            float(row.get("Usage Frequency", 0)),
            float(row.get("Support Calls", 0)),
            float(row.get("Payment Delay", 0)),
            float(row.get("Total Spend", 0)),
            float(row.get("Last Interaction", 0)),
            1.0 if gender == "male" else 0.0,
            1.0 if subscription == "standard" else 0.0,
            1.0 if subscription == "premium" else 0.0,
            1.0 if contract == "quarterly" else 0.0,
            1.0 if contract == "annual" else 0.0,
        ],
        dtype=float,
    )


def load_dataset(path: Path, max_rows: int = 30_000) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    positive_features: list[np.ndarray] = []
    negative_features: list[np.ndarray] = []
    rows = 0
    missing_rows = 0
    positive_rows = 0
    negative_rows = 0
    per_class_limit = max_rows // 2

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            churn_value = row.get("Churn", "").strip()
            if not churn_value:
                missing_rows += 1
                continue
            label = int(float(churn_value))
            if label == 1:
                positive_rows += 1
            else:
                negative_rows += 1

            needs_positive = label == 1 and len(positive_features) < per_class_limit
            needs_negative = label == 0 and len(negative_features) < per_class_limit
            if needs_positive or needs_negative:
                if any(_as_float(row, field) is None for field in NUMERIC_FIELDS):
                    missing_rows += 1
                    continue
                item = row_to_features(row)
                if label == 1:
                    positive_features.append(item)
                else:
                    negative_features.append(item)

    features = positive_features + negative_features
    labels = [1] * len(positive_features) + [0] * len(negative_features)
    y = np.array(labels, dtype=float)
    stats = {
        "rows": rows,
        "usable_rows": positive_rows + negative_rows,
        "sample_rows": len(labels),
        "churn_rate": positive_rows / max(1, positive_rows + negative_rows),
    }
    return np.vstack(features), y, stats


def train_model(path: Path) -> RetentionModel:
    x, y, stats = load_dataset(path)
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    stds[stds == 0] = 1.0
    xs = (x - means) / stds

    rng = np.random.default_rng(11)
    order = rng.permutation(len(y))
    split = int(len(y) * 0.82)
    train_idx = order[:split]
    test_idx = order[split:]

    weights = np.zeros(xs.shape[1], dtype=float)
    bias = 0.0
    learning_rate = 0.12
    l2 = 0.001

    for _ in range(90):
        z = xs[train_idx] @ weights + bias
        preds = 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
        error = preds - y[train_idx]
        weights -= learning_rate * ((xs[train_idx].T @ error) / len(train_idx) + l2 * weights)
        bias -= learning_rate * float(error.mean())

    test_preds = 1.0 / (1.0 + np.exp(-np.clip(xs[test_idx] @ weights + bias, -35, 35)))
    accuracy = float(((test_preds >= 0.5) == y[test_idx]).mean())

    return RetentionModel(
        weights=weights,
        bias=float(bias),
        means=means,
        stds=stds,
        training_rows=int(stats["usable_rows"]),
        sample_rows=int(stats["sample_rows"]),
        churn_rate=float(stats["churn_rate"]),
        accuracy=accuracy,
        trained_at=time.time(),
    )


MODEL = train_model(DATA_PATH)


def predict_churn(payload: dict[str, Any]) -> dict[str, Any]:
    raw_features = row_to_features(payload)
    scaled = (raw_features - MODEL.means) / MODEL.stds
    logit = float(scaled @ MODEL.weights + MODEL.bias)
    probability = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, logit))))
    drivers = explain_drivers(scaled, raw_features)
    actions = recommend_actions(payload, probability, drivers)

    return {
        "churn_probability": round(probability, 4),
        "risk_band": risk_band(probability),
        "drivers": drivers,
        "actions": actions,
        "model": model_summary(),
    }


def explain_drivers(scaled_features: np.ndarray, raw_features: np.ndarray) -> list[dict[str, Any]]:
    contributions = scaled_features * MODEL.weights
    ranked = []
    for index, name in enumerate(FEATURE_NAMES):
        is_inactive_category = index >= len(NUMERIC_FIELDS) and raw_features[index] == 0
        if not is_inactive_category:
            ranked.append((name, contributions[index]))
    ranked = sorted(ranked, key=lambda item: abs(float(item[1])), reverse=True)
    return [
        {
            "feature": prettify_feature(name),
            "impact": round(float(value), 3),
            "direction": "raises risk" if value > 0 else "lowers risk",
        }
        for name, value in ranked[:5]
    ]


def prettify_feature(name: str) -> str:
    return name.replace("_", " ").title()


def risk_band(probability: float) -> str:
    if probability >= 0.72:
        return "Critical"
    if probability >= 0.5:
        return "High"
    if probability >= 0.28:
        return "Watch"
    return "Healthy"


def recommend_actions(payload: dict[str, Any], probability: float, drivers: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    support_calls = float(payload.get("Support Calls", 0))
    payment_delay = float(payload.get("Payment Delay", 0))
    usage = float(payload.get("Usage Frequency", 0))
    last_interaction = float(payload.get("Last Interaction", 0))
    contract = str(payload.get("Contract Length", "")).lower()
    spend = float(payload.get("Total Spend", 0))

    if support_calls >= 6:
        actions.append("Route to senior support and close the top unresolved issue within 24 hours.")
    if payment_delay >= 14:
        actions.append("Offer a payment-plan nudge or billing reminder before the next invoice date.")
    if usage <= 8:
        actions.append("Trigger an activation playbook with three use-case tips tied to their subscription.")
    if last_interaction >= 18:
        actions.append("Schedule a proactive check-in from customer success this week.")
    if contract == "monthly" and probability >= 0.5:
        actions.append("Offer an annual-plan incentive after the support or billing issue is resolved.")
    if spend >= 750 and probability >= 0.5:
        actions.append("Mark as high-value save opportunity and assign an owner.")

    if not actions:
        top = drivers[0]["feature"].lower() if drivers else "recent behavior"
        actions.append(f"Keep in nurture and monitor {top} for movement over the next 14 days.")
    return actions[:4]


def model_summary() -> dict[str, Any]:
    return {
        "training_rows": MODEL.training_rows,
        "sample_rows": MODEL.sample_rows,
        "baseline_churn_rate": round(MODEL.churn_rate, 4),
        "holdout_accuracy": round(MODEL.accuracy, 4),
        "trained_at": MODEL.trained_at,
    }


def dataset_summary() -> dict[str, Any]:
    return {
        "model": model_summary(),
        "fields": {
            "numeric": NUMERIC_FIELDS,
            "categorical": {
                "Gender": ["Female", "Male"],
                "Subscription Type": ["Basic", "Standard", "Premium"],
                "Contract Length": ["Monthly", "Quarterly", "Annual"],
            },
        },
        "examples": [
            {
                "name": "At-risk monthly subscriber",
                "payload": {
                    "Age": 52,
                    "Gender": "Female",
                    "Tenure": 8,
                    "Usage Frequency": 4,
                    "Support Calls": 8,
                    "Payment Delay": 21,
                    "Subscription Type": "Basic",
                    "Contract Length": "Monthly",
                    "Total Spend": 420,
                    "Last Interaction": 24,
                },
            },
            {
                "name": "Stable premium subscriber",
                "payload": {
                    "Age": 34,
                    "Gender": "Male",
                    "Tenure": 44,
                    "Usage Frequency": 25,
                    "Support Calls": 1,
                    "Payment Delay": 2,
                    "Subscription Type": "Premium",
                    "Contract Length": "Annual",
                    "Total Spend": 880,
                    "Last Interaction": 5,
                },
            },
        ],
    }


class RetentionHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_json({"ok": True, "model": model_summary()})
            return
        if parsed.path == "/api/summary":
            self.send_json(dataset_summary())
            return
        if parsed.path == "/api/predict":
            payload = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
            self.send_json(predict_churn(payload))
            return
        self.serve_static(parsed.path)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(body or "{}")
            self.send_json(predict_churn(payload))
        except (ValueError, KeyError) as exc:
            self.send_json({"error": str(exc)}, status=400)

    def serve_static(self, request_path: str) -> None:
        target = STATIC_DIR / ("index.html" if request_path in {"", "/"} else request_path.lstrip("/"))
        target = target.resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.exists():
            self.send_error(404)
            return

        content_types = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
        }
        content = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_types.get(target.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        content = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), RetentionHandler)
    print(f"Retention AI Agent running at http://127.0.0.1:{port}")
    print(f"Model accuracy: {MODEL.accuracy:.3f} on sampled holdout rows")
    server.serve_forever()


if __name__ == "__main__":
    main()
