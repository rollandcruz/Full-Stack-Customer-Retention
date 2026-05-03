# Customer Retention AI Agent

A lightweight full-stack customer retention dashboard that predicts churn risk and recommends next-best save actions for a customer profile.

The app trains a simple logistic regression model at startup from `customer_churn_dataset-training-master.csv`, serves a browser UI from `static/`, and exposes JSON API endpoints for health checks, dataset metadata, and churn predictions.

## Features

- Interactive customer profile form for churn analysis
- At-risk and stable sample profiles for quick demos
- Churn probability with risk bands: `Healthy`, `Watch`, `High`, and `Critical`
- Top risk drivers with directional impact
- Recommended retention actions based on customer behavior
- Model summary metrics including sampled holdout accuracy, training rows, baseline churn rate, and sample size
- No frontend build step required

## Project Structure

```text
.
|-- app.py
|-- customer_churn_dataset-training-master.csv
|-- README.md
`-- static/
    |-- app.js
    |-- index.html
    `-- styles.css
```

## Requirements

- Python 3.10+
- NumPy

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install numpy
```

## Run the App

Start the local server:

```powershell
python app.py
```

Open the app in your browser:

```text
http://127.0.0.1:8000
```

To use a different port:

```powershell
$env:PORT = "8080"
python app.py
```

## API Endpoints

### Health Check

```http
GET /api/health
```

Returns server status and model summary information.

### Dataset Summary

```http
GET /api/summary
```

Returns model metadata, supported input fields, categorical values, and example customer payloads.

### Predict Churn

```http
POST /api/predict
Content-Type: application/json
```

Example request:

```json
{
  "Age": 52,
  "Gender": "Female",
  "Tenure": 8,
  "Usage Frequency": 4,
  "Support Calls": 8,
  "Payment Delay": 21,
  "Subscription Type": "Basic",
  "Contract Length": "Monthly",
  "Total Spend": 420,
  "Last Interaction": 24
}
```

Example response shape:

```json
{
  "churn_probability": 0.9945,
  "risk_band": "Critical",
  "drivers": [
    {
      "feature": "Support Calls",
      "impact": 1.774,
      "direction": "raises risk"
    }
  ],
  "actions": [
    "Route to senior support and close the top unresolved issue within 24 hours."
  ],
  "model": {
    "training_rows": 440832,
    "sample_rows": 30000,
    "baseline_churn_rate": 0.5671,
    "holdout_accuracy": 0.878,
    "trained_at": 1777787930.4044836
  }
}
```

The exact prediction values and model metrics can vary if the dataset or training logic changes.

## Input Fields

| Field | Type | Notes |
| --- | --- | --- |
| `Age` | Number | Customer age |
| `Gender` | Category | `Female` or `Male` |
| `Tenure` | Number | Customer tenure |
| `Usage Frequency` | Number | Recent usage frequency |
| `Support Calls` | Number | Number of support calls |
| `Payment Delay` | Number | Days of payment delay |
| `Subscription Type` | Category | `Basic`, `Standard`, or `Premium` |
| `Contract Length` | Category | `Monthly`, `Quarterly`, or `Annual` |
| `Total Spend` | Number | Customer spend |
| `Last Interaction` | Number | Days since last interaction |

## How It Works

1. `app.py` loads the churn CSV at startup.
2. The training routine samples up to 30,000 balanced rows across churned and retained customers.
3. Numeric and encoded categorical features are standardized.
4. A logistic regression model is trained with gradient descent.
5. Predictions return churn probability, a risk band, top model drivers, and retention actions.
6. The static frontend calls `/api/predict` and renders the result.

## Notes

- The model is trained in memory each time the server starts.
- The CSV file is required at the project root and is not downloaded automatically.
- This project is intended as a demo or learning app, not as a production churn model.
