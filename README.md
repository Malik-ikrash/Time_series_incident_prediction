# Time_series_incident_prediction
This project implements a machine learning model that predicts whether a system incident will occur within the next **H time steps** based on the previous **W time steps** of system metrics. The model uses a **sliding-window formulation** to transform time-series data into supervised learning examples and trains a **Random Forest classifier** to detect precursor patterns that may indicate an upcoming incident.

The goal of this project is to demonstrate correct **problem formulation, model training, evaluation, and analysis** rather than focusing on dataset complexity or large models.

---

# Problem Overview

In real-world monitoring systems, incidents (such as outages, latency spikes, or resource exhaustion) are often preceded by abnormal behavior in system metrics. Detecting these patterns early allows systems to raise alerts before the incident occurs.

This project simulates such a scenario by generating synthetic time-series metrics and training a model to predict whether an incident will occur within a future time horizon.

---

# Approach

## Sliding Window Formulation

Time-series data must be converted into supervised learning examples.

We define:

* **W** – number of past time steps used as input features
* **H** – future prediction horizon

For each time step `t`:

Input features:

```
metrics from t-W → t
```

Prediction target:

```
whether an incident occurs between t+1 → t+H
```

In this implementation:

```
W = 20
H = 10
```

This means the model looks at the **last 20 time steps** of system metrics to predict whether an incident will occur in the **next 10 time steps**.

---

# Synthetic Dataset

A synthetic dataset is generated to simulate system behavior.

Metrics include:

* CPU utilization
* Memory utilization
* Request latency

Incidents are defined when **latency exceeds a threshold**, and precursor patterns are introduced before incidents by gradually increasing CPU usage and latency. These patterns allow the model to learn signals that may indicate an upcoming failure.

This approach keeps the dataset simple while still demonstrating the modeling process.

---

# Model

The model used is a **Random Forest classifier** from `scikit-learn`.

Random Forest was chosen because:

* It works well with tabular feature data
* It handles nonlinear relationships
* It is robust to noise
* It requires minimal preprocessing

The model predicts the **probability of an incident occurring within the prediction horizon**.

An alert is raised if this probability exceeds a defined threshold.

---

# Alert Threshold

Predictions are converted into alerts using a probability threshold:

```
alert if P(incident) ≥ threshold
```

In this project:

```
threshold = 0.3
```

Lowering the threshold increases **recall** (more incidents detected) but may decrease **precision** (more false alerts). This reflects the typical trade-off in real monitoring systems.

---

# Evaluation Metrics

The model is evaluated using several standard classification metrics:

### Precision

Measures how many alerts were actually correct.

```
Precision = True Positives / Predicted Positives
```

### Recall

Measures how many real incidents were successfully detected.

```
Recall = True Positives / Actual Incidents
```

### F1 Score

The harmonic mean of precision and recall.

### ROC-AUC

Measures how well the model separates incident vs non-incident cases across thresholds.

---

# Example Results

Example output from the model:

```
precision    recall  f1-score   support

0       1.00      1.00      1.00      1438
1       1.00      0.92      0.96        53

ROC-AUC: 1.0
```

These results indicate that the model successfully identifies precursor patterns before incidents in the synthetic dataset.

Because the dataset contains clear signals before incidents, the model achieves near-perfect performance. In real-world systems, more noise and variability would typically lead to lower metrics unlike the results achieved here.

---

# Limitations

This implementation is intentionally very simplified.

Key limitations include:

* The dataset is synthetic and contains strong, consistent patterns
* Real-world systems have noisier metrics and less predictable failures
* Incidents may occur due to many complex factors not captured by a few metrics
* The model does not operate in real-time streaming mode

---

# How to Run

Clone the repository (Two commands):

```
git clone https://github.com/Malik-ikrash/Time-series-incident-prediction.git
cd Time-series-incident-prediction
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the model:

```
python predict_incident.py
```

The script should create a synthetic time-series data, create sliding-window training samples,
train a random forest classifie and thene evaluate the predictions using classification metrics.
