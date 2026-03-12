import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


#Create synthetic dataset
def generate_data(n_steps=5000, seed=42):

    # Creates a synthetic time-series metrics with incident events.
    #Incidents are recognized after a pattern of rising CPU and latency

    np.random.seed(seed)
    
    #The base metrics
    cpu = np.clip(np.random.normal(0.5, 0.15, n_steps), 0, 1)
    memory = np.clip(np.random.normal(0.6, 0.1, n_steps), 0, 1)
    latency = np.random.normal(100, 15, n_steps)
    
    #Introduce precursor patterns leading to incidents
    for i in range(200, n_steps, 400):
        length = 30
        cpu[i:i+length] += np.linspace(0.1, 0.5, length)         # rising CPU
        latency[i:i+length] += np.linspace(50, 300, length)      # rising latency
    
    # Mark incidents when latency exceeds the threshold
    incident = (latency > 350).astype(int)
    
    #Combine it into DataFrame
    df = pd.DataFrame({
        "cpu": cpu,
        "memory": memory,
        "latency": latency,
        "incident": incident
    })
    return df


#Create a sliding window dataset
def create_sliding_windows(df, W=20, H=10):

    #Converts time-series into supervised learning windows.
    #W = number of steps before to use as features
    #H = number of future steps to predict incident

    X, y = [], []
    for start in range(len(df) - W - H):
        end = start + W
        window = df.iloc[start:end][["cpu", "memory", "latency"]].values.flatten()
        # Incident occurs in the next H steps?
        label = int(df.iloc[end:end+H]["incident"].max() > 0)
        X.append(window)
        y.append(label)
    return np.array(X), np.array(y)


#Main runing program 
if __name__ == "__main__":
    print("Generating dataset...")
    df = generate_data()
    
    print("Creating sliding windows...")
    X, y = create_sliding_windows(df)
    print("Dataset shape:", X.shape)
    
    # Split into train/test sets (time-based)
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    

    #Train Random Forest model
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=300,       # number of trees
        max_depth=12,           # tree depth
        class_weight="balanced",# handle class imbalance
        random_state=42
    )
    model.fit(X_train, y_train)
    

    #Evaluate the model
    print("Evaluating model...")
    y_prob = model.predict_proba(X_test)[:, 1]   # probability of incident happening
    
    threshold = 0.3                               # probability threshold for alerts to happen
    y_pred = (y_prob >= threshold).astype(int)   # binary predictions
    
    # Classification metrics
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    # ROC-AUC metric
    roc_auc = roc_auc_score(y_test, y_prob)
    print("ROC-AUC:", round(roc_auc, 3))