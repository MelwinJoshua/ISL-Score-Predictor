
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib

# Load data
df = pd.read_csv("team_data.csv")

# Features to use
features_to_use = ["Gls.1", "G+A.1", "Poss", "CrdY"]
teams = df["Squad"].tolist()
data = []
targets = []

# Generate synthetic match data
for i in range(len(teams)):
    for j in range(len(teams)):
        if i == j:
            continue
        home = df.iloc[i]
        away = df.iloc[j]

        features = [
            home["Gls.1"], home["G+A.1"], home["Poss"], home["CrdY"],
            away["Gls.1"], away["G+A.1"], away["Poss"], away["CrdY"]
        ]

        # Synthetic target (example logic)
        home_score = (home["Gls.1"] + home["G+A.1"]) - (away["CrdY"] * 0.05)
        away_score = (away["Gls.1"] + away["G+A.1"]) - (home["CrdY"] * 0.05)
        targets.append([max(0, round(home_score)), max(0, round(away_score))])
        data.append(features)

X = np.array(data)
y = np.array(targets)

# Train model
model = MultiOutputRegressor(RandomForestRegressor())
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
