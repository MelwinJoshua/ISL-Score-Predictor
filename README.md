⚽ ISL Football Score Predictor

A machine learning web app to predict scores of Indian Super League matches using team statistics and recent form.

📦 Features

- Select Home & Away teams
- View recent form (last 5 matches)
- Visualize key stats (goals, possession, discipline)
- Predict match result using a trained Random Forest model

🧠 Model

- **Algorithm**: Random Forest with MultiOutputRegressor
- **Features Used**:
  - Goals per 90
  - Goals + Assists per 90
  - Possession %
  - Yellow Cards (discipline)
- **Target**: Home and Away scores

🛠️ Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit app UI |
| `train_model.py` | Model training script |
| `model.pkl` | Saved trained model |
| `team_data.csv` | Team performance stats |
| `isl_matches.csv` | Match results data |
| `isl table.csv` | (Optional) League table data |
| `requirements.txt` | Python dependencies |

🚀 Run Locally

```bash
git clone https://github.com/your-username/ISL-Score-Predictor.git
cd ISL-Score-Predictor
pip install -r requirements.txt
streamlit run app.py
