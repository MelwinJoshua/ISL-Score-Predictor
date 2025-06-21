
# ⚽ ISL Football Score Predictor

This is a simple and fun machine learning web app that predicts the scores of Indian Super League (ISL) matches. It uses team statistics and recent form to guess the outcome, and shows everything with cool visuals using Streamlit.

---

## 📌 What This Project Does

You can pick any two teams from the ISL and the app will:

- Show their recent match form (last 5 games)
- Compare stats like goals, possession, and discipline
- Predict the final score using a trained machine learning model
- Tell you who might win (or if it could be a draw)

It’s built mainly for fun and to learn about data analysis and machine learning in football.

---

## 🔍 How It Works

I trained a Random Forest regression model using team stats like:

- Goals per 90 minutes
- Combined goals + assists
- Possession percentage
- Yellow cards (discipline)

The model uses these stats for both the home and away team and predicts the number of goals each team might score.

---

## 📁 What's Inside the Project

```
ISL-Score-Predictor/
├── app.py               # Streamlit app that runs the UI
├── train_model.py       # Script to train the prediction model
├── model.pkl            # Saved trained model
├── team_data.csv        # ISL team statistics
├── isl_matches.csv      # Match history for form data
├── isl table.csv        # League table (optional)
├── requirements.txt     # Python libraries needed
├── assets/
│   └── logos/           # Team logos used in the app
└── README.md            # This file :)
```

---

## 🚀 How to Run It

1. Make sure you have Python installed (3.8 or higher is recommended).
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Launch the app:

```bash
streamlit run app.py
```

4. Your browser should open automatically at `http://localhost:8501`. If not, just paste that in.

---

## 🤔 Why I Built This

I’ve always enjoyed watching football, especially the ISL. I wanted to combine that interest with my passion for coding and machine learning. This project helped me understand how team stats can influence outcomes — and how we can try (even if imperfectly) to predict results using data.



## 📜 License

Feel free to use this project for learning or experimenting — it's open-source under the MIT License.
