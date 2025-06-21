import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import re

# Load team stats and model
df = pd.read_csv("team_data.csv")
model = joblib.load("model.pkl")

# Load match data
matches_df = pd.read_csv("isl_matches.csv", encoding="ISO-8859-1")
matches_df["Date"] = pd.to_datetime(matches_df["Date"], errors='coerce')

# Normalize team names
matches_df["Home"] = matches_df["Home"].str.replace(" FC", "", regex=False).str.strip()
matches_df["Away"] = matches_df["Away"].str.replace(" FC", "", regex=False).str.strip()

# Fix score parsing
def parse_score(score):
    try:
        score = str(score)
        match = re.search(r'(\d+)\D+(\d+)', score)
        if match:
            return int(match.group(1)), int(match.group(2))
    except:
        pass
    return None, None

matches_df[["HomeGoals", "AwayGoals"]] = matches_df["Score"].apply(lambda x: pd.Series(parse_score(x)))

# Get recent form
def get_team_form(team_name):
    form = []
    df_sorted = matches_df.sort_values("Date", ascending=False)
    for _, row in df_sorted.iterrows():
        if pd.isna(row["HomeGoals"]) or pd.isna(row["AwayGoals"]):
            continue
        if row["Home"] == team_name:
            res = "W" if row["HomeGoals"] > row["AwayGoals"] else "L" if row["HomeGoals"] < row["AwayGoals"] else "D"
            form.append((res, row["HomeGoals"], row["AwayGoals"]))
        elif row["Away"] == team_name:
            res = "W" if row["AwayGoals"] > row["HomeGoals"] else "L" if row["AwayGoals"] < row["HomeGoals"] else "D"
            form.append((res, row["AwayGoals"], row["HomeGoals"]))
        if len(form) == 5:
            break
    return form

# Render icons
def render_form(form_list):
    colors = {"W": "green", "D": "orange", "L": "red"}
    html = ""
    for result, gf, ga in form_list:
        color = colors.get(result, "black")
        html += f'<span style="color:{color}; font-weight:bold; font-size:20px; margin-right:10px;">{result}</span>'
    return html


# Mini form chart
def plot_form_chart(form, team_name):
    match_numbers = list(range(len(form), 0, -1))  # Most recent = M1
    gf = [g for _, g, _ in reversed(form)]
    ga = [a for _, _, a in reversed(form)]

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot(match_numbers, gf, marker='o', label='Goals Scored', color='blue')
    ax.plot(match_numbers, ga, marker='o', label='Goals Conceded', color='red')

    ax.set_xticks(match_numbers)
    ax.set_xticklabels([f'M{i}' for i in match_numbers])
    ax.set_title(f"{team_name} Form", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.3)
    return fig

# 📊 Grouped bar comparison chart
def plot_comparison_bar(home_stats, away_stats, labels, home_name, away_name):
    home_values = [home_stats[col] for col in labels]
    away_values = [away_stats[col] for col in labels]

    home_values[3] = 100 - home_values[3]
    away_values[3] = 100 - away_values[3]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width/2, home_values, width, label=home_name, color='blue')
    bars2 = ax.bar(x + width/2, away_values, width, label=away_name, color='red')

    ax.set_ylabel('Value')
    ax.set_title('Team Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(["Goals/90", "G+A/90", "Possession", "Discipline"])
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    return fig

# Streamlit UI
st.title("⚽ ISL Football Score Predictor")

teams = df["Squad"].tolist()
home_team = st.selectbox("Select Home Team", teams)
away_team = st.selectbox("Select Away Team", teams)

if home_team == away_team:
    st.warning("Home and Away teams must be different.")
    st.stop()

# Logos and headers
col1, col2 = st.columns(2)
with col1:
    logo_path = f"assets/logos/{home_team}.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=150)
    st.markdown(f"### {home_team}")
with col2:
    logo_path = f"assets/logos/{away_team}.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=150)
    st.markdown(f"### {away_team}")

# 📊 Recent Form
st.markdown("### 📊 Recent Form (Last 5 Matches)")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"**{home_team}**")
    form = get_team_form(home_team)
    if form:
        st.markdown(render_form(form), unsafe_allow_html=True)
        st.markdown(f"GF: {sum(g for _, g, _ in form)} | GA: {sum(a for _, _, a in form)}")
        st.pyplot(plot_form_chart(form, home_team))
    else:
        st.markdown("_No recent data available_")

with col2:
    st.markdown(f"**{away_team}**")
    form = get_team_form(away_team)
    if form:
        st.markdown(render_form(form), unsafe_allow_html=True)
        st.markdown(f"GF: {sum(g for _, g, _ in form)} | GA: {sum(a for _, _, a in form)}")
        st.pyplot(plot_form_chart(form, away_team))
    else:
        st.markdown("_No recent data available_")

# 📈 Stat Comparison Bar Chart
home_stats = df[df["Squad"] == home_team].iloc[0]
away_stats = df[df["Squad"] == away_team].iloc[0]
bar_labels = ['Gls.1', 'G+A.1', 'Poss', 'CrdY']
fig = plot_comparison_bar(home_stats, away_stats, bar_labels, home_team, away_team)
st.pyplot(fig)

# 🔮 Score prediction
features = np.array([[
    home_stats['Gls.1'], home_stats['G+A.1'], home_stats['Poss'], home_stats['CrdY'],
    away_stats['Gls.1'], away_stats['G+A.1'], away_stats['Poss'], away_stats['CrdY']
]])

if st.button("Predict Match Result"):
    prediction = model.predict(features)
    home_score, away_score = prediction[0].round().astype(int)

    # Determine winner
    if home_score > away_score:
        winner = home_team
        color = "#6c63ff"
        message = f"🏆 <b>{winner}</b> Wins!"
    elif away_score > home_score:
        winner = away_team
        color = "#f45b69"
        message = f"🏆 <b>{winner}</b> Wins!"
    else:
        winner = "Draw"
        color = "#888888"
        message = "🤝 It's a Draw!"

    result_html = f"""
    <div style="background-color:#f9f9fc; padding:30px; border-radius:15px; box-shadow:0 4px 12px rgba(0,0,0,0.1);">
        <h3 style="text-align:center; margin-bottom:20px;">🔮 <span style="color:#222;">Predicted Match Result</span></h3>
        <div style="display:flex; align-items:center; justify-content:center; gap:40px; font-size:24px; font-weight:bold;">
            <span style="color:#6c63ff;">{home_team}</span>
            <span style="color:#333;">{home_score} - {away_score}</span>
            <span style="color:#f45b69;">{away_team}</span>
        </div>
        <div style="text-align:center; margin-top:20px; font-size:20px; font-weight:600; color:{color};">
            {message}
        </div>
    </div>
    """
    st.markdown(result_html, unsafe_allow_html=True)
