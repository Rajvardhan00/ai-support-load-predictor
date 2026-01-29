import streamlit as st
import pandas as pd
import joblib
import math
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
st.set_page_config(
    page_title="AI Support Load Predictor",
    layout="centered"
)

# ---------- LOAD MODEL ----------
model = joblib.load("models/load_model.pkl")

# ---------- LOAD DATA ----------
df = pd.read_csv("data/support_tickets.csv", parse_dates=["created_at"])

daily = (
    df.groupby(df["created_at"].dt.date)
      .size()
      .reset_index(name="ticket_count")
)

daily.rename(columns={"created_at": "date"}, inplace=True)
daily["date"] = pd.to_datetime(daily["date"])

daily["day_of_week"] = daily["date"].dt.weekday
daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)
daily["lag_1"] = daily["ticket_count"].shift(1)
daily["lag_7"] = daily["ticket_count"].shift(7)
daily["rolling_7"] = daily["ticket_count"].rolling(7).mean()
daily = daily.dropna()

# ---------- UI ----------
st.title("📞 AI-Based Customer Support Load Predictor")
st.write("Predict ticket volume and staffing needs using ML")

tickets_per_agent = st.slider(
    "Tickets handled per agent per day",
    min_value=10,
    max_value=40,
    value=20
)

forecast_days = st.selectbox(
    "Forecast duration",
    [1, 7]
)

# ---------- FORECAST ----------
temp = daily.copy()
results = []

for i in range(forecast_days):
    latest = temp.iloc[-1]

    X_next = [[
        latest["day_of_week"],
        latest["is_weekend"],
        latest["lag_1"],
        latest["lag_7"],
        latest["rolling_7"]
    ]]

    pred = int(model.predict(X_next)[0])
    agents = math.ceil(pred / tickets_per_agent)

    results.append({
        "Day": f"Day {i+1}",
        "Predicted Tickets": pred,
        "Agents Required": agents
    })

    new_row = latest.copy()
    new_row["ticket_count"] = pred
    temp = pd.concat([temp, pd.DataFrame([new_row])], ignore_index=True)

result_df = pd.DataFrame(results)

# ---------- OUTPUT ----------
st.subheader(" Forecast Results")
st.dataframe(result_df, use_container_width=True)

# ---------- CHART ----------
st.subheader(" Ticket Volume Forecast")

fig, ax = plt.subplots()
ax.plot(result_df["Predicted Tickets"], marker="o")
ax.set_ylabel("Tickets")
ax.set_xlabel("Day")
st.pyplot(fig)

# ---------- BUSINESS INSIGHT ----------
st.subheader(" Recommendation")
st.success(
    f"Recommended staffing: **{result_df['Agents Required'].max()} agents**"
)
