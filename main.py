# 6G tarmog'ida Random Forest asosida optimal bazaviy stansiyani tanlash modeli
# Tuzatilgan variant: AI tanlashda bazaviy stansiyalar yuklamasi dinamik hisobga olinadi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(7)

# 1. Tarmoq parametrlarini belgilash
n_users = 400
n_bts = 5
area_size = 1000

# Bazaviy stansiyalar koordinatalari
bts_positions = np.array([
    [150, 200],
    [850, 180],
    [500, 500],
    [220, 820],
    [800, 780]
])

# Foydalanuvchilar koordinatalari
user_positions = np.random.uniform(0, area_size, size=(n_users, 2))

# Bazaviy stansiyalarning boshlang'ich yuklama darajalari
base_loads = np.array([0.45, 0.48, 0.35, 0.42, 0.40])

rows = []

# 2. Dataset yaratish
for user_id, user_pos in enumerate(user_positions):
    user_traffic = np.random.uniform(1, 10)  # Mbps
    
    candidates = []
    
    for bts_id, bts_pos in enumerate(bts_positions):
        distance = np.linalg.norm(user_pos - bts_pos) + 1
        
        # Signal kuchi: masofa oshgani sari signal kamayadi
        signal_strength = -30 - 20 * np.log10(distance / 10) + np.random.normal(0, 2)
        
        # SINR: signal sifati, masofa va yuklama bilan bog'liq
        sinr = 38 - 0.022 * distance - 10 * base_loads[bts_id] + np.random.normal(0, 2)
        sinr = np.clip(sinr, 2, 38)
        
        # Kechikish: masofa, yuklama va trafik oshsa kechikish ortadi
        latency = 1.5 + distance / 180 + base_loads[bts_id] * 8 + user_traffic * 0.08 + np.random.normal(0, 0.5)
        latency = max(latency, 0.8)
        
        load = base_loads[bts_id]
        
        candidates.append({
            "user_id": user_id,
            "bts_id": bts_id,
            "distance": distance,
            "signal_strength": signal_strength,
            "sinr": sinr,
            "load": load,
            "latency": latency,
            "traffic": user_traffic
        })
    
    df_c = pd.DataFrame(candidates)
    
    # Har bir foydalanuvchi uchun lokal normalizatsiya
    for col in ["distance", "signal_strength", "sinr", "load", "latency", "traffic"]:
        df_c[col + "_n"] = (df_c[col] - df_c[col].min()) / (df_c[col].max() - df_c[col].min() + 1e-9)
    
    # Optimal stansiyani belgilash uchun nazariy baholash mezoni
    # SINR va signal yuqori bo'lsa yaxshi, yuklama, masofa va kechikish past bo'lsa yaxshi
    df_c["score"] = (
        0.35 * df_c["sinr_n"] +
        0.20 * df_c["signal_strength_n"] -
        0.20 * df_c["load_n"] -
        0.15 * df_c["distance_n"] -
        0.10 * df_c["latency_n"] -
        0.02 * df_c["traffic_n"]
    )
    
    best_index = df_c["score"].idxmax()
    df_c["is_optimal"] = 0
    df_c.loc[best_index, "is_optimal"] = 1
    
    df_c = df_c.drop(columns=[
        "distance_n", "signal_strength_n", "sinr_n",
        "load_n", "latency_n", "traffic_n"
    ])
    
    rows.extend(df_c.to_dict("records"))

dataset = pd.DataFrame(rows)

print("Dataset hajmi:", dataset.shape)
display(dataset.head(10))

# 3. Random Forest modelini o'qitish
features = ["distance", "signal_strength", "sinr", "load", "latency", "traffic"]
X = dataset[features]
y = dataset["is_optimal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=7,
    stratify=y
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=7,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nRandom Forest model aniqligi:", round(accuracy * 100, 2), "%")
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# 4. An'anaviy va AI usullar bo'yicha BTS tanlash

def choose_by_closest(group):
    return group.loc[group["distance"].idxmin()]
  def choose_by_signal(group):
    return group.loc[group["signal_strength"].idxmax()]

def normalize_group(group, col):
    return (group[col] - group[col].min()) / (group[col].max() - group[col].min() + 1e-9)

# Eng yaqin BTS usuli
closest_rows = []
for user_id in sorted(dataset["user_id"].unique()):
    group = dataset[dataset["user_id"] == user_id]
    closest_rows.append(choose_by_closest(group))
closest_choices = pd.DataFrame(closest_rows)

# Eng kuchli signal usuli
signal_rows = []
for user_id in sorted(dataset["user_id"].unique()):
    group = dataset[dataset["user_id"] == user_id]
    signal_rows.append(choose_by_signal(group))
signal_choices = pd.DataFrame(signal_rows)

# Random Forest AI usuli: dinamik yuklama hisobga olinadi
ai_rows = []
dynamic_counts = np.zeros(n_bts)

for user_id in sorted(dataset["user_id"].unique()):
    group = dataset[dataset["user_id"] == user_id].copy()
    
    rf_probability = rf_model.predict_proba(group[features])[:, 1]
    
    sinr_n = normalize_group(group, "sinr")
    signal_n = normalize_group(group, "signal_strength")
    distance_n = normalize_group(group, "distance")
    latency_n = normalize_group(group, "latency")
    
    current_dynamic_load = dynamic_counts[group["bts_id"].values] / max(1, len(ai_rows) + 1)
    
    if current_dynamic_load.max() != current_dynamic_load.min():
        dynamic_load_n = (current_dynamic_load - current_dynamic_load.min()) / (current_dynamic_load.max() - current_dynamic_load.min() + 1e-9)
    else:
        dynamic_load_n = np.zeros_like(current_dynamic_load)
    
    # AI yakuniy tanlovi:
    # RF ehtimoli + SINR + signal, lekin dinamik yuklama, masofa va kechikish jazolanadi
    group["ai_score"] = (
        0.45 * rf_probability +
        0.25 * sinr_n +
        0.10 * signal_n -
        0.10 * distance_n -
        0.10 * latency_n -
        0.40 * dynamic_load_n
    )
    
    best_row = group.loc[group["ai_score"].idxmax()]
    ai_rows.append(best_row)
    dynamic_counts[int(best_row["bts_id"])] += 1

ai_choices = pd.DataFrame(ai_rows)

# 5. Usullarni baholash

def evaluate_method(df, method_name):
    bts_counts = df["bts_id"].value_counts().reindex(range(n_bts), fill_value=0)
    
    # Yuklama notekisligi qancha kichik bo'lsa, balans shuncha yaxshi
    load_unevenness = bts_counts.std() / bts_counts.mean()
    
    avg_sinr = df["sinr"].mean()
    avg_latency = df["latency"].mean()
    avg_load = df["load"].mean()
    
    # Umumiy samaradorlik:
    # SINR yuqori bo'lsa yaxshi, kechikish, yuklama va notekislik past bo'lsa yaxshi
    efficiency = (
        0.35 * (avg_sinr / 38) +
        0.25 * (1 - avg_latency / 20) +
        0.25 * (1 - load_unevenness) +
        0.15 * (1 - avg_load)
    ) * 100
    
    return {
        "Usul": method_name,
        "O'rtacha SINR (dB)": avg_sinr,
        "O'rtacha kechikish (ms)": avg_latency,
        "O'rtacha BTS yuklamasi": avg_load,
        "Yuklama notekisligi": load_unevenness,
        "Umumiy samaradorlik (%)": efficiency
    }

results = pd.DataFrame([
    evaluate_method(closest_choices, "Eng yaqin BTS"),
    evaluate_method(signal_choices, "Eng kuchli signal"),
    evaluate_method(ai_choices, "Random Forest AI")
])

results_rounded = results.copy()
for col in results_rounded.columns[1:]:
    results_rounded[col] = results_rounded[col].round(2)

print("\nNatijalar jadvali:")
display(results_rounded)

# 6. Foizli yaxshilanishlarni hisoblash

base_closest = results[results["Usul"] == "Eng yaqin BTS"].iloc[0]
base_signal = results[results["Usul"] == "Eng kuchli signal"].iloc[0]
ai = results[results["Usul"] == "Random Forest AI"].iloc[0]
improvements = pd.DataFrame([
    {
        "Taqqoslash": "AI vs Eng yaqin BTS",
        "SINR oshishi (%)": (ai["O'rtacha SINR (dB)"] - base_closest["O'rtacha SINR (dB)"]) / base_closest["O'rtacha SINR (dB)"] * 100,
        "Kechikish kamayishi (%)": (base_closest["O'rtacha kechikish (ms)"] - ai["O'rtacha kechikish (ms)"]) / base_closest["O'rtacha kechikish (ms)"] * 100,
        "Yuklama notekisligi kamayishi (%)": (base_closest["Yuklama notekisligi"] - ai["Yuklama notekisligi"]) / base_closest["Yuklama notekisligi"] * 100,
        "Samaradorlik oshishi (%)": (ai["Umumiy samaradorlik (%)"] - base_closest["Umumiy samaradorlik (%)"]) / base_closest["Umumiy samaradorlik (%)"] * 100
    },
    {
        "Taqqoslash": "AI vs Eng kuchli signal",
        "SINR oshishi (%)": (ai["O'rtacha SINR (dB)"] - base_signal["O'rtacha SINR (dB)"]) / base_signal["O'rtacha SINR (dB)"] * 100,
        "Kechikish kamayishi (%)": (base_signal["O'rtacha kechikish (ms)"] - ai["O'rtacha kechikish (ms)"]) / base_signal["O'rtacha kechikish (ms)"] * 100,
        "Yuklama notekisligi kamayishi (%)": (base_signal["Yuklama notekisligi"] - ai["Yuklama notekisligi"]) / base_signal["Yuklama notekisligi"] * 100,
        "Samaradorlik oshishi (%)": (ai["Umumiy samaradorlik (%)"] - base_signal["Umumiy samaradorlik (%)"]) / base_signal["Umumiy samaradorlik (%)"] * 100
    }
])

for col in improvements.columns[1:]:
    improvements[col] = improvements[col].round(2)

print("\nFoizli yaxshilanishlar:")
display(improvements)

# 7. Grafiklar yaratish va saqlash

plt.figure(figsize=(7, 4))
plt.bar(results["Usul"], results["O'rtacha SINR (dB)"])
plt.title("SINR ko'rsatkichlari taqqoslanishi")
plt.ylabel("O'rtacha SINR (dB)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("sinr_taqqoslash.png", dpi=300)
plt.show()

plt.figure(figsize=(7, 4))
plt.bar(results["Usul"], results["O'rtacha kechikish (ms)"])
plt.title("Kechikish qiymatlari taqqoslanishi")
plt.ylabel("O'rtacha kechikish (ms)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("kechikish_taqqoslash.png", dpi=300)
plt.show()

load_df = pd.DataFrame({
    "Eng yaqin BTS": closest_choices["bts_id"].value_counts().reindex(range(n_bts), fill_value=0),
    "Eng kuchli signal": signal_choices["bts_id"].value_counts().reindex(range(n_bts), fill_value=0),
    "Random Forest AI": ai_choices["bts_id"].value_counts().reindex(range(n_bts), fill_value=0)
})

plt.figure(figsize=(8, 4))
load_df.plot(kind="bar")
plt.title("Bazaviy stansiyalar bo'yicha foydalanuvchilar taqsimoti")
plt.xlabel("Bazaviy stansiya ID")
plt.ylabel("Ulangan foydalanuvchilar soni")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("bts_yuklama_taqsimoti.png", dpi=300)
plt.show()

plt.figure(figsize=(7, 4))
plt.bar(results["Usul"], results["Umumiy samaradorlik (%)"])
plt.title("Umumiy samaradorlik ko'rsatkichlari")
plt.ylabel("Samaradorlik (%)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("umumiy_samaradorlik.png", dpi=300)
plt.show()

# 8. Feature importance grafigi
importance = pd.DataFrame({
    "Parametr": features,
    "Muhimlik": rf_model.feature_importances_
}).sort_values(by="Muhimlik", ascending=False)

print("\nModel parametrlarining muhimlik darajasi:")
display(importance)

plt.figure(figsize=(7, 4))
plt.bar(importance["Parametr"], importance["Muhimlik"])
plt.title("Random Forest modelida parametrlar muhimligi")
plt.ylabel("Muhimlik darajasi")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("parametrlar_muhimligi.png", dpi=300)
plt.show()

# 9. Fayllarni saqlash
dataset.to_csv("dataset_6g_random_forest.csv", index=False)
results_rounded.to_csv("natijalar_jadvali.csv", index=False)
improvements.to_csv("foizli_yaxshilanish.csv", index=False)
importance.to_csv("parametrlar_muhimligi.csv", index=False)
print("\nTayyor fayllar:")
print("- sinr_taqqoslash.png")
print("- kechikish_taqqoslash.png")
print("- bts_yuklama_taqsimoti.png")
print("- umumiy_samaradorlik.png")
print("- parametrlar_muhimligi.png")
print("- dataset_6g_random_forest.csv")
print("- natijalar_jadvali.csv")
print("- foizli_yaxshilanish.csv")
print("- parametrlar_muhimligi.csv")
