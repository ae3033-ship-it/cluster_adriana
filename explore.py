import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("gym_members.csv")

print(df.shape)
print(df.isnull().sum())

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")
print(df.describe())

plt.scatter(df["visits_per_month"], df["monthly_extras_spend"])
plt.xlabel("visits_per_month")
plt.ylabel("monthly_extras_spend")
plt.tight_layout()
plt.savefig("scatter.png")
plt.close()
