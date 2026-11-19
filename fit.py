import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("gym_members.csv")
X = df.drop(columns=["member_id"])

X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ks = list(range(1, 9))
train_spreads = []
test_spreads = []

for k in ks:
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    model.fit(X_train_scaled)
    labels = model.predict(X_test_scaled)
    centers = model.cluster_centers_[labels]
    test_spread = float(np.sum((X_test_scaled - centers) ** 2))
    train_spread = float(model.inertia_)
    print(k, round(train_spread, 1), round(test_spread, 1))
    train_spreads.append(train_spread)
    test_spreads.append(test_spread)

plt.plot(ks, train_spreads, marker="o", label="training spread")
plt.plot(ks, test_spreads, marker="o", label="test spread")
plt.xlabel("k")
plt.ylabel("spread")
plt.legend()
plt.tight_layout()
plt.savefig("elbow.png")
plt.close()

kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
kmeans.fit(X_train_scaled)

train_counts = np.bincount(kmeans.labels_, minlength=4)
for cluster, count in enumerate(train_counts):
    print(cluster, int(count))

centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=X.columns,
).round(1)
print(centers.to_string())

all_labels = kmeans.predict(scaler.transform(X))
assigned = df.copy()
assigned["cluster"] = all_labels
summary = assigned.groupby("cluster").agg(
    member_count=("member_id", "count"),
    avg_monthly_extras_spend=("monthly_extras_spend", "mean"),
)
print(summary.to_string())

segments = {
    0: ("Casual Newcomers", True),
    1: ("Loyal Veterans", True),
    2: ("Dedicated Gym Regulars", False),
    3: ("Class Enthusiasts", False),
}


def evaluate_member(
    age,
    visits_per_month,
    avg_visit_minutes,
    classes_per_month,
    months_member,
    monthly_extras_spend,
):
    row = pd.DataFrame(
        [
            [
                age,
                visits_per_month,
                avg_visit_minutes,
                classes_per_month,
                months_member,
                monthly_extras_spend,
            ]
        ],
        columns=X.columns,
    )
    scaled = scaler.transform(row)
    cluster = int(kmeans.predict(scaled)[0])
    name, gets_credit = segments[cluster]
    return cluster, name, gets_credit


print(
    evaluate_member(
        age=30,
        visits_per_month=4,
        avg_visit_minutes=42,
        classes_per_month=1,
        months_member=5,
        monthly_extras_spend=10,
    )
)


def member_net(monthly_extras_spend, receives_credit):
    if not receives_credit:
        return 0
    if monthly_extras_spend >= 20:
        return -10
    return 5


baseline_nets = [
    member_net(spend, True) for spend in assigned["monthly_extras_spend"]
]
v1_nets = [
    member_net(spend, segments[cluster][1])
    for spend, cluster in zip(assigned["monthly_extras_spend"], assigned["cluster"])
]
baseline_total = sum(baseline_nets)
v1_total = sum(v1_nets)
print(baseline_total)
print(v1_total)
print(v1_total - baseline_total)
