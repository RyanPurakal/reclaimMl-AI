import pandas as pd
import numpy as np

def generate_data(days=60, seed=42):
    np.random.seed(seed)
    data = []
    days_since_rest = 0

    for d in range(days):
        rest_day = np.random.rand() < 0.2

        if rest_day:
            sets = reps = avg_weight = volume = 0
            days_since_rest = 0
        else:
            sets = np.random.randint(12, 20)
            reps = np.random.randint(60, 120)
            avg_weight = np.random.uniform(90, 135)
            volume = sets * reps * avg_weight
            days_since_rest += 1

        sleep_base = 7.5
        sleep = sleep_base - 0.15 * days_since_rest + np.random.normal(0, 0.3)
        sleep = np.clip(sleep, 4.5, 9)

        data.append([d, sets, reps, avg_weight, volume, sleep, int(rest_day), days_since_rest])

    df = pd.DataFrame(data, columns=[
        "day", "sets", "reps", "avg_weight", "volume_load", "sleep_hours", "rest_day", "days_since_rest"
    ])

    df["rolling_volume_7d"] = df["volume_load"].rolling(7).sum()
    df["rolling_volume_14d"] = df["volume_load"].rolling(14).sum()
    df["sleep_avg_7d"] = df["sleep_hours"].rolling(7).mean()
    df["sleep_std_7d"] = df["sleep_hours"].rolling(7).std()

    # Acute: CWR (Chronic Workload Ratio)
    df["acwr"] = df["rolling_volume_7d"] / (df["rolling_volume_14d"] / 2)

    return df