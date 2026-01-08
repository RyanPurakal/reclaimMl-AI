def label_risk(df):
    df = df.copy()

    df["recovery_risk"] = (
        (df["acwr"] > 1.1) |      
        (df["sleep_avg_7d"] < 6.8) |
        (df["days_since_rest"] > 5)
    ).astype(int)

    return df
