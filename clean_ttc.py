import sys
import pandas as pd

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "ttc-bus-delay-data-2025.csv"
DELAY_THRESHOLD = 10  # minutes; "delayed" = Min Delay >= this
OUT_PATH = "ttc_clean.csv"


def load(path):
    df = pd.read_csv(path)
    # normalize column names (headers vary between years: "Min Delay", " Min Delay", "Delay", etc.)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"min_delay": "delay", "min_gap": "gap"})
    # newer files use different headers; map common variants to what the script expects
    aliases = {
        "route": ["route", "line", "route_number", "bus_route"],
        "time": ["time", "report_time"],
        "date": ["date", "report_date"],
        "delay": ["delay", "min_delay", "delay_min"],
        "direction": ["direction", "bound"],
    }
    for target, options in aliases.items():
        if target in df.columns:
            continue
        for opt in options:
            if opt in df.columns:
                df = df.rename(columns={opt: target})
                break
    print("Columns after mapping:", list(df.columns))
    missing = [c for c in ["date", "time", "route", "delay"] if c not in df.columns]
    if missing:
        sys.exit(f"Missing columns {missing}. Paste me this header row: {list(df.columns)}")
    return df


def clean(df):
    print("Raw shape:", df.shape)
    print("Columns:", list(df.columns))

    # date + time -> datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    t = pd.to_datetime(df["time"].astype(str), format="mixed", errors="coerce")
    df["hour"] = t.dt.hour
    df["minute"] = t.dt.minute
    df = df.dropna(subset=["date", "hour", "delay", "route"])

    # route: keep numeric-looking bus routes only (drops junk like "N/A" or text)
    df["route"] = df["route"].astype(str).str.strip().str.extract(r"^(\d{1,3})")[0]
    df = df.dropna(subset=["route"])
    df["route"] = df["route"].astype(int)

    # direction: keep the standard values, bucket the rest as "other"
    if "direction" in df.columns:
        d = df["direction"].astype(str).str.strip().str.upper().str[0]
        df["direction"] = d.where(d.isin(list("NSEWB")), "O")
    else:
        df["direction"] = "O"

    # numeric delay, drop negatives/nonsense
    df["delay"] = pd.to_numeric(df["delay"], errors="coerce")
    df = df.dropna(subset=["delay"])
    df = df[(df["delay"] >= 0) & (df["delay"] < 300)]

    # time features
    df["hour"] = df["hour"].astype(int)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["date"].dt.month

    # target
    df["delayed"] = (df["delay"] >= DELAY_THRESHOLD).astype(int)
    return df


if __name__ == "__main__":
    df = clean(load(CSV_PATH))
    print("Clean shape:", df.shape)
    print("Class balance (delayed):")
    print(df["delayed"].value_counts(normalize=True).round(3))
    keep = ["date", "route", "direction", "hour", "day_of_week", "is_weekend", "month", "delay", "delayed"]
    df[keep].to_csv(OUT_PATH, index=False)
    print("Saved", OUT_PATH)