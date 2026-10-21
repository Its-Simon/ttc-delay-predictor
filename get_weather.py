import json
import urllib.parse
import urllib.request
from datetime import date, timedelta

import pandas as pd

LAT, LON = 43.6532, -79.3832  # downtown Toronto
WINDOW_DAYS = 180  # Open-Meteo caps how much one request can cover
MIN_AMOUNT = 0.1  # mm of rain / cm of snow in the hour to count as rain or snow

dates = pd.read_csv("ttc_clean.csv", usecols=["date"], parse_dates=["date"])["date"]
start = dates.min().date()
# the archive lags a few days behind today
end = min(dates.max().date(), date.today() - timedelta(days=7))
print(f"Fetching hourly weather {start} to {end}")


def fetch(s, e):
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": s.isoformat(),
        "end_date": e.isoformat(),
        "hourly": "temperature_2m,precipitation,rain,snowfall",
        "timezone": "America/Toronto",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)["hourly"]


frames = []
cur = start
while cur <= end:
    stop = min(cur + timedelta(days=WINDOW_DAYS - 1), end)
    frames.append(pd.DataFrame(fetch(cur, stop)))
    print("  got", cur, "to", stop)
    cur = stop + timedelta(days=1)

w = pd.concat(frames, ignore_index=True)
t = pd.to_datetime(w["time"])
out = pd.DataFrame(
    {
        "date": t.dt.normalize(),
        "hour": t.dt.hour,
        "temp_c": w["temperature_2m"].round(1),
        "precip_mm": w["precipitation"],
        "is_rain": (w["rain"] >= MIN_AMOUNT).astype(int),
        "is_snow": (w["snowfall"] >= MIN_AMOUNT).astype(int),
    }
).dropna(subset=["temp_c"])

out.to_csv("weather.csv", index=False)
print(f"Saved weather.csv ({len(out)} hourly rows)")
print(out[["temp_c", "is_rain", "is_snow"]].describe().round(2))