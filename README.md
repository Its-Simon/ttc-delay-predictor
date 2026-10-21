# TTC Bus Delay Predictor

A machine learning web app that estimates the chance a reported TTC bus delay runs 10 minutes or more, based on route, direction, time of day, day of week, and month.

**Live demo:** _add your Render link here_

_Add a screenshot here: `![screenshot](screenshot.png)`_

## What it does

Pick a route, direction, time, day, and month. The app returns the probability that the delay lasts 10+ minutes, and compares it to the overall rate across all logged delays (about 71%), so you can see whether a given situation is riskier than usual.

## Data

- **TTC bus delay logs** from Toronto Open Data (Open Government Licence - Toronto), January 2025 onward. About 102,000 raw records, 99,810 after cleaning.
- **Hourly weather** for downtown Toronto from the Open-Meteo historical archive (modeled reanalysis data, not station readings). Used only in an experiment, see below.

## Method

1. `clean_ttc.py`: normalizes column names, keeps numeric bus routes, buckets odd direction values, drops invalid delays, builds time features, and sets the target `delayed` = delay of 10 or more minutes.
2. `get_weather.py`: downloads hourly weather for the date range of the delay data.
3. `train_model.py`: joins weather, splits the data **by time** (oldest 80% to train, newest 20% to test), compares models with and without weather, and saves the best one.
4. `app.py`: Flask app that loads the saved model and serves predictions in a browser and through a small JSON API.

Incident type is deliberately not used as a feature. It is only known after a delay happens, so using it would leak the answer.

## Results

Test set is the most recent 20% of the data, which the models never saw.

| Model | Weather | Accuracy | Macro F1 | ROC AUC |
|---|---|---|---|---|
| Majority-class baseline | no | 0.723 | 0.420 | 0.500 |
| Logistic regression | no | 0.733 | 0.687 | 0.770 |
| Random Forest | no | 0.776 | 0.719 | 0.784 |
| **Gradient boosting** | **no** | **0.776** | **0.723** | **0.798** |
| Logistic regression | yes | 0.726 | 0.683 | 0.769 |
| Random Forest | yes | 0.771 | 0.714 | 0.782 |
| Gradient boosting | yes | 0.772 | 0.719 | 0.794 |

Findings:

- Gradient boosting performed best (ROC AUC 0.798) and is the model the app uses.
- Weather features did not help. Every model scored the same or slightly worse with them, so the final model leaves them out.
- A time-based split gives a lower, more honest score than a random split would. An earlier random-split run scored about 79% accuracy.

## Limitations

- The logs only contain delays that already happened. The model answers "if a delay is reported, how likely is it to run 10+ minutes?", not "will a delay happen?". Predicting whether a delay occurs would need scheduled trip counts, for example from TTC GTFS data.
- Features are limited to route, direction, and time. Location, vehicle, and traffic data are not used.
- Probabilities have not been calibrated.

## Run it locally

```
pip install -r requirements.txt
python3 app.py
```

Then open http://127.0.0.1:5001.

To rebuild everything from the raw data:

```
python3 clean_ttc.py "<raw TTC bus delay csv>"
python3 get_weather.py
python3 train_model.py
```

## Stack

Python, pandas, scikit-learn, Flask, gunicorn.