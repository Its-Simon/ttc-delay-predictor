import json
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)
model = joblib.load("delay_model.joblib")

# written by train_model.py, so the page text always matches the trained model
try:
    with open("model_info.json") as fh:
        INFO = json.load(fh)
except FileNotFoundError:
    INFO = {}
THRESHOLD = INFO.get("threshold", 10)
USES_WEATHER = INFO.get("uses_weather", True)
BASE_RATE = INFO.get("base_rate", 0.7)
N_NOTE = "~{:,}".format(round(INFO["n_rows"], -3)) if "n_rows" in INFO else "thousands of"

# routes the model actually saw, so the dropdown only offers valid ones
encoder = model.named_steps["pre"].named_transformers_["cat"]
ROUTES = sorted(int(r) for r in encoder.categories_[0])

DIRECTIONS = {
    "N": "North",
    "S": "South",
    "E": "East",
    "W": "West",
    "B": "Both",
    "O": "Other",
}
CONDITIONS = [("clear", "Clear"), ("rain", "Rain"), ("snow", "Snow")]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TTC bus delay predictor</title>
  <style>
    :root {
      --paper: #f7f6f2;
      --ink: #1b1d21;
      --muted: #6b6f76;
      --line: #d9d8d2;
      --ttc: #da291c;
      --sign: #15130f;
      --amber: #ffb200;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; background: var(--paper); color: var(--ink);
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; line-height: 1.45;
    }
    main { max-width: 920px; margin: 0 auto; padding: 48px 24px 64px; }
    h1 { font-size: 34px; line-height: 1.1; letter-spacing: -0.02em; margin: 0 0 10px; font-weight: 700; }
    .lede { color: var(--muted); margin: 0 0 36px; max-width: 56ch; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 56px; align-items: start; }
    @media (max-width: 760px) { .grid { grid-template-columns: 1fr; gap: 32px; } h1 { font-size: 28px; } }

    form { display: grid; gap: 22px; }
    fieldset { border: 0; padding: 0; margin: 0; min-width: 0; }
    legend, .lbl { font-size: 14px; font-weight: 600; margin-bottom: 8px; padding: 0; display: block; }
    select {
      width: 100%; padding: 11px 12px; font: inherit; color: var(--ink);
      background: #fff; border: 1px solid var(--line); border-radius: 4px;
    }
    :focus-visible { outline: 3px solid var(--ttc); outline-offset: 2px; }

    .seg { display: flex; flex-wrap: wrap; gap: 0; border: 1px solid var(--line); border-radius: 4px; overflow: hidden; background: #fff; }
    .seg label { flex: 1 1 auto; text-align: center; }
    .seg input { position: absolute; opacity: 0; pointer-events: none; }
    .seg span { display: block; padding: 10px 8px; cursor: pointer; font-size: 14px; border-right: 1px solid var(--line); }
    .seg label:last-child span { border-right: 0; }
    .seg input:checked + span { background: var(--ink); color: #fff; }
    .seg input:focus-visible + span { outline: 3px solid var(--ttc); outline-offset: -3px; }

    .slider-row { display: flex; align-items: center; gap: 14px; }
    .slider-row output { min-width: 78px; text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
    input[type=range] { flex: 1; accent-color: var(--ttc); }

    .out { position: sticky; top: 24px; }
    .sign {
      background: var(--sign); color: var(--amber); border-radius: 6px;
      padding: 18px 22px; display: flex; align-items: baseline; gap: 18px;
      border: 3px solid #2b2822; box-shadow: inset 0 0 0 2px #000;
      font-family: "Arial Narrow", "Helvetica Neue Condensed", "Helvetica Neue", Arial, sans-serif;
      font-weight: 700; text-shadow: 0 0 10px rgba(255, 178, 0, .45);
    }
    .sign .num { font-size: 64px; line-height: 1; letter-spacing: 0.02em; }
    .sign .dir { font-size: 30px; letter-spacing: 0.08em; text-transform: uppercase; }
    .when { color: var(--muted); font-size: 14px; margin: 12px 0 26px; }

    .pct { font-size: 76px; line-height: 1; font-weight: 700; letter-spacing: -0.03em; font-variant-numeric: tabular-nums; }
    .msg { font-size: 18px; margin: 8px 0 18px; }
    .meter { position: relative; height: 14px; background: #e6e5df; border-radius: 2px; }
    .fill { height: 100%; width: 0; background: var(--ink); border-radius: 2px; transition: width .35s ease, background-color .2s; }
    .fill.long { background: var(--ttc); }
    .tick { position: absolute; top: -5px; bottom: -5px; width: 2px; background: var(--ink); }
    .note { color: var(--muted); font-size: 14px; margin: 14px 0 0; max-width: 46ch; }
    .err { color: #b00020; font-size: 15px; margin-top: 12px; min-height: 1.4em; }
    @media (prefers-reduced-motion: reduce) { .fill { transition: none; } }
  </style>
</head>
<body>
<main>
  <h1>Will that bus delay run long?</h1>
  <p class="lede">Pick a route and a time{% if uses_weather %}, plus the weather{% endif %}. A model trained on {{ n_note }} TTC bus delay reports estimates the chance the delay lasts {{ threshold }} minutes or more.</p>

  <div class="grid">
    <form id="f" onsubmit="return false">
      <div>
        <label class="lbl" for="route">Route</label>
        <select id="route" name="route">
          {% for r in routes %}<option value="{{ r }}" {{ 'selected' if r == d.route }}>{{ r }}</option>{% endfor %}
        </select>
      </div>

      <fieldset>
        <legend>Direction</legend>
        <div class="seg">
          {% for k, name in directions.items() %}
          <label><input type="radio" name="direction" value="{{ k }}" data-sign="{{ name }}" {{ 'checked' if k == d.direction }}><span>{{ name }}</span></label>
          {% endfor %}
        </div>
      </fieldset>

      <div>
        <label class="lbl" for="hour">Time of day</label>
        <div class="slider-row">
          <input type="range" id="hour" name="hour" min="0" max="23" value="{{ d.hour }}">
          <output id="hourOut" for="hour"></output>
        </div>
      </div>

      <fieldset>
        <legend>Day</legend>
        <div class="seg">
          {% for i, name in days %}
          <label><input type="radio" name="day_of_week" value="{{ i }}" data-name="{{ name }}" {{ 'checked' if i == d.day_of_week }}><span>{{ name }}</span></label>
          {% endfor %}
        </div>
      </fieldset>

      <div>
        <label class="lbl" for="month">Month</label>
        <select id="month" name="month">
          {% for i, m in months %}<option value="{{ i }}" {{ 'selected' if i == d.month }}>{{ m }}</option>{% endfor %}
        </select>
      </div>

      {% if uses_weather %}
      <fieldset>
        <legend>Weather</legend>
        <div class="seg">
          {% for k, name in conditions %}
          <label><input type="radio" name="conditions" value="{{ k }}" data-name="{{ name }}" {{ 'checked' if k == d.conditions }}><span>{{ name }}</span></label>
          {% endfor %}
        </div>
      </fieldset>

      <div>
        <label class="lbl" for="temp_c">Temperature</label>
        <div class="slider-row">
          <input type="range" id="temp_c" name="temp_c" min="-25" max="35" value="{{ d.temp_c }}">
          <output id="tempOut" for="temp_c"></output>
        </div>
      </div>
      {% endif %}
    </form>

    <section class="out" aria-live="polite">
      <div class="sign" aria-hidden="true"><span class="num" id="signNum"></span><span class="dir" id="signDir"></span></div>
      <div class="when" id="when"></div>
      <div class="pct" id="pct">&ndash;</div>
      <div class="msg" id="msg"></div>
      <div class="meter"><div class="fill" id="fill"></div><div class="tick" id="tick" style="left: {{ base_pct }}%"></div></div>
      <p class="note">The tick marks {{ base_pct }}%, the share of all logged delays that ran {{ threshold }}+ minutes. Above it means riskier than usual.</p>
      <div class="err" id="err" role="alert"></div>
    </section>
  </div>
</main>

<script>
  const f = document.getElementById('f');
  const $ = (id) => document.getElementById(id);
  const MONTHS = {{ months_json|safe }};
  let timer, seq = 0;

  function fmtHour(h) {
    const suffix = h < 12 ? 'AM' : 'PM';
    return ((h % 12) || 12) + ':00 ' + suffix;
  }

  function labels() {
    const v = Object.fromEntries(new FormData(f).entries());
    const dir = f.querySelector('input[name=direction]:checked');
    const day = f.querySelector('input[name=day_of_week]:checked');
    const wx = f.querySelector('input[name=conditions]:checked');
    $('signNum').textContent = v.route;
    $('signDir').textContent = dir ? dir.dataset.sign : '';
    $('hourOut').textContent = fmtHour(+v.hour);
    let when = (day ? day.dataset.name : '') + ' at ' + fmtHour(+v.hour) + ', ' + MONTHS[+v.month - 1];
    if ($('tempOut')) {
      $('tempOut').textContent = v.temp_c + ' \u00b0C';
      when += ', ' + v.temp_c + ' \u00b0C, ' + (wx ? wx.dataset.name.toLowerCase() : '');
    }
    $('when').textContent = when;
    return v;
  }

  async function update() {
    const v = labels();
    const mine = ++seq;
    try {
      const r = await fetch('/api/predict?' + new URLSearchParams(v));
      if (!r.ok) throw new Error('bad response');
      const j = await r.json();
      if (mine !== seq) return;
      const pct = Math.round(j.probability * 100);
      $('pct').textContent = pct + '%';
      $('fill').style.width = pct + '%';
      $('fill').classList.toggle('long', j.long_delay);
      $('msg').textContent = j.long_delay
        ? 'More likely than not to run {{ threshold }}+ minutes.'
        : 'More likely to clear in under {{ threshold }} minutes.';
      $('err').textContent = '';
    } catch (e) {
      if (mine !== seq) return;
      $('err').textContent = 'Could not get a prediction. Check that the server is still running.';
    }
  }

  f.addEventListener('input', () => {
    labels();
    clearTimeout(timer);
    timer = setTimeout(update, 120);
  });
  update();
</script>
</body>
</html>
"""


def predict(route, direction, hour, day_of_week, month, temp_c, is_rain, is_snow):
    row = pd.DataFrame(
        [
            {
                "route": route,
                "direction": direction,
                "hour": hour,
                "day_of_week": day_of_week,
                "is_weekend": int(day_of_week >= 5),
                "month": month,
                "temp_c": temp_c,
                "is_rain": is_rain,
                "is_snow": is_snow,
            }
        ]
    )
    row = row[list(model.feature_names_in_)]  # only the columns this model was trained on
    return float(model.predict_proba(row)[0][1])


def parse(values):
    route = int(values["route"])
    direction = values["direction"]
    hour = int(values["hour"])
    day_of_week = int(values["day_of_week"])
    month = int(values["month"])
    temp_c = float(values.get("temp_c", 8))
    conditions = values.get("conditions", "clear")
    if route not in ROUTES:
        raise ValueError("Unknown route")
    if direction not in DIRECTIONS:
        raise ValueError("Unknown direction")
    if conditions not in ("clear", "rain", "snow"):
        raise ValueError("Unknown conditions")
    if not (0 <= hour <= 23 and 0 <= day_of_week <= 6 and 1 <= month <= 12 and -40 <= temp_c <= 45):
        raise ValueError("Value out of range")
    return (
        route, direction, hour, day_of_week, month, temp_c,
        int(conditions == "rain"), int(conditions == "snow"),
    )


@app.route("/")
def index():
    now = datetime.now()
    defaults = {
        "route": ROUTES[0],
        "direction": "N",
        "hour": now.hour,
        "day_of_week": now.weekday(),
        "month": now.month,
        "temp_c": 8,
        "conditions": "clear",
    }
    return render_template_string(
        PAGE,
        routes=ROUTES,
        directions=DIRECTIONS,
        conditions=CONDITIONS,
        days=list(enumerate(DAYS)),
        months=list(enumerate(MONTHS, start=1)),
        months_json=json.dumps(MONTHS),
        d=defaults,
        threshold=THRESHOLD,
        base_pct=round(BASE_RATE * 100),
        n_note=N_NOTE,
        uses_weather=USES_WEATHER,
    )


@app.route("/api/predict")
def api_predict():
    # e.g. /api/predict?route=32&direction=E&hour=8&day_of_week=1&month=10&temp_c=5&conditions=rain
    try:
        prob = predict(*parse(request.args))
    except (KeyError, ValueError):
        return jsonify(error="Bad or missing parameters"), 400
    return jsonify(probability=round(prob, 3), long_delay=prob >= 0.5)


if __name__ == "__main__":
    app.run(debug=True, port=5001)