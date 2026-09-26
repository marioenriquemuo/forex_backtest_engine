# DSR — Deflated Sharpe Ratio (simple guide)

You tried many setups. The **best** one can look good just by luck.

**DSR** asks: *is it still good after we punish that luck?*

**Pass rule:** if **DSR ≥ 0.90** → you may keep the setup. Else → throw it away.

**Try it now** (from the repo folder):

```bash
python run_dsr_demo.py
```

Tiny fake bars. For **real D1** from Cross-Currency Basket Matrix cleaned CSVs:

```bash
python run_dsr_d1_basket.py
```

That script runs a real tiny engine search and prints `passes` / DSR / best knobs.

---

## Words you need

| Word | Meaning |
|------|---------|
| **Trial** | One backtest with one set of knobs |
| **N** | How many trials **you actually ran** (not how many you *could* have run) |
| **Sharpe** | Rough score of “how much you made vs how bumpy the ride was” |
| **DSR** | A score from **0 to 1**. Higher means “more likely the edge is real after many tries” |

---

## What you prepare (checklist)

- [ ] Lists of allowed knobs (example: stop sizes `[10, 20, 30]`, reward ratios `[1.5, 2.0]`)
- [ ] How many random tries (`n_trials`), example: `50`
- [ ] Same data and same engine settings on every try
- [ ] A function that builds a **fresh** strategy each try (do not reuse one dirty object)

---

## Do this

1. Write your lists of knobs.
2. Write `run_fn(params)` that builds a **new** strategy, runs the engine, and returns `equity_curve`.
3. Call `random_param_search(...)`.
4. Read `best_params`, `dsr["dsr"]`, and `passes`.
5. If `passes` is `True` (DSR ≥ 0.90), you may keep that setup. Else reject it.

```
lists of knobs
      |
      v
 random unique tries  (N = how many you ran)
      |
      v
 pick the best Sharpe
      |
      v
 compute DSR with that N
      |
      v
  DSR >= 0.90 ?  -->  keep / reject
```

---

## The big warning

**N = tries you ran.**

If you ran **50** random tries, **N is 50** — even if the full list of all combos is 10,000.

Do **not** lie about N. A fake big score helps nobody.

---

## Copy-paste example

This example uses fake equity curves (no live data). It shows the *shape* of the call.

```python
import numpy as np
from stress import random_param_search

# 1) Lists of knobs you allow
space = {
    "edge": [0.0001, 0.0005, 0.002],  # pretend "how strong the edge is"
}

# 2) One trial: take knobs -> return an equity curve
def run_fn(params):
    # IMPORTANT: in a real backtest, build a NEW strategy + OOEngine here.
    edge = float(params["edge"])
    return [100.0 * ((1.0 + edge) ** i) for i in range(120)]

# 3) Run random search (here the grid is tiny, so it runs all 3 combos)
out = random_param_search(
    run_fn,
    space,
    n_trials=50,              # ask for up to 50; only 3 unique combos exist
    rng=np.random.default_rng(0),
    periods_per_year=252.0,   # or pass index= from your bars
    dsr_threshold=0.90,
)

# 4) Read the answer
print("best knobs:", out["best_params"])
print("best Sharpe:", out["best_sharpe"])
print("N used:", out["n_trials"])
print("DSR:", out["dsr"]["dsr"])
print("passes (>= 0.90)?", out["passes"])

# 5) Keep only if passes is True
if out["passes"]:
    print("KEEP this setup")
else:
    print("REJECT this setup")
```

### Real engine sketch (you fill in your strategy)

```python
from data_handler import DataHandler
from engine import OOEngine
from stress import random_param_search
import numpy as np

def run_fn(params):
    # Build EVERYTHING fresh each trial
    handler = DataHandler(my_df, pair="EUR/USD")
    strategy = MyStrategy(sl_pips=params["sl_pips"], rr=params["rr"])
    result = OOEngine(
        {"EUR/USD": handler},
        {"EUR/USD": strategy},
        start_balance=100000.0,
        slippage_pips=1.0,
        max_risk_per_trade=0.01,
    ).run()
    return result  # equity_curve is read automatically

space = {"sl_pips": [10, 20, 30], "rr": [1.5, 2.0, 2.5]}
out = random_param_search(
    run_fn, space, n_trials=40, rng=np.random.default_rng(42), dsr_threshold=0.90
)
```

You can also score one finished equity curve yourself:

```python
from metrics import dsr_from_equity

info = dsr_from_equity(result.equity_curve, n_trials=40)  # N = tries YOU ran
print(info["dsr"], info["dsr"] >= 0.90)
```

---

## What this guide does *not* do

- No Optuna / Hyperopt
- No fancy “effective N” math for correlated knobs
- No live trading button

Adults who want the paper name: Bailey & López de Prado, *The Deflated Sharpe Ratio* (2014).
