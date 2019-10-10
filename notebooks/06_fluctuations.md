---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: crystal
    language: python
    name: crystal
---

```python
import pandas as pd
import altair as alt
import numpy as np
import sys

sys.path.append("../src")
import figures
import fluctuations

%matplotlib inline
import matplotlib.pyplot as plt
```

```python
file = "../data/analysis/fluctuation_rs.h5"
with pd.HDFStore(file) as src:
    df = src.get("ordering")
df = df.query(" 1.30 < temperature and temperature < 1.8")
df = df.query("pressure==13.50")
df = df[df["crystal"].isin(["liquid", "p2"])]
```

```python
c = (
    alt.Chart(df)
    .mark_line()
    .encode(
        x=alt.X("bins", title="Orientation Order"),
        y=alt.Y("mean(count)", title="Distribution"),
        color=alt.Color("crystal", title="Crystal"),
    )
)
with alt.data_transformers.enable("default"):
    c.save("../figures/fluctuation_distribution.svg")
c
```

```python
thermo = pd.read_hdf("../data/analysis/thermodynamics.h5", "thermo")
```

```python
xvals = np.linspace(0, 1, 1001)
df["log"] = -np.log(df["count"])
groups = df.groupby(["temperature", "pressure", "crystal"])
for index, group in groups:
    if index in ["p2gg", "pg"]:
        continue
    subset = group[group["log"] < 0]
    p = np.poly1d(np.polyfit(subset["bins"], subset["log"], 2))
    pe = thermo.loc[(13.50, 1.35, index), ("potential_energy", "mean")]
    plt.plot(xvals, p(xvals), label=index)

# plt.legend()
plt.ylim(-10, 4)
```

```python
thermo.loc[(13.50, 1.35, "p2"), ("potential_energy", "mean")]
```

```python
df_single = pd.read_csv("../data/analysis/fluctuation/dump-Trimer-P1.00-T0.40-p2.csv")
hist = fluctuations.aggregate(df_single["orient_order"])
hist_df = pd.DataFrame({"bins": hist[0], "count": hist[1]})
alt.Chart(hist_df).mark_line().encode(
    x=alt.X("bins", title="Orientation Order"),
    y=alt.Y("mean(count)", title="Distribution"),
    color=alt.Color("class", title="Crystal"),
)
```
