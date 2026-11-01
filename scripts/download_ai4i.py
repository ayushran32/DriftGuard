"""Download the real AI4I 2020 dataset to data/ai4i2020.csv  (needs internet).

    python scripts/download_ai4i.py

Tries the `ucimlrepo` package first, then the UCI zip URL. If both fail, download the CSV
manually from https://archive.ics.uci.edu/dataset/601 and save it as data/ai4i2020.csv.
NOTE: written without access to UCI from the build environment, so it is untested; the CSV
loader itself (driftguard.data.load_raw_csv) is tested with a stand-in file in the same format.
"""
import io, sys, zipfile, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "ai4i2020.csv"
URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"


def via_ucimlrepo():
    from ucimlrepo import fetch_ucirepo
    return fetch_ucirepo(id=601).data.original


def via_url():
    import pandas as pd
    import ssl
    ctx = ssl._create_unverified_context()
    raw = urllib.request.urlopen(URL, context=ctx, timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        return pd.read_csv(z.open(name))


def main():
    df = None
    for fn in (via_ucimlrepo, via_url):
        try:
            df = fn(); print("downloaded via", fn.__name__); break
        except Exception as e:
            print(f"{fn.__name__} failed: {e}")
    if df is None:
        sys.exit("Could not download. Get the CSV manually from the UCI page and save it as data/ai4i2020.csv")
    OUT.parent.mkdir(exist_ok=True); df.to_csv(OUT, index=False)
    from driftguard.data import load_raw_csv
    check = load_raw_csv(OUT)       # raises a clear error if columns are not as expected
    print(f"saved {OUT}  rows={len(check)}  columns={list(check.columns)}")


if __name__ == "__main__":
    main()
