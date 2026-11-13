"""Download SciFact from the official release URL given in configs/config.yaml."""
from _common import load_config, resolve
from src.data import download_scifact

cfg = load_config()
data_dir = download_scifact(cfg["data"]["url"], resolve(cfg["data"]["dir"]))
print("data directory:", data_dir)
for p in sorted(data_dir.iterdir()):
    print(f"  {p.name}  {p.stat().st_size} bytes")
print("Read the licence terms before using the data and record them in data/README.md.")
