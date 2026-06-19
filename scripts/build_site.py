"""Build the publishable static site into public/.

This is the single entry point used by CI (and runnable locally) to regenerate
the dashboard from the latest available data and stage it for deployment:

  1. Download the freshest international results (GitHub mirror, no auth).
  2. Rebuild the processed match + feature tables.
  3. Run the visualization stack (model fit -> simulation -> charts -> HTML).
  4. Copy the self-contained dashboard to public/index.html (what Vercel serves)
     and copy the PNG/CSV artifacts alongside for direct linking.

Because the dashboard HTML embeds its images as base64, the deployable site is
essentially one self-contained file, which keeps hosting trivial and fast.

Usage:
    python scripts/build_site.py --sims 30000
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
REPORTS = ROOT / "reports"
PUBLIC = ROOT / "public"


def run(args: list[str]) -> None:
    print(f"\n$ {' '.join(args)}")
    subprocess.run([PY, *args], cwd=ROOT, check=True)


def main() -> None:
    n_sims = "30000"
    if "--sims" in sys.argv:
        n_sims = sys.argv[sys.argv.index("--sims") + 1]

    # 1-2. Refresh data and rebuild processed tables.
    run(["scripts/01_download_data.py", "--force"])
    run(["-m", "src.data.ingest"])
    run(["-m", "src.data.features"])

    # 3. Generate the visualization stack.
    run(["scripts/08_visualize.py", "--sims", n_sims])

    # 4. Stage into public/.
    PUBLIC.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPORTS / "dashboard.html", PUBLIC / "index.html")
    for name in ("bracket.png", "champion_bar.png", "round_heatmap.png",
                 "games.csv"):
        src = REPORTS / name
        if src.exists():
            shutil.copyfile(src, PUBLIC / name)

    print(f"\nSite built -> {PUBLIC / 'index.html'}")


if __name__ == "__main__":
    main()
