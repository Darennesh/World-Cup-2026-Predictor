"""Step 8: generate the full visualization stack for the 2026 predictions.

Produces, under reports/:
  * bracket.png         -- expected knockout bracket with per-game probabilities
  * champion_bar.png    -- title-win probability by team
  * round_heatmap.png   -- reach-probability grid
  * games.csv           -- every predicted knockout tie + win probability
  * dashboard.html      -- a single page embedding all of the above

Reuses the validated LightGBM model and the same simulation engine that powers
the headline probabilities, so the pictures are consistent with the numbers.

Usage:
    python scripts/08_visualize.py --sims 30000
"""
from __future__ import annotations

import base64
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR, ROOT  # noqa: E402
from src.models.gbm import GBMModel  # noqa: E402
from src.ratings.bayesian import (fit_prior_then_update,  # noqa: E402
                                  group_results_2026)
from src.ratings.adjustments import load_adjustments  # noqa: E402
from src.simulation.sampler import ScoreSampler  # noqa: E402
from src.simulation.engine import Tournament  # noqa: E402
from src.simulation.groups_2026 import resolve_groups  # noqa: E402
from src.simulation.bracket import build_expected_bracket  # noqa: E402
from src.simulation.group_forecast import (forecast_group,  # noqa: E402
                                           played_results_for)
from src.visualization.plots import (plot_bracket, plot_champion_bar,  # noqa: E402
                                     plot_round_heatmap)

REPORTS = ROOT / "reports"


def _img_tag(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;">'


def _games_table_html(games_df: pd.DataFrame) -> str:
    rows = []
    for _, r in games_df.iterrows():
        rows.append(
            f"<tr><td>{r['round']}</td><td>{r['home']}</td>"
            f"<td style='text-align:right'>{r['p_home']:.0%}</td>"
            f"<td>{r['away']}</td>"
            f"<td style='text-align:right'>{1-r['p_home']:.0%}</td>"
            f"<td><b>{r['winner']}</b></td></tr>"
        )
    return ("<table><thead><tr><th>Round</th><th>Team A</th><th>P(A)</th>"
            "<th>Team B</th><th>P(B)</th><th>Predicted winner</th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table>")


def _group_forecast_html(forecasts: list) -> str:
    """Render per-group remaining-fixture predictions and advance odds."""
    blocks = []
    for fc in forecasts:
        fix_rows = "".join(
            f"<tr><td>{m.home}</td>"
            f"<td style='text-align:center'>{m.exp_home:.1f}&ndash;{m.exp_away:.1f}</td>"
            f"<td>{m.away}</td>"
            f"<td style='text-align:right'>{m.p_home:.0%}/{m.p_draw:.0%}/{m.p_away:.0%}</td></tr>"
            for m in fc.remaining
        )
        adv = sorted(fc.advance_prob.items(), key=lambda kv: kv[1], reverse=True)
        adv_rows = "".join(
            f"<tr><td>{t}</td>"
            f"<td style='text-align:right'>{p:.0%}</td>"
            f"<td style='text-align:right'>{fc.finish_first[t]:.0%}</td></tr>"
            for t, p in adv
        )
        fixtures = (f"<table><thead><tr><th>Home</th><th>xG</th><th>Away</th>"
                    f"<th>W/D/L</th></tr></thead><tbody>{fix_rows}</tbody></table>"
                    if fc.remaining else "<p><i>All matches played.</i></p>")
        blocks.append(
            f"<div style='flex:1 1 460px'><h3 style='margin:.2rem 0;color:#0b3d2e'>"
            f"Group {fc.name}</h3>{fixtures}"
            f"<table style='margin-top:6px'><thead><tr><th>Team</th>"
            f"<th>Advance</th><th>Win grp</th></tr></thead><tbody>{adv_rows}"
            f"</tbody></table></div>"
        )
    return f"<div style='display:flex;flex-wrap:wrap;gap:18px'>{''.join(blocks)}</div>"


def main() -> None:
    n_sims = 30_000
    if "--sims" in sys.argv:
        n_sims = int(sys.argv[sys.argv.index("--sims") + 1])

    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")

    elo, update = fit_prior_then_update(matches)
    print(f"Fitting model (2026 group matches applied: {update.n_group_matches}) ...")
    model = GBMModel().fit(features, matches)

    # Apply any curated team-news (injury/return) adjustments.
    adjustments = load_adjustments()
    if adjustments:
        model.apply_adjustments(adjustments)
        print(f"Applied team-news adjustments: {adjustments}")

    groups, source = resolve_groups(matches, elo)
    known = set(model.teams)
    groups = {g: [t for t in teams if t in known] for g, teams in groups.items()}
    groups = {g: teams for g, teams in groups.items() if len(teams) == 4}
    print(f"Groups: {source}")

    sampler = ScoreSampler(model)
    tt = Tournament(groups, sampler)

    print(f"Running {n_sims:,} simulations for round probabilities ...")
    pred = tt.run(n_sims=n_sims)

    print("Building expected bracket with per-game probabilities ...")
    bracket = build_expected_bracket(tt)

    print("Forecasting remaining group-stage matches ...")
    all_2026 = group_results_2026(matches)
    group_forecasts = [
        forecast_group(model, name, teams,
                       played_results_for(teams, all_2026), n_sims=4000)
        for name, teams in groups.items()
    ]

    # Persist per-game predictions.
    games = [{"round": g.round, "home": g.home, "away": g.away,
              "p_home": g.p_home, "winner": g.winner}
             for g in bracket.all_games()]
    games_df = pd.DataFrame(games)
    REPORTS.mkdir(parents=True, exist_ok=True)
    games_df.to_csv(REPORTS / "games.csv", index=False)

    print("Rendering charts ...")
    p_bracket = plot_bracket(bracket, REPORTS / "bracket.png")
    p_bar = plot_champion_bar(pred, REPORTS / "champion_bar.png")
    p_heat = plot_round_heatmap(pred, REPORTS / "round_heatmap.png")

    # Assemble a self-contained dashboard.
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    champ_col = [c for c in pred.columns if c.startswith("P_")][-1]
    champ_prob = pred.iloc[0][champ_col]
    desc = (f"{bracket.champion} are the model's favourites to win the 2026 "
            f"FIFA World Cup ({champ_prob:.0%}). Full bracket, per-game "
            f"probabilities and Monte Carlo forecasts.")

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>2026 World Cup Predictions &mdash; {bracket.champion} favoured</title>
<meta name="description" content="{desc}">
<!-- Open Graph / social sharing -->
<meta property="og:type" content="website">
<meta property="og:title" content="2026 FIFA World Cup &mdash; Model Predictions">
<meta property="og:description" content="{desc}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="2026 FIFA World Cup &mdash; Model Predictions">
<meta name="twitter:description" content="{desc}">
<style>
 :root {{ --green:#1b7837; --ink:#1a1a1a; --muted:#666; }}
 * {{ box-sizing: border-box; }}
 body {{ font-family: system-ui, -apple-system, Segoe UI, Arial, sans-serif;
        margin: 0; color: var(--ink); background:#fafafa; line-height:1.5; }}
 header {{ background: linear-gradient(135deg,#0b3d2e,#1b7837); color:#fff;
          padding: 28px 24px; }}
 header h1 {{ margin: 0 0 6px; font-size: 1.7rem; }}
 header p {{ margin: 0; opacity: .92; }}
 main {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
 section {{ background:#fff; border:1px solid #eee; border-radius:10px;
           padding:18px; margin: 22px 0; box-shadow: 0 1px 3px rgba(0,0,0,.05); }}
 section h2 {{ margin-top: 0; font-size: 1.15rem; color: var(--green); }}
 img {{ max-width: 100%; height: auto; }}
 table {{ border-collapse: collapse; font-size: 13px; width: 100%; }}
 th, td {{ border: 1px solid #e6e6e6; padding: 5px 9px; }}
 th {{ background:#f3f7f4; text-align:left; }}
 .badge {{ display:inline-block; background:#fff; color:var(--green);
          border-radius:999px; padding:2px 12px; font-weight:700;
          font-size:.95rem; }}
 footer {{ color: var(--muted); font-size: 12px; text-align:center;
          padding: 24px; }}
 footer code {{ background:#eee; padding:1px 5px; border-radius:4px; }}
</style></head><body>
<header>
 <h1>2026 FIFA World Cup &mdash; Model Predictions</h1>
 <p>Predicted champion: <span class="badge">{bracket.champion} &middot; {champ_prob:.1%}</span></p>
</header>
<main>
<p class="sub" style="color:var(--muted)">Group layout: {source} &middot;
 {n_sims:,} Monte Carlo simulations &middot; last updated {updated}</p>
<section><h2>Expected knockout bracket</h2>{_img_tag(p_bracket)}</section>
<section><h2>Live group-stage forecast</h2>
 <p style="color:var(--muted);margin-top:0">Predicted result of each remaining
  group fixture (expected goals and win/draw/loss), with each team's chance of
  advancing. Reflects current form blended with long-run strength, so strong
  sides that stumbled early are still favoured to recover.</p>
 {_group_forecast_html(group_forecasts)}</section>
<section><h2>Title probability</h2>{_img_tag(p_bar)}</section>
<section><h2>Reach-round probabilities</h2>{_img_tag(p_heat)}</section>
<section><h2>Per-game knockout predictions</h2>{_games_table_html(games_df)}</section>
</main>
<footer>
 Built with a LightGBM goal model + Monte Carlo tournament simulation.
 Forecasts are probabilistic; football is high-variance. &middot;
 Updated {updated}.
</footer>
</body></html>"""
    (REPORTS / "dashboard.html").write_text(html, encoding="utf-8")

    print(f"\nDone. Open: {REPORTS / 'dashboard.html'}")
    print(f"  bracket : {p_bracket}")
    print(f"  bar     : {p_bar}")
    print(f"  heatmap : {p_heat}")
    print(f"  games   : {REPORTS / 'games.csv'}")


if __name__ == "__main__":
    main()
