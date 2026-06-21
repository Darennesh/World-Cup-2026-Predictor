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
from src.simulation.fixtures import upcoming_fixtures, load_schedule  # noqa: E402
from src.evaluation.tracker import (update_log, summarize,  # noqa: E402
                                    OUTCOME_LABELS, PREDICTIONS_START)
from src.visualization.plots import (plot_bracket, plot_champion_bar,  # noqa: E402
                                     plot_round_heatmap)

REPORTS = ROOT / "reports"
PUBLIC = ROOT / "public"
# The track record is persisted (and publicly served) so predictions stay
# locked across daily rebuilds and accumulate over the tournament.
TRACK_LOG = PUBLIC / "prediction_log.csv"


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
    """Render per-group remaining-fixture predictions and advance odds as cards."""
    blocks = []
    for fc in forecasts:
        # Standings ordered by advancement probability; top 2 highlighted.
        adv = sorted(fc.advance_prob.items(), key=lambda kv: kv[1], reverse=True)
        st_rows = ""
        for i, (t, p) in enumerate(adv):
            wg = fc.finish_first[t]
            cls = " class='g-q'" if i < 2 else ""
            st_rows += (
                f"<tr{cls}><td class='g-team'>{t}</td>"
                f"<td class='g-num'><div class='gbar'>"
                f"<i style='width:{p*100:.0f}%'></i><b>{p:.0%}</b></div></td>"
                f"<td class='g-num'><div class='gbar gw'>"
                f"<i style='width:{wg*100:.0f}%'></i><b>{wg:.0%}</b></div></td></tr>"
            )

        if fc.remaining:
            items = ""
            for m in fc.remaining:
                items += (
                    f"<li><div class='gf-row'>"
                    f"<span class='gf-h'>{m.home}</span>"
                    f"<span class='gf-xg'>{m.exp_home:.1f}&ndash;{m.exp_away:.1f}</span>"
                    f"<span class='gf-a'>{m.away}</span></div>"
                    f"<div class='gf-wdl'>"
                    f"{m.p_home:.0%} W &middot; {m.p_draw:.0%} D &middot; {m.p_away:.0%} L"
                    f"</div></li>"
                )
            fixtures = (f"<div class='gf-sub'>Remaining fixtures</div>"
                        f"<ul class='gf-list'>{items}</ul>")
        else:
            fixtures = "<p class='gf-done'>All matches played.</p>"

        blocks.append(
            f"<div class='gcard'><div class='gcard-h'>Group {fc.name}</div>"
            f"<table class='gtable'><thead><tr><th>Team</th>"
            f"<th>Advance</th><th>Win grp</th></tr></thead>"
            f"<tbody>{st_rows}</tbody></table>{fixtures}</div>"
        )
    return f"<div class='ggrid'>{''.join(blocks)}</div>"


def _upcoming_html(fixtures: list) -> str:
    """Render upcoming fixtures with EAT kickoff and the live prediction."""
    if not fixtures:
        return ("<p style='color:#666'>No upcoming group-stage fixtures &mdash; "
                "the group stage is complete.</p>")

    n_timed = sum(1 for f in fixtures if f.kickoff_utc is not None)
    note = ("" if n_timed == fixtures.__len__() else
            "<p style='color:#888;font-size:.82rem;margin:.2rem 0 8px'>"
            "Kickoff times shown in EAT (UTC+3) where scheduled; remaining times "
            "populate from <code>config/schedule_2026.yaml</code>. Predictions are "
            "live and update every build.</p>")

    rows = []
    for f in fixtures:
        fav = ("Draw" if f.favourite == "Draw"
               else f"<b>{f.favourite}</b>")
        conf = max(f.p_home, f.p_draw, f.p_away)
        rows.append(
            f"<tr><td style='white-space:nowrap'>{f.eat_label}</td>"
            f"<td style='text-align:center;color:#888'>{f.group}</td>"
            f"<td style='text-align:right'>{f.home}</td>"
            f"<td style='text-align:center;color:#888'>"
            f"{f.exp_home:.1f}&ndash;{f.exp_away:.1f}</td>"
            f"<td>{f.away}</td>"
            f"<td style='text-align:center'>"
            f"{f.p_home:.0%}/{f.p_draw:.0%}/{f.p_away:.0%}</td>"
            f"<td>{fav} <span style='color:#999'>({conf:.0%})</span></td></tr>"
        )
    table = (
        "<table><thead><tr><th>Kickoff (EAT)</th><th>Grp</th>"
        "<th style='text-align:right'>Home</th><th>xScore</th><th>Away</th>"
        "<th>P(H/D/A)</th><th>Prediction</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )
    return note + table



def _track_record_html(log: pd.DataFrame, stats: dict) -> str:
    """Render the model's locked predictions vs actual results + summary stats."""
    if log.empty or stats.get("n", 0) == 0:
        return ("<p style='color:#666'>No completed group-stage fixtures logged "
                "yet &mdash; the track record will populate as matches are played.</p>")

    # Summary cards.
    cards = [
        ("Matches scored", f"{stats['n']}"),
        ("Outcome accuracy", f"{stats['hit_rate']:.0%}"),
        ("Avg RPS (lower=better)", f"{stats['avg_rps']:.3f}"),
        ("vs base-rate RPS", f"{stats['base_rps']:.3f}"),
        ("Exact scoreline", f"{stats['exact_score']:.0%}"),
    ]
    card_html = "".join(
        f"<div style='flex:1 1 130px;background:#f3f7f4;border:1px solid #e0ebe4;"
        f"border-radius:8px;padding:10px 12px'>"
        f"<div style='font-size:1.3rem;font-weight:700;color:#0b3d2e'>{v}</div>"
        f"<div style='font-size:.78rem;color:#555'>{k}</div></div>"
        for k, v in cards
    )

    # Per-fixture table, most recent first.
    rows = []
    for r in log.sort_values("date", ascending=False).itertuples(index=False):
        pred_lbl = OUTCOME_LABELS[int(r.pred_outcome)]
        tick = ("<span class='pill pill-ok'>&#10004; hit</span>" if r.correct
                else "<span class='pill pill-no'>&#10008; miss</span>")
        conf = max(r.p_home, r.p_draw, r.p_away)
        rows.append(
            f"<tr><td>{pd.Timestamp(r.date).strftime('%b %d')}</td>"
            f"<td style='text-align:right'>{r.home_team}</td>"
            f"<td style='text-align:center;color:#888'>"
            f"{int(r.pred_home_goals)}&ndash;{int(r.pred_away_goals)}</td>"
            f"<td>{r.away_team}</td>"
            f"<td style='text-align:center'>{r.p_home:.0%}/{r.p_draw:.0%}/{r.p_away:.0%}</td>"
            f"<td>{pred_lbl} <span style='color:#999'>({conf:.0%})</span></td>"
            f"<td style='text-align:center;font-weight:700'>"
            f"{int(r.home_goals)}&ndash;{int(r.away_goals)}</td>"
            f"<td style='text-align:center'>{tick}</td>"
            f"<td style='text-align:right;color:#666'>{r.rps:.3f}</td></tr>"
        )
    table = (
        "<table><thead><tr><th>Date</th><th style='text-align:right'>Home</th>"
        "<th>xScore</th><th>Away</th><th>P(H/D/A)</th><th>Predicted</th>"
        "<th>Actual</th><th>Hit</th><th>RPS</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )
    cards_wrap = (f"<div style='display:flex;flex-wrap:wrap;gap:10px;"
                  f"margin-bottom:14px'>{card_html}</div>")
    return cards_wrap + table


def _next_upcoming(fixtures: list):
    """The next fixture to kick off (first future one, else the earliest)."""
    now = datetime.now(timezone.utc)
    future = [f for f in fixtures if f.kickoff_utc and f.kickoff_utc >= now]
    if future:
        return future[0]
    return fixtures[0] if fixtures else None


def _stat_card(value: str, label: str) -> str:
    return (f"<div class='stat'><div class='stat-val'>{value}</div>"
            f"<div class='stat-lbl'>{label}</div></div>")


def _hero_stats_html(champion: str, champ_prob: float, track_stats: dict,
                     next_match) -> str:
    """At-a-glance KPI cards shown in the hero."""
    cards = [_stat_card(f"{champ_prob:.0%}", f"{champion} to win")]
    if track_stats.get("n"):
        cards.append(_stat_card(f"{track_stats['hit_rate']:.0%}",
                                "Outcome accuracy"))
        cards.append(_stat_card(f"{track_stats['avg_rps']:.3f}",
                                "Avg RPS (lower better)"))
        cards.append(_stat_card(f"{track_stats['n']}", "Predictions scored"))
    if next_match is not None:
        cards.append(_stat_card(next_match.favourite if
                                next_match.favourite != "Draw" else "Even",
                                "Next-match pick"))
    return f"<div class='stat-grid'>{''.join(cards)}</div>"


def _next_match_html(f) -> str:
    """A prominent card for the next fixture to be played."""
    if f is None:
        return ""
    pick = ("an even contest" if f.favourite == "Draw"
            else f"<b>{f.favourite}</b> favoured")
    conf = max(f.p_home, f.p_draw, f.p_away)
    return (
        "<div class='nextmatch'>"
        f"<div class='nm-when'>&#9201; {f.eat_label} &middot; Group {f.group}</div>"
        f"<div class='nm-teams'>{f.home} <span class='nm-v'>vs</span> {f.away}</div>"
        f"<div class='nm-line'>Model: {pick} ({conf:.0%}) &middot; "
        f"expected score {f.exp_home:.1f}&ndash;{f.exp_away:.1f}</div>"
        "<div class='nm-bar'>"
        f"<span style='width:{f.p_home*100:.0f}%' class='seg seg-h' "
        f"title='{f.home} win'>{f.p_home:.0%}</span>"
        f"<span style='width:{f.p_draw*100:.0f}%' class='seg seg-d' "
        f"title='Draw'>{f.p_draw:.0%}</span>"
        f"<span style='width:{f.p_away*100:.0f}%' class='seg seg-a' "
        f"title='{f.away} win'>{f.p_away:.0%}</span>"
        "</div></div>"
    )


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

    print("Computing upcoming fixtures with EAT kickoff + live predictions ...")
    schedule = load_schedule()
    upcoming = upcoming_fixtures(model, matches, groups, schedule)
    n_timed = sum(1 for f in upcoming if f.kickoff_utc is not None)
    print(f"  {len(upcoming)} upcoming fixtures ({n_timed} with scheduled times)")

    print("Updating model track record (walk-forward predictions vs actuals) ...")
    track_log = update_log(matches, features, TRACK_LOG)
    track_stats = summarize(track_log)
    if track_stats.get("n"):
        print(f"  Logged {track_stats['n']} fixtures | "
              f"accuracy {track_stats['hit_rate']:.0%} | "
              f"avg RPS {track_stats['avg_rps']:.3f}")

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
    next_match = _next_upcoming(upcoming)
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
 :root {{
   --green:#15803d; --green-d:#0b3d2e; --gold:#d9a400; --ink:#13211b;
   --muted:#5d6b63; --line:#e6ece8; --bg:#f4f7f5; --card:#ffffff;
   --h:#1b7837; --d:#9aa0a6; --a:#b2182b;
   --head:#1e3a5f; --head-2:#3a6ea5; --head-soft:#eef2f8;
 }}
 * {{ box-sizing: border-box; }}
 html {{ scroll-behavior: smooth; }}
 body {{ font-family: 'Segoe UI', system-ui, -apple-system, Arial, sans-serif;
        margin:0; color:var(--ink); background:var(--bg); line-height:1.55; }}

 /* ---- Header / hero ---- */
 header {{ background:linear-gradient(135deg,#0b3d2e 0%,#15803d 60%,#1ea34d 100%);
          color:#fff; padding:34px 20px 30px; position:relative; overflow:hidden; }}
 header::after {{ content:""; position:absolute; right:-60px; top:-60px;
   width:240px; height:240px; border-radius:50%;
   background:rgba(255,255,255,.06); }}
 .hero-wrap {{ max-width:1140px; margin:0 auto; position:relative; z-index:1; }}
 .eyebrow {{ text-transform:uppercase; letter-spacing:.14em; font-size:.72rem;
            font-weight:700; opacity:.85; margin:0 0 6px; }}
 header h1 {{ margin:0 0 6px; font-size:1.95rem; font-weight:800; }}
 header .sub {{ margin:0; opacity:.9; font-size:.92rem; }}
 .badge {{ display:inline-block; background:var(--gold); color:#3a2b00;
          border-radius:999px; padding:3px 14px; font-weight:800;
          font-size:.95rem; }}
 .stat-grid {{ display:grid; gap:12px; margin-top:22px;
   grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); }}
 .stat {{ background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.18);
         border-radius:12px; padding:13px 15px; backdrop-filter:blur(3px); }}
 .stat-val {{ font-size:1.5rem; font-weight:800; line-height:1.1; }}
 .stat-lbl {{ font-size:.76rem; opacity:.9; margin-top:3px; }}

 /* ---- Sticky nav ---- */
 nav {{ position:sticky; top:0; z-index:20; background:rgba(255,255,255,.92);
       backdrop-filter:blur(8px); border-bottom:1px solid var(--line); }}
 .nav-wrap {{ max-width:1140px; margin:0 auto; display:flex; gap:4px;
   overflow-x:auto; padding:8px 16px; }}
 nav a {{ white-space:nowrap; text-decoration:none; color:var(--muted);
         font-size:.82rem; font-weight:600; padding:6px 12px; border-radius:8px; }}
 nav a:hover {{ background:var(--bg); color:var(--head); }}

 main {{ max-width:1140px; margin:0 auto; padding:8px 16px 24px; }}
 section {{ background:var(--card); border:1px solid var(--line);
   border-radius:16px; padding:20px 22px; margin:20px 0;
   box-shadow:0 1px 2px rgba(16,40,28,.04),0 6px 18px rgba(16,40,28,.04);
   scroll-margin-top:60px; }}
 .sec-head {{ display:flex; align-items:baseline; gap:10px; margin:0 0 4px; }}
 section h2 {{ margin:0; font-size:1.16rem; color:var(--head);
   display:flex; align-items:center; gap:9px; }}
 section h2::before {{ content:""; width:6px; height:20px; border-radius:3px;
   background:linear-gradient(var(--head),var(--head-2)); display:inline-block; }}
 .lead {{ color:var(--muted); margin:.35rem 0 14px; font-size:.9rem; }}
 img {{ max-width:100%; height:auto; border-radius:8px; }}

 /* ---- Tables ---- */
 .tbl-scroll {{ overflow-x:auto; }}
 table {{ border-collapse:collapse; font-size:13px; width:100%; min-width:520px; }}
 th, td {{ padding:7px 10px; border-bottom:1px solid var(--line); }}
 thead th {{ position:sticky; top:0; background:var(--head-soft); color:var(--head);
   text-align:left; font-weight:700; font-size:.78rem; text-transform:uppercase;
   letter-spacing:.03em; }}
 tbody tr:nth-child(even) {{ background:#fafcfb; }}
 tbody tr:hover {{ background:#eef3f9; }}

 /* ---- Next match card ---- */
 .nextmatch {{ background:linear-gradient(135deg,var(--head),var(--head-2)); color:#fff;
   border-radius:14px; padding:18px 20px; }}
 .nm-when {{ font-size:.8rem; opacity:.9; font-weight:600; }}
 .nm-teams {{ font-size:1.5rem; font-weight:800; margin:4px 0 2px; }}
 .nm-v {{ opacity:.65; font-weight:500; font-size:1rem; margin:0 6px; }}
 .nm-line {{ font-size:.86rem; opacity:.95; margin-bottom:10px; }}
 .nm-bar {{ display:flex; height:22px; border-radius:6px; overflow:hidden;
   font-size:.72rem; font-weight:700; }}
 .seg {{ display:flex; align-items:center; justify-content:center; color:#fff;
   min-width:26px; }}
 .seg-h {{ background:#0e7a39; }} .seg-d {{ background:#7d8a82; }}
 .seg-a {{ background:#b9532a; }}

 .pill {{ display:inline-block; padding:1px 8px; border-radius:999px;
   font-size:.72rem; font-weight:700; }}
 .pill-ok {{ background:#e3f3e9; color:#15803d; }}
 .pill-no {{ background:#fbe6e6; color:#b2182b; }}

 /* ---- Group stage cards ---- */
 .ggrid {{ display:grid; gap:16px;
   grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); }}
 .gcard {{ border:1px solid var(--line); border-radius:14px; overflow:hidden;
   background:#fff; box-shadow:0 1px 2px rgba(16,40,28,.04); }}
 .gcard-h {{ background:linear-gradient(135deg,var(--head),var(--head-2));
   color:#fff; font-weight:800; font-size:.92rem; letter-spacing:.02em;
   padding:9px 14px; }}
 .gtable {{ width:100%; min-width:0; border-collapse:collapse; font-size:12.5px; }}
 .gtable th {{ background:var(--head-soft); color:var(--head); padding:6px 12px;
   font-size:.68rem; text-transform:uppercase; letter-spacing:.04em;
   font-weight:700; text-align:center; }}
 .gtable th:first-child {{ text-align:left; }}
 .gtable td {{ padding:5px 12px; border-bottom:1px solid var(--line);
   vertical-align:middle; }}
 .gtable td.g-team {{ font-weight:600; }}
 .gtable tr.g-q {{ background:#f3f7fc; }}
 .gtable tr.g-q td.g-team {{ color:var(--head); }}
 .gtable tr.g-q td.g-team::before {{ content:"\\2713"; color:var(--green);
   font-weight:800; margin-right:5px; font-size:.78rem; }}
 .g-num {{ width:33%; }}
 .gbar {{ position:relative; height:18px; border-radius:5px; background:#eef1f5;
   overflow:hidden; }}
 .gbar > i {{ position:absolute; left:0; top:0; bottom:0; background:#bcd3e6; }}
 .gbar.gw > i {{ background:#e7d9a6; }}
 .gbar > b {{ position:absolute; inset:0; display:flex; align-items:center;
   justify-content:center; font-size:.72rem; font-weight:700; color:#23364a;
   font-style:normal; }}
 .gf-sub {{ font-size:.66rem; text-transform:uppercase; letter-spacing:.05em;
   color:var(--muted); font-weight:800; padding:10px 14px 2px; }}
 .gf-list {{ list-style:none; margin:0; padding:2px 14px 14px; }}
 .gf-list li {{ padding:7px 0; border-bottom:1px solid #f1f4f2; }}
 .gf-list li:last-child {{ border-bottom:none; }}
 .gf-row {{ display:flex; align-items:center; justify-content:space-between;
   gap:8px; font-size:12.5px; }}
 .gf-row .gf-h {{ font-weight:600; text-align:right; flex:1; }}
 .gf-row .gf-a {{ font-weight:600; flex:1; }}
 .gf-xg {{ color:#46586b; font-size:.72rem; background:var(--head-soft);
   border-radius:5px; padding:2px 7px; white-space:nowrap; font-weight:700; }}
 .gf-wdl {{ text-align:center; color:var(--muted); font-size:.72rem;
   margin-top:3px; letter-spacing:.02em; }}
 .gf-done {{ color:var(--muted); padding:8px 14px 14px; font-size:.84rem;
   font-style:italic; }}

 footer {{ color:var(--muted); font-size:12px; text-align:center; padding:26px 16px; }}
 footer code {{ background:#e7ece9; padding:1px 5px; border-radius:4px; }}
 @media (max-width:560px) {{
   header h1 {{ font-size:1.5rem; }}
   .nm-teams {{ font-size:1.2rem; }}
 }}
</style></head><body>
<header>
 <div class="hero-wrap">
  <p class="eyebrow">FIFA World Cup 2026 &middot; Mexico &middot; USA &middot; Canada</p>
  <h1>World Cup 2026 &mdash; AI Match Predictions</h1>
  <p class="sub">Predicted champion:
   <span class="badge">{bracket.champion} &middot; {champ_prob:.0%}</span>
   &nbsp;&middot;&nbsp; updated {updated}</p>
  {_hero_stats_html(bracket.champion, champ_prob, track_stats, next_match)}
 </div>
</header>
<nav><div class="nav-wrap">
 <a href="#next">Next match</a>
 <a href="#upcoming">Upcoming fixtures</a>
 <a href="#track">Track record</a>
 <a href="#title">Title odds</a>
 <a href="#bracket">Bracket</a>
 <a href="#groups">Group forecast</a>
 <a href="#rounds">Round odds</a>
 <a href="#ko">Knockout games</a>
</div></nav>
<main>

<section id="next">
 <h2>Next match</h2>
 <p class="lead">The next fixture to kick off, with the model's live call.</p>
 {_next_match_html(next_match)}
</section>

<section id="upcoming">
 <h2>Upcoming fixtures &amp; live predictions</h2>
 <p class="lead">Every remaining group-stage fixture with kickoff in East Africa
  Time (EAT) and the model's live call (win/draw/loss + expected score). Each
  game moves to the track record once played.</p>
 <div class="tbl-scroll">{_upcoming_html(upcoming)}</div>
</section>

<section id="track">
 <h2>Model track record</h2>
 <p class="lead">Every group-stage prediction the model made <i>before</i>
  kickoff (walk-forward, no look-ahead), scored against the actual result, since
  the model went live on {PREDICTIONS_START:%b %d, %Y}. Locked when first made.</p>
 <div class="tbl-scroll">{_track_record_html(track_log, track_stats)}</div>
</section>

<section id="title">
 <h2>Title probability</h2>
 <p class="lead">Each team's chance of lifting the trophy, across all simulated
  tournaments.</p>
 {_img_tag(p_bar)}
</section>

<section id="bracket">
 <h2>Expected knockout bracket</h2>
 <p class="lead">The most-likely path to the final, with the model's win
  probability for every tie.</p>
 {_img_tag(p_bracket)}
</section>

<section id="groups">
 <h2>Live group-stage forecast</h2>
 <p class="lead">Predicted result of each remaining group fixture and every
  team's chance of advancing &mdash; current form blended with long-run strength,
  so strong sides that stumbled early can still recover.</p>
 {_group_forecast_html(group_forecasts)}
</section>

<section id="rounds">
 <h2>Reach-round probabilities</h2>
 <p class="lead">How far each team is projected to go.</p>
 {_img_tag(p_heat)}
</section>

<section id="ko">
 <h2>Per-game knockout predictions</h2>
 <p class="lead">Win probability for every projected knockout tie.</p>
 <div class="tbl-scroll">{_games_table_html(games_df)}</div>
</section>

</main>
<footer>
 Forecasts are probabilistic; football is high-variance.
 &middot; Updated {updated}.
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
