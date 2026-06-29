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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR, ROOT  # noqa: E402
from src.models.gbm import GBMModel  # noqa: E402
from src.ratings.bayesian import (fit_prior_then_update,  # noqa: E402
                                  group_results_2026)
from src.ratings.adjustments import load_adjustments  # noqa: E402
from src.simulation.sampler import ScoreSampler  # noqa: E402
from src.simulation.engine import Tournament  # noqa: E402
from src.simulation.groups_2026 import resolve_groups, actual_knockout_bracket  # noqa: E402
from src.simulation.bracket import build_expected_bracket  # noqa: E402
from src.simulation.group_forecast import (forecast_group,  # noqa: E402
                                           played_results_for)
from src.simulation.fixtures import (upcoming_fixtures, load_schedule,  # noqa: E402
                                     upcoming_knockout_fixtures)
from src.evaluation.tracker import (update_log, summarize,  # noqa: E402
                                    OUTCOME_LABELS, PREDICTIONS_START)
from src.visualization.plots import (plot_bracket, plot_champion_bar,  # noqa: E402
                                     plot_round_heatmap, plot_calibration)
from src.visualization.flags import flag, with_flag  # noqa: E402
from src.visualization.form import recent_form  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    ranked_probability_score, log_loss_3way, brier_score_multiclass,
    expected_calibration_error, classification_report_3way,
    confusion_matrix_3way, skill_score)

REPORTS = ROOT / "reports"
PUBLIC = ROOT / "public"
# The track record is persisted (and publicly served) so predictions stay
# locked across daily rebuilds and accumulate over the tournament.
TRACK_LOG = PUBLIC / "prediction_log.csv"


def _img_tag(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;">'


def _wdl_chip(p_home: float, p_draw: float, p_away: float) -> str:
    """A compact win/draw/loss probability bar with the project colour code."""
    return (
        "<span class='wdl' title='Home / Draw / Away'>"
        f"<i class='wdl-h' style='width:{p_home*100:.0f}%'>{p_home:.0%}</i>"
        f"<i class='wdl-d' style='width:{p_draw*100:.0f}%'>{p_draw:.0%}</i>"
        f"<i class='wdl-a' style='width:{p_away*100:.0f}%'>{p_away:.0%}</i>"
        "</span>"
    )


def _form_dots(results: list[str]) -> str:
    """Render a team's recent results as small W/D/L dots (oldest -> newest)."""
    if not results:
        return "<span class='form'></span>"
    dots = "".join(f"<i class='fd fd-{r.lower()}' title='{r}'></i>"
                   for r in results)
    return f"<span class='form'>{dots}</span>"


def _games_table_html(games_df: pd.DataFrame) -> str:
    rows = []
    for _, r in games_df.iterrows():
        home_w = r['winner'] == r['home']
        rows.append(
            f"<tr><td><span class='rnd'>{r['round']}</span></td>"
            f"<td class='{'win' if home_w else ''}'>{with_flag(r['home'])}</td>"
            f"<td style='text-align:right'>{r['p_home']:.0%}</td>"
            f"<td class='{'' if home_w else 'win'}'>{with_flag(r['away'])}</td>"
            f"<td style='text-align:right'>{1-r['p_home']:.0%}</td>"
            f"<td><b>{with_flag(r['winner'])}</b></td></tr>"
        )
    return ("<table><thead><tr><th>Round</th><th>Team A</th><th>P(A)</th>"
            "<th>Team B</th><th>P(B)</th><th>Predicted winner</th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table>")


def _group_forecast_html(forecasts: list, form_map: dict[str, list[str]]) -> str:
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
                f"<tr{cls}><td class='g-team'>{flag(t)} {t}"
                f"{_form_dots(form_map.get(t, []))}</td>"
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
                    f"<span class='gf-h'>{m.home} {flag(m.home)}</span>"
                    f"<span class='gf-xg'>{m.exp_home:.1f}&ndash;{m.exp_away:.1f}</span>"
                    f"<span class='gf-a'>{flag(m.away)} {m.away}</span></div>"
                    f"<div class='gf-chip'>{_wdl_chip(m.p_home, m.p_draw, m.p_away)}</div>"
                    f"</li>"
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
        return ("<p style='color:#666'>No upcoming fixtures &mdash; "
                "the next round's matchups are still being decided.</p>")

    knockout = any(getattr(f, "is_knockout", False) for f in fixtures)
    col_label = "Round" if knockout else "Grp"

    n_timed = sum(1 for f in fixtures if f.kickoff_utc is not None)
    note = ("" if n_timed == fixtures.__len__() else
            "<p style='color:#888;font-size:.82rem;margin:.2rem 0 8px'>"
            "Kickoff times shown in EAT (UTC+3) where scheduled; remaining times "
            "populate from <code>config/schedule_2026.yaml</code>. Predictions are "
            "live and update every build.</p>")

    rows = []
    for f in fixtures:
        fav = ("Draw" if f.favourite == "Draw"
               else f"<b>{with_flag(f.favourite)}</b>")
        conf = (max(f.p_home, f.p_away) if getattr(f, "is_knockout", False)
                else max(f.p_home, f.p_draw, f.p_away))
        rows.append(
            f"<tr><td style='white-space:nowrap'>{f.eat_label}</td>"
            f"<td style='text-align:center;color:var(--muted);white-space:nowrap'>{f.group}</td>"
            f"<td style='text-align:right'>{with_flag(f.home)}</td>"
            f"<td style='text-align:center;color:var(--muted)'>"
            f"{f.exp_home:.1f}&ndash;{f.exp_away:.1f}</td>"
            f"<td>{with_flag(f.away)}</td>"
            f"<td style='min-width:140px'>{_wdl_chip(f.p_home, f.p_draw, f.p_away)}</td>"
            f"<td>{fav} <span style='color:var(--muted)'>({conf:.0%})</span></td></tr>"
        )
    table = (
        f"<table><thead><tr><th>Kickoff (EAT)</th><th>{col_label}</th>"
        "<th style='text-align:right'>Home</th><th>xScore</th><th>Away</th>"
        "<th>W / D / L</th><th>Prediction</th></tr></thead><tbody>"
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
        f"<div class='kpi'><div class='kpi-val'>{v}</div>"
        f"<div class='kpi-lbl'>{k}</div></div>"
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
            f"<tr><td style='white-space:nowrap'>{pd.Timestamp(r.date).strftime('%b %d')}</td>"
            f"<td style='text-align:right'>{with_flag(r.home_team)}</td>"
            f"<td style='text-align:center;color:var(--muted)'>"
            f"{int(r.pred_home_goals)}&ndash;{int(r.pred_away_goals)}</td>"
            f"<td>{with_flag(r.away_team)}</td>"
            f"<td style='min-width:140px'>{_wdl_chip(r.p_home, r.p_draw, r.p_away)}</td>"
            f"<td>{pred_lbl} <span style='color:var(--muted)'>({conf:.0%})</span></td>"
            f"<td style='text-align:center;font-weight:700'>"
            f"{int(r.home_goals)}&ndash;{int(r.away_goals)}</td>"
            f"<td style='text-align:center'>{tick}</td>"
            f"<td style='text-align:right;color:var(--muted)'>{r.rps:.3f}</td></tr>"
        )
    table = (
        "<table><thead><tr><th>Date</th><th style='text-align:right'>Home</th>"
        "<th>xScore</th><th>Away</th><th>W / D / L</th><th>Predicted</th>"
        "<th>Actual</th><th>Hit</th><th>RPS</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )
    cards_wrap = (f"<div class='kpi-grid'>{card_html}</div>")
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
            else f"<b>{with_flag(f.favourite)}</b> favoured")
    conf = max(f.p_home, f.p_draw, f.p_away)
    # Knockouts label the round directly; group games prefix "Group".
    stage = f.group if getattr(f, "is_knockout", False) else f"Group {f.group}"
    # ISO kickoff for the client-side countdown (omitted if time unknown).
    cd = ""
    if f.kickoff_utc is not None:
        iso = f.kickoff_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        cd = (f"<div class='nm-countdown' data-kickoff='{iso}'>"
              f"&#9203; kickoff time loading…</div>")
    return (
        "<div class='nextmatch'>"
        f"<div class='nm-when'>&#9201; {f.eat_label} &middot; {stage}</div>"
        f"<div class='nm-teams'>{with_flag(f.home)} <span class='nm-v'>vs</span> "
        f"{with_flag(f.away)}</div>"
        f"{cd}"
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


def _bracket_html(bracket) -> str:
    """Render the expected knockout bracket as responsive HTML (replaces the PNG).

    Each round is a column of tie cards; the favourite is highlighted and the
    model's win probability shown per side. Scrolls horizontally on small
    screens and uses flags for quick recognition.
    """
    cols = []
    for rnd, games in bracket.games_by_round.items():
        cards = []
        for g in games:
            home_w = g.winner == g.home
            cards.append(
                "<div class='bx-tie'>"
                f"<div class='bx-team {'bx-w' if home_w else ''}'>"
                f"<span>{with_flag(g.home)}</span>"
                f"<b>{g.p_home:.0%}</b></div>"
                f"<div class='bx-team {'bx-w' if not home_w else ''}'>"
                f"<span>{with_flag(g.away)}</span>"
                f"<b>{1-g.p_home:.0%}</b></div>"
                "</div>"
            )
        cols.append(
            f"<div class='bx-col'><div class='bx-rnd'>{rnd}</div>"
            f"{''.join(cards)}</div>"
        )
    champ = (f"<div class='bx-col bx-champ'><div class='bx-rnd'>Champion</div>"
             f"<div class='bx-trophy'>&#127942;<br>{with_flag(bracket.champion)}</div></div>")
    return f"<div class='bx-scroll'><div class='bx'>{''.join(cols)}{champ}</div></div>"


def _performance_html(log: pd.DataFrame, cal_png) -> str:
    """Probabilistic-quality metrics + calibration for the locked predictions."""
    if log.empty or len(log) < 3:
        return ("<p style='color:var(--muted)'>Not enough scored matches yet "
                "for a full performance breakdown &mdash; it populates as more "
                "games are played.</p>")
    probs = log[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    actual = log["actual_outcome"].to_numpy(dtype=int)

    base = np.bincount(actual, minlength=3) / len(actual)
    base_rps = ranked_probability_score(np.tile(base, (len(actual), 1)), actual)
    rps = ranked_probability_score(probs, actual)

    rep = classification_report_3way(probs, actual)
    cm = confusion_matrix_3way(probs, actual)

    kpis = [
        ("RPS (lower better)", f"{rps:.3f}"),
        ("Skill vs base rate", f"{skill_score(rps, base_rps):+.0%}"),
        ("Brier score", f"{brier_score_multiclass(probs, actual):.3f}"),
        ("Log loss", f"{log_loss_3way(probs, actual):.3f}"),
        ("Calibration error (ECE)", f"{expected_calibration_error(probs, actual):.3f}"),
        ("Accuracy", f"{rep['accuracy']:.0%}"),
    ]
    kpi_html = "".join(
        f"<div class='kpi'><div class='kpi-val'>{v}</div>"
        f"<div class='kpi-lbl'>{k}</div></div>" for k, v in kpis)

    # Per-outcome precision/recall/F1.
    pr_rows = "".join(
        f"<tr><td><b>{name}</b></td>"
        f"<td style='text-align:center'>{m['precision']:.0%}</td>"
        f"<td style='text-align:center'>{m['recall']:.0%}</td>"
        f"<td style='text-align:center'>{m['f1']:.0%}</td>"
        f"<td style='text-align:center;color:var(--muted)'>{m['support']}</td></tr>"
        for name, m in rep["classes"].items())
    pr_table = (
        "<table style='max-width:520px'><thead><tr><th>Outcome</th>"
        "<th>Precision</th><th>Recall</th><th>F1</th><th>N</th></tr></thead>"
        f"<tbody>{pr_rows}</tbody></table>")

    # Confusion matrix.
    labels = ["Home", "Draw", "Away"]
    cm_rows = ""
    for i, name in enumerate(labels):
        cells = "".join(
            f"<td style='text-align:center;"
            f"{'font-weight:700;color:var(--head)' if i==j else 'color:var(--muted)'}'>"
            f"{cm[i, j]}</td>" for j in range(3))
        cm_rows += f"<tr><td><b>{name}</b></td>{cells}</tr>"
    cm_table = (
        "<table style='max-width:420px'><thead><tr><th>Actual \\ Pred</th>"
        f"<th>Home</th><th>Draw</th><th>Away</th></tr></thead>"
        f"<tbody>{cm_rows}</tbody></table>")

    return (
        f"<div class='kpi-grid'>{kpi_html}</div>"
        f"<div style='display:flex;flex-wrap:wrap;gap:24px;align-items:flex-start'>"
        f"<div style='flex:1 1 360px;max-width:440px'>{_img_tag(cal_png)}</div>"
        f"<div style='flex:1 1 360px'>"
        f"<h3 style='color:var(--head);font-size:.95rem;margin:.2rem 0 6px'>"
        f"Per-outcome quality</h3>{pr_table}"
        f"<h3 style='color:var(--head);font-size:.95rem;margin:14px 0 6px'>"
        f"Confusion matrix</h3>{cm_table}</div></div>"
        f"<p style='color:var(--muted);font-size:.82rem;margin-top:12px'>"
        f"Football is high-variance; even strong models reach ~55&ndash;65% "
        f"single-match accuracy. The goal is calibration and beating the base "
        f"rate on RPS, not raw accuracy.</p>"
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

    # If the group stage is complete, lock the real Round-of-32 bracket so the
    # knockout forecast is conditioned on the actual qualifiers (not re-drawn).
    locked = actual_knockout_bracket(matches, groups)
    if locked is not None:
        tt = Tournament(groups, sampler, fixed_bracket=locked)
        print(f"Group stage complete -> locked real R32 bracket "
              f"({len(locked)} teams).")
    else:
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
    form_map = recent_form(matches, n=5)

    print("Computing upcoming fixtures with EAT kickoff + live predictions ...")
    schedule = load_schedule()
    if locked is not None:
        # Group stage done -> show the next knockout ties (participants decided).
        upcoming = upcoming_knockout_fixtures(model, matches, groups, locked,
                                              schedule)
        print(f"  {len(upcoming)} upcoming knockout fixture(s).")
    else:
        upcoming = upcoming_fixtures(model, matches, groups, schedule)
        n_timed = sum(1 for f in upcoming if f.kickoff_utc is not None)
        print(f"  {len(upcoming)} upcoming group fixtures "
              f"({n_timed} with scheduled times).")

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

    # Calibration plot from the locked track record (if enough data).
    p_cal = None
    if not track_log.empty and len(track_log) >= 3:
        cal_probs = track_log[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
        cal_out = track_log["actual_outcome"].to_numpy(dtype=int)
        p_cal = plot_calibration(cal_probs, cal_out, REPORTS / "calibration.png")

    # Assemble a self-contained dashboard.
    now_utc = datetime.now(timezone.utc)
    updated = now_utc.strftime("%Y-%m-%d %H:%M UTC")
    updated_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    champ_col = [c for c in pred.columns if c.startswith("P_")][-1]
    champ_prob = pred.iloc[0][champ_col]
    next_match = _next_upcoming(upcoming)
    desc = (f"{bracket.champion} are the model's favourites to win the 2026 "
            f"FIFA World Cup ({champ_prob:.0%}). Full bracket, live group-stage "
            f"and knockout predictions, updated daily.")

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
   --row-alt:#fafcfb; --row-hover:#eef3f9; --xg-bg:#eef2f8;
   --win-c:#0e7a39; --draw-c:#7d8a82; --loss-c:#b9532a;
 }}
 [data-theme="dark"] {{
   --ink:#e8eef0; --muted:#9fb0ad; --line:#26333a; --bg:#0e1518;
   --card:#161f24; --head:#8fb4e0; --head-2:#5b8bc4; --head-soft:#1b2730;
   --row-alt:#1a242a; --row-hover:#22303a; --xg-bg:#22303a;
 }}
 * {{ box-sizing: border-box; }}
 html {{ scroll-behavior: smooth; }}
 body {{ font-family: 'Segoe UI', system-ui, -apple-system, Arial, sans-serif;
        margin:0; color:var(--ink); background:var(--bg); line-height:1.55;
        transition:background .25s, color .25s; }}

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
 .theme-btn {{ position:absolute; top:18px; right:20px; z-index:2;
   background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.3);
   color:#fff; border-radius:999px; padding:6px 14px; cursor:pointer;
   font-size:.8rem; font-weight:700; }}
 .theme-btn:hover {{ background:rgba(255,255,255,.28); }}
 .badge {{ display:inline-block; background:var(--gold); color:#3a2b00;
          border-radius:999px; padding:3px 14px; font-weight:800;
          font-size:.95rem; }}
 .stat-grid {{ display:grid; gap:12px; margin-top:22px;
   grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); }}
 .stat {{ background:rgba(7,28,18,.30); border:1px solid rgba(255,255,255,.22);
         border-radius:12px; padding:13px 15px; }}
 .stat-val {{ font-size:1.5rem; font-weight:800; line-height:1.1; color:#fff;
   text-shadow:0 1px 2px rgba(0,0,0,.25); }}
 .stat-lbl {{ font-size:.76rem; opacity:.95; margin-top:3px; color:#eaf3ee; }}

 /* ---- Sticky nav / tabs ---- */
 nav {{ position:sticky; top:0; z-index:20; background:color-mix(in srgb,var(--card) 92%,transparent);
       backdrop-filter:blur(8px); border-bottom:1px solid var(--line); }}
 .nav-wrap {{ max-width:1140px; margin:0 auto; display:flex; gap:4px;
   overflow-x:auto; padding:8px 16px; }}
 nav a {{ white-space:nowrap; text-decoration:none; color:var(--muted);
         font-size:.82rem; font-weight:600; padding:6px 12px; border-radius:8px;
         cursor:pointer; transition:background .15s, color .15s; }}
 nav a:hover {{ background:var(--bg); color:var(--head); }}
 nav a.active {{ background:var(--head); color:#fff; }}

 /* ---- Pager (prev / next) ---- */
 .pager {{ display:flex; justify-content:space-between; align-items:center;
   gap:12px; margin:18px 0 4px; }}
 .pager button {{ background:var(--card); border:1px solid var(--line);
   color:var(--head); font-weight:700; font-size:.85rem; border-radius:10px;
   padding:9px 16px; cursor:pointer; transition:background .15s, border-color .15s; }}
 .pager button:hover:not(:disabled) {{ background:var(--head-soft);
   border-color:var(--head-2); }}
 .pager button:disabled {{ opacity:.4; cursor:not-allowed; }}
 .pager .pg-count {{ color:var(--muted); font-size:.82rem; font-weight:600; }}

 main {{ max-width:1140px; margin:0 auto; padding:8px 16px 24px; }}
 section {{ background:var(--card); border:1px solid var(--line);
   border-radius:16px; padding:20px 22px; margin:20px 0;
   box-shadow:0 1px 2px rgba(16,40,28,.04),0 6px 18px rgba(16,40,28,.04);
   scroll-margin-top:60px; }}
 /* Tabbed pagination: show only the active section. The no-JS fallback below
    reveals everything if scripting is disabled. */
 .js section {{ display:none; }}
 .js section.active {{ display:block; animation:fadeIn .35s ease; }}
 @keyframes fadeIn {{ from {{ opacity:0; transform:translateY(10px); }}
                     to {{ opacity:1; transform:none; }} }}
 .sec-head {{ display:flex; align-items:baseline; gap:10px; margin:0 0 4px; }}
 section h2 {{ margin:0; font-size:1.16rem; color:var(--head);
   display:flex; align-items:center; gap:9px; }}
 section h2::before {{ content:""; width:6px; height:20px; border-radius:3px;
   background:linear-gradient(var(--head),var(--head-2)); display:inline-block; }}
 .lead {{ color:var(--muted); margin:.35rem 0 14px; font-size:.9rem; }}
 img {{ max-width:100%; height:auto; border-radius:8px; }}
 img.emoji {{ height:1.05em; width:1.05em; margin:0 .12em 0 0; border-radius:0;
   vertical-align:-0.16em; display:inline-block; }}

 /* ---- Tables ---- */
 .tbl-scroll {{ overflow-x:auto; }}
 table {{ border-collapse:collapse; font-size:13px; width:100%; min-width:520px; }}
 th, td {{ padding:7px 10px; border-bottom:1px solid var(--line); }}
 thead th {{ position:sticky; top:0; background:var(--head-soft); color:var(--head);
   text-align:left; font-weight:700; font-size:.78rem; text-transform:uppercase;
   letter-spacing:.03em; }}
 tbody tr:nth-child(even) {{ background:var(--row-alt); }}
 tbody tr:hover {{ background:var(--row-hover); }}
 td.win {{ font-weight:700; color:var(--win-c); }}
 .rnd {{ display:inline-block; background:var(--head-soft); color:var(--head);
   font-size:.68rem; font-weight:800; letter-spacing:.04em; padding:1px 8px;
   border-radius:999px; }}

 /* ---- W/D/L chip (#2) ---- */
 .wdl {{ display:flex; height:18px; border-radius:5px; overflow:hidden;
   min-width:130px; font-size:.66rem; font-weight:700; }}
 .wdl i {{ display:flex; align-items:center; justify-content:center; color:#fff;
   font-style:normal; min-width:0; overflow:hidden; }}
 .wdl-h {{ background:var(--win-c); }}
 .wdl-d {{ background:var(--draw-c); }}
 .wdl-a {{ background:var(--loss-c); }}

 /* ---- Form dots (#4) ---- */
 .form {{ display:inline-flex; gap:3px; margin-left:7px; vertical-align:middle; }}
 .fd {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
 .fd-w {{ background:var(--win-c); }}
 .fd-d {{ background:var(--draw-c); }}
 .fd-l {{ background:var(--loss-c); }}

 /* ---- KPI cards (track record) ---- */
 .kpi-grid {{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:14px; }}
 .kpi {{ flex:1 1 130px; background:var(--head-soft); border:1px solid var(--line);
   border-radius:10px; padding:11px 13px; }}
 .kpi-val {{ font-size:1.35rem; font-weight:800; color:var(--head); }}
 .kpi-lbl {{ font-size:.75rem; color:var(--muted); margin-top:2px; }}

 /* ---- Next match card ---- */
 .nextmatch {{ background:linear-gradient(135deg,var(--head),var(--head-2)); color:#fff;
   border-radius:14px; padding:18px 20px; }}
 .nm-when {{ font-size:.8rem; opacity:.9; font-weight:600; }}
 .nm-teams {{ font-size:1.5rem; font-weight:800; margin:4px 0 2px; }}
 .nm-v {{ opacity:.65; font-weight:500; font-size:1rem; margin:0 6px; }}
 .nm-countdown {{ display:inline-block; background:rgba(255,255,255,.16);
   border-radius:999px; padding:3px 12px; font-size:.82rem; font-weight:700;
   margin:2px 0 8px; }}
 .nm-line {{ font-size:.86rem; opacity:.95; margin-bottom:10px; }}
 .nm-bar {{ display:flex; height:22px; border-radius:6px; overflow:hidden;
   font-size:.72rem; font-weight:700; }}
 .seg {{ display:flex; align-items:center; justify-content:center; color:#fff;
   min-width:26px; }}
 .seg-h {{ background:var(--win-c); }} .seg-d {{ background:var(--draw-c); }}
 .seg-a {{ background:var(--loss-c); }}

 .pill {{ display:inline-block; padding:1px 8px; border-radius:999px;
   font-size:.72rem; font-weight:700; }}
 .pill-ok {{ background:#e3f3e9; color:#15803d; }}
 .pill-no {{ background:#fbe6e6; color:#b2182b; }}

 /* ---- Bracket (HTML, #7) ---- */
 .bx-scroll {{ overflow-x:auto; padding-bottom:6px; }}
 .bx {{ display:flex; gap:14px; min-width:max-content; align-items:stretch; }}
 .bx-col {{ display:flex; flex-direction:column; justify-content:space-around;
   gap:8px; min-width:150px; }}
 .bx-rnd {{ text-align:center; font-size:.7rem; font-weight:800;
   text-transform:uppercase; letter-spacing:.05em; color:var(--head);
   margin-bottom:2px; }}
 .bx-tie {{ background:var(--head-soft); border:1px solid var(--line);
   border-radius:9px; overflow:hidden; }}
 .bx-team {{ display:flex; justify-content:space-between; gap:8px;
   padding:5px 9px; font-size:12px; color:var(--muted);
   border-bottom:1px solid var(--line); }}
 .bx-team:last-child {{ border-bottom:none; }}
 .bx-team.bx-w {{ color:var(--ink); font-weight:700;
   background:color-mix(in srgb,var(--win-c) 12%,transparent); }}
 .bx-champ {{ justify-content:center; }}
 .bx-trophy {{ text-align:center; font-weight:800; color:var(--head);
   font-size:1.05rem; background:color-mix(in srgb,var(--gold) 18%,transparent);
   border:1px solid var(--line); border-radius:10px; padding:14px 10px; }}

 /* ---- Group stage cards ---- */
 .ggrid {{ display:grid; gap:16px;
   grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); }}
 .gcard {{ border:1px solid var(--line); border-radius:14px; overflow:hidden;
   background:var(--card); box-shadow:0 1px 2px rgba(16,40,28,.04); }}
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
 .gtable td.g-team .form {{ margin-left:6px; }}
 .gtable tr.g-q {{ background:color-mix(in srgb,var(--head) 7%,transparent); }}
 .gtable tr.g-q td.g-team {{ color:var(--head); }}
 .g-num {{ width:33%; }}
 .gbar {{ position:relative; height:18px; border-radius:5px; background:var(--head-soft);
   overflow:hidden; }}
 .gbar > i {{ position:absolute; left:0; top:0; bottom:0;
   background:color-mix(in srgb,var(--head-2) 45%,transparent); }}
 .gbar.gw > i {{ background:color-mix(in srgb,var(--gold) 55%,transparent); }}
 .gbar > b {{ position:absolute; inset:0; display:flex; align-items:center;
   justify-content:center; font-size:.72rem; font-weight:700; color:var(--ink);
   font-style:normal; }}
 .gf-sub {{ font-size:.66rem; text-transform:uppercase; letter-spacing:.05em;
   color:var(--muted); font-weight:800; padding:10px 14px 2px; }}
 .gf-list {{ list-style:none; margin:0; padding:2px 14px 14px; }}
 .gf-list li {{ padding:7px 0; border-bottom:1px solid var(--line); }}
 .gf-list li:last-child {{ border-bottom:none; }}
 .gf-row {{ display:flex; align-items:center; justify-content:space-between;
   gap:8px; font-size:12.5px; }}
 .gf-row .gf-h {{ font-weight:600; text-align:right; flex:1; }}
 .gf-row .gf-a {{ font-weight:600; flex:1; }}
 .gf-xg {{ color:var(--head); font-size:.72rem; background:var(--head-soft);
   border-radius:5px; padding:2px 7px; white-space:nowrap; font-weight:700; }}
 .gf-chip {{ margin-top:5px; }}
 .gf-done {{ color:var(--muted); padding:8px 14px 14px; font-size:.84rem;
   font-style:italic; }}

 footer {{ color:var(--muted); font-size:12px; text-align:center; padding:26px 16px; }}
 footer code {{ background:var(--head-soft); padding:1px 5px; border-radius:4px; }}
 @media (max-width:560px) {{
   header h1 {{ font-size:1.5rem; }}
   .nm-teams {{ font-size:1.2rem; }}
 }}
</style></head><body>
<script>document.documentElement.className += ' js';</script>
<header>
 <button class="theme-btn" id="themeBtn" type="button" aria-label="Toggle theme">🌙 Dark</button>
 <div class="hero-wrap">
  <p class="eyebrow">FIFA World Cup 2026 &middot; Mexico &middot; USA &middot; Canada</p>
  <h1>World Cup 2026 &mdash; AI Match Predictions</h1>
  <p class="sub">Predicted champion:
   <span class="badge">{bracket.champion} &middot; {champ_prob:.0%}</span>
   &nbsp;&middot;&nbsp; <span id="updated" data-iso="{updated_iso}"
     title="{updated}">updated {updated}</span></p>
  {_hero_stats_html(bracket.champion, champ_prob, track_stats, next_match)}
 </div>
</header>
<nav><div class="nav-wrap">
 <a href="#next" class="tab" data-tab="next">Next match</a>
 <a href="#title" class="tab" data-tab="title">Title odds</a>
 <a href="#bracket" class="tab" data-tab="bracket">Bracket</a>
 <a href="#upcoming" class="tab" data-tab="upcoming">Upcoming fixtures</a>
 <a href="#groups" class="tab" data-tab="groups">Group forecast</a>
 <a href="#track" class="tab" data-tab="track">Track record</a>
 <a href="#performance" class="tab" data-tab="performance">Model performance</a>
 <a href="#rounds" class="tab" data-tab="rounds">Round odds</a>
 <a href="#ko" class="tab" data-tab="ko">Knockout games</a>
</div></nav>
<main>

<section id="next">
 <h2>Next match</h2>
 <p class="lead">The next fixture to kick off, with the model's live call.</p>
 {_next_match_html(next_match)}
</section>

<section id="title">
 <h2>Title probability</h2>
 <p class="lead">Each team's chance of lifting the trophy.</p>
 {_img_tag(p_bar)}
</section>

<section id="bracket">
 <h2>Expected knockout bracket</h2>
 <p class="lead">The most-likely path to the final, with the win probability for
  every tie.</p>
 {_bracket_html(bracket)}
</section>

<section id="upcoming">
 <h2>Upcoming fixtures &amp; live predictions</h2>
 <p class="lead">Every upcoming fixture in EAT, with the model's live
  win/draw/loss call and expected score.</p>
 <div class="tbl-scroll">{_upcoming_html(upcoming)}</div>
</section>

<section id="groups">
 <h2>Live group-stage forecast</h2>
 <p class="lead">Remaining fixtures, recent form, and every team's chance of
  advancing &mdash; form blended with long-run strength.</p>
 {_group_forecast_html(group_forecasts, form_map)}
</section>

<section id="track">
 <h2>Model track record</h2>
 <p class="lead">Each pre-kickoff prediction (no look-ahead) scored against the
  actual result, since the model went live on {PREDICTIONS_START:%b %d, %Y}.</p>
 <div class="tbl-scroll">{_track_record_html(track_log, track_stats)}</div>
</section>

<section id="performance">
 <h2>Model performance &amp; calibration</h2>
 <p class="lead">Probabilistic-quality metrics on the locked predictions: RPS,
  Brier, log loss, calibration error, and how well each outcome is called.</p>
 {_performance_html(track_log, p_cal) if p_cal is not None
   else "<p style='color:var(--muted)'>Calibration populates once enough "
        "matches are scored.</p>"}
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
<script src="https://cdn.jsdelivr.net/npm/twemoji@14.0.2/dist/twemoji.min.js" crossorigin="anonymous"></script>
<script>
(function() {{
  // ---- Flags: render as consistent SVG on every OS (incl. Windows) (#3) ----
  try {{
    if (window.twemoji) twemoji.parse(document.body, {{ folder: 'svg', ext: '.svg' }});
  }} catch (e) {{}}

  // ---- Theme toggle (#5) ----
  var root = document.documentElement, btn = document.getElementById('themeBtn');
  function setTheme(t) {{
    root.setAttribute('data-theme', t);
    try {{ localStorage.setItem('wc26-theme', t); }} catch (e) {{}}
    if (btn) btn.textContent = t === 'dark' ? '☀️ Light' : '🌙 Dark';
  }}
  var saved = null;
  try {{ saved = localStorage.getItem('wc26-theme'); }} catch (e) {{}}
  if (!saved && window.matchMedia &&
      window.matchMedia('(prefers-color-scheme: dark)').matches) saved = 'dark';
  setTheme(saved || 'light');
  if (btn) btn.addEventListener('click', function() {{
    setTheme(root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
  }});

  // ---- Relative 'updated' time (#9) ----
  var up = document.getElementById('updated');
  if (up && up.dataset.iso) {{
    var then = new Date(up.dataset.iso), mins = Math.round((Date.now() - then) / 60000);
    var rel = mins < 1 ? 'just now' : mins < 60 ? mins + 'm ago'
      : mins < 1440 ? Math.round(mins / 60) + 'h ago'
      : Math.round(mins / 1440) + 'd ago';
    up.textContent = 'updated ' + rel;
  }}

  // ---- Next-match countdown (#6) ----
  var cd = document.querySelector('.nm-countdown');
  if (cd && cd.dataset.kickoff) {{
    var ko = new Date(cd.dataset.kickoff);
    var tick = function() {{
      var ms = ko - Date.now();
      if (ms <= 0) {{ cd.textContent = '\\u23F1 Kicking off now'; return; }}
      var h = Math.floor(ms / 3600000), m = Math.floor((ms % 3600000) / 60000),
          s = Math.floor((ms % 60000) / 1000);
      cd.textContent = '\\u23F3 Kicks off in ' +
        (h > 0 ? h + 'h ' : '') + m + 'm ' + s + 's';
    }};
    tick(); setInterval(tick, 1000);
  }}

  // ---- Tabbed pagination ----
  // Each nav item maps to one section; only the active section is shown, so the
  // page reads as discrete tabs instead of one long scroll. A prev/next pager
  // and the browser hash keep navigation easy and deep-linkable.
  var tabs = Array.prototype.slice.call(document.querySelectorAll('nav a.tab'));
  var ids = tabs.map(function(t) {{ return t.dataset.tab; }});
  var labels = tabs.map(function(t) {{ return t.textContent.trim(); }});

  // Build the prev/next pager and append it to <main>.
  var main = document.querySelector('main');
  var pager = document.createElement('div');
  pager.className = 'pager';
  pager.innerHTML =
    '<button type="button" id="pgPrev">&#8592; Previous</button>' +
    '<span class="pg-count" id="pgCount"></span>' +
    '<button type="button" id="pgNext">Next &#8594;</button>';
  if (main) main.appendChild(pager);
  var prevBtn = document.getElementById('pgPrev'),
      nextBtn = document.getElementById('pgNext'),
      count = document.getElementById('pgCount');

  var current = 0;
  function activate(i, push) {{
    if (i < 0 || i >= ids.length) return;
    current = i;
    tabs.forEach(function(t, k) {{ t.classList.toggle('active', k === i); }});
    ids.forEach(function(id, k) {{
      var s = document.getElementById(id);
      if (s) s.classList.toggle('active', k === i);
    }});
    if (prevBtn) prevBtn.disabled = (i === 0);
    if (nextBtn) {{
      nextBtn.disabled = (i === ids.length - 1);
      nextBtn.innerHTML = (i === ids.length - 1)
        ? 'Next &#8594;'
        : labels[i + 1] + ' &#8594;';
    }}
    if (prevBtn) prevBtn.innerHTML = (i === 0)
      ? '&#8592; Previous' : '&#8592; ' + labels[i - 1];
    if (count) count.textContent = 'Section ' + (i + 1) + ' of ' + ids.length;
    if (push && history.replaceState) history.replaceState(null, '', '#' + ids[i]);
    window.scrollTo({{ top: 0, behavior: 'smooth' }});
  }}

  tabs.forEach(function(t, i) {{
    t.addEventListener('click', function(e) {{ e.preventDefault(); activate(i, true); }});
  }});
  if (prevBtn) prevBtn.addEventListener('click', function() {{ activate(current - 1, true); }});
  if (nextBtn) nextBtn.addEventListener('click', function() {{ activate(current + 1, true); }});
  window.addEventListener('hashchange', function() {{
    var j = ids.indexOf(location.hash.replace('#', ''));
    if (j >= 0 && j !== current) activate(j, false);
  }});

  // Start on the section named in the URL hash, else the first tab.
  var start = ids.indexOf(location.hash.replace('#', ''));
  activate(start >= 0 ? start : 0, false);
}})();
</script>
</body></html>"""
    (REPORTS / "dashboard.html").write_text(html, encoding="utf-8")

    print(f"\nDone. Open: {REPORTS / 'dashboard.html'}")
    print(f"  bracket : {p_bracket}")
    print(f"  bar     : {p_bar}")
    print(f"  heatmap : {p_heat}")
    print(f"  games   : {REPORTS / 'games.csv'}")


if __name__ == "__main__":
    main()
