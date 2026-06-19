"""Visualization stack for the 2026 World Cup predictions.

Three views, all saved as PNGs into reports/:
  * bracket diagram  -- the modal knockout bracket from R16 to the Final, each
    tie annotated with the model's win probability and the predicted winner
    highlighted; shows the expected path to the title.
  * champion bar     -- each team's probability of winning the tournament.
  * round heatmap    -- a team x round grid of "reach this round" probabilities,
    the compact summary of the whole Monte Carlo run.

These turn the numeric outputs into the at-a-glance picture of the tournament
the user asked for.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless rendering to file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.simulation.bracket import ExpectedBracket, Game

# A simple, readable palette.
WIN_COLOR = "#1b7837"
LOSE_COLOR = "#b2182b"
BOX_FACE = "#f5f5f5"
EDGE = "#333333"


def _short(name: str, n: int = 14) -> str:
    return name if len(name) <= n else name[: n - 1] + "."


# --------------------------------------------------------------------------
# Bracket diagram
# --------------------------------------------------------------------------
def plot_bracket(bracket: ExpectedBracket, out: Path,
                 title: str = "2026 World Cup — Expected Knockout Bracket") -> Path:
    """Draw the bracket round-by-round with per-game win probabilities."""
    rounds = list(bracket.games_by_round.keys())
    n_rounds = len(rounds)
    if n_rounds == 0:
        raise ValueError("Bracket has no games to plot.")

    max_games = max(len(g) for g in bracket.games_by_round.values())
    fig_h = max(6.0, max_games * 0.9)
    fig_w = max(10.0, n_rounds * 3.2 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=16)

    box_w, box_h = 2.5, 0.62
    col_gap = 3.1

    # Vertical positions: round r has its games centred and spread out.
    positions = {}  # (round_index, game_index) -> y-centre
    for ri, rnd in enumerate(rounds):
        games = bracket.games_by_round[rnd]
        span = max_games
        step = span / (len(games) + 1)
        for gi in range(len(games)):
            positions[(ri, gi)] = span - (gi + 1) * step

    for ri, rnd in enumerate(rounds):
        games = bracket.games_by_round[rnd]
        x = ri * col_gap
        # Round header.
        ax.text(x + box_w / 2, max_games + 0.3, rnd, ha="center",
                fontsize=11, fontweight="bold", color=EDGE)
        for gi, g in enumerate(games):
            y = positions[(ri, gi)]
            _draw_game_box(ax, x, y, box_w, box_h, g)
            # Connector to the next round's game (every two feed one).
            if ri + 1 < n_rounds:
                ny = positions[(ri + 1, gi // 2)]
                x2 = (ri + 1) * col_gap
                ax.plot([x + box_w, x2], [y, ny], color="#bbbbbb",
                        lw=1.0, zorder=1)

    # Champion banner.
    cx = n_rounds * col_gap
    cy = max_games / 2
    ax.text(cx + 0.2, cy, f"WINNER\n{bracket.champion}", fontsize=14,
            fontweight="bold", color=WIN_COLOR, va="center", ha="left")

    ax.set_xlim(-0.5, cx + 3)
    ax.set_ylim(-0.5, max_games + 1)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _draw_game_box(ax, x, y, w, h, g: Game) -> None:
    """Render one tie as two stacked team rows with the winner highlighted."""
    rows = [(g.home, g.p_home), (g.away, 1.0 - g.p_home)]
    for i, (team, p) in enumerate(rows):
        ry = y + (h / 2) - i * (h / 2)
        is_win = (team == g.winner)
        ax.add_patch(plt.Rectangle((x, ry - h / 4), w, h / 2,
                                   facecolor=BOX_FACE,
                                   edgecolor=EDGE, lw=0.8, zorder=2))
        ax.text(x + 0.08, ry, _short(team), va="center", ha="left",
                fontsize=8.5, fontweight="bold" if is_win else "normal",
                color=WIN_COLOR if is_win else "#444444", zorder=3)
        ax.text(x + w - 0.08, ry, f"{p:.0%}", va="center", ha="right",
                fontsize=8, color=WIN_COLOR if is_win else "#888888", zorder=3)


# --------------------------------------------------------------------------
# Champion bar chart
# --------------------------------------------------------------------------
def plot_champion_bar(pred: pd.DataFrame, out: Path, top: int = 15,
                      champ_col: str = "P_Champion") -> Path:
    """Horizontal bar chart of title-winning probability for the top teams."""
    d = pred.sort_values(champ_col, ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, top * 0.4)))
    bars = ax.barh(d["team"], d[champ_col] * 100, color="#2166ac")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    ax.set_xlabel("Probability of winning the tournament (%)")
    ax.set_title("2026 World Cup — Title Probability", fontweight="bold")
    ax.margins(x=0.12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# Round-reach heatmap
# --------------------------------------------------------------------------
def plot_round_heatmap(pred: pd.DataFrame, out: Path, top: int = 20) -> Path:
    """Team x round grid of reach-probabilities."""
    cols = [c for c in pred.columns if c.startswith("P_")]
    champ_col = cols[-1]
    d = pred.sort_values(champ_col, ascending=False).head(top)
    data = d[cols].to_numpy() * 100

    fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 2, 0.45 * len(d) + 1.5))
    im = ax.imshow(data, cmap="YlGnBu", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("P_", "") for c in cols])
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["team"])
    for i in range(len(d)):
        for j in range(len(cols)):
            v = data[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=7, color="black" if v < 60 else "white")
    ax.set_title("Probability of reaching each round (%)", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
