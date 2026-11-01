"""
Downloads real NBA game logs with nba_api and writes the two files that
build_training_csv.py expects:

    team_games.csv    game_id, season, date, team, home, pts, opp_pts, poss
    player_games.csv  date, team, player, min

Usage:
    pip install nba_api pandas
    python fetch_nba_data.py                                  # default seasons
    python fetch_nba_data.py --seasons 2022-23 2023-24 2024-25 2025-26
    python fetch_nba_data.py --out C:\\hoops_data --playoffs

Then:
    python build_training_csv.py team_games.csv player_games.csv

Raw downloads are cached in <out>/raw/, so re-running only fetches what is missing
(delete a cached file to force a re-download).
"""
import argparse, sys, time
from pathlib import Path
import pandas as pd

DEFAULT_SEASONS = ["2023-24", "2024-25", "2025-26"]
PAUSE_SECONDS = 2.0       # be polite; stats.nba.com throttles rapid requests
RETRIES = 4


def fetch_season(season: str, kind: str, season_type: str) -> pd.DataFrame:
    """kind: 'T' (team logs) or 'P' (player logs). Network call, with retries."""
    from nba_api.stats.endpoints import leaguegamelog
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            df = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation=kind,
                timeout=60,
            ).get_data_frames()[0]
            time.sleep(PAUSE_SECONDS)
            return df
        except Exception as e:                       # timeouts / throttling are common
            last_err = e
            wait = 5 * attempt
            print(f"  attempt {attempt}/{RETRIES} failed ({type(e).__name__}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Could not download {season} ({kind}): {last_err}")


def transform_team(df: pd.DataFrame, season: str) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    df["home"] = df["MATCHUP"].str.contains(" vs. ", regex=False).astype(int)   # "BOS vs. NYK" = home, "BOS @ NYK" = away
    # Estimated possessions for each team's box score
    df["poss_est"] = df["FGA"] + 0.44 * df["FTA"] - df["OREB"] + df["TOV"]

    # Pair the two teams of each game to get opponent points and a shared possession count
    g = df.groupby("GAME_ID")
    df["n"] = g["TEAM_ID"].transform("count")
    df = df[df["n"] == 2].copy()                    # drop any game missing a side
    df["opp_pts"] = df.groupby("GAME_ID")["PTS"].transform("sum") - df["PTS"]
    df["poss"] = df.groupby("GAME_ID")["poss_est"].transform("mean")   # both teams share the game's possessions

    out = pd.DataFrame({
        "game_id": df["GAME_ID"], "season": season, "date": df["date"],
        "team": df["TEAM_ABBREVIATION"], "home": df["home"],
        "pts": df["PTS"].astype(int), "opp_pts": df["opp_pts"].astype(int),
        "poss": df["poss"].round(2),
    })
    return out[out["poss"] > 0]


def transform_player(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["min"] = pd.to_numeric(df["MIN"], errors="coerce")
    df = df[df["min"] > 0]                          # only players who actually played
    return pd.DataFrame({
        "date": pd.to_datetime(df["GAME_DATE"]).dt.strftime("%Y-%m-%d"),
        "team": df["TEAM_ABBREVIATION"], "player": df["PLAYER_NAME"], "min": df["min"],
    })


def get_cached(raw_dir: Path, season: str, kind: str, season_type: str) -> pd.DataFrame:
    tag = season_type.replace(" ", "")
    path = raw_dir / f"{season}_{kind}_{tag}.csv"
    if path.exists():
        print(f"  using cached {path.name}")
        return pd.read_csv(path, dtype={"GAME_ID": str})
    print(f"  downloading {season} {'team' if kind == 'T' else 'player'} logs ({season_type})...")
    df = fetch_season(season, kind, season_type)
    df.to_csv(path, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS, help="e.g. 2023-24 2024-25")
    ap.add_argument("--out", default=".", help="output folder")
    ap.add_argument("--playoffs", action="store_true", help="also include playoff games")
    ap.add_argument("--no-players", action="store_true", help="skip player logs (no injury feature)")
    a = ap.parse_args()

    out = Path(a.out); raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    types = ["Regular Season"] + (["Playoffs"] if a.playoffs else [])

    teams, players = [], []
    for season in a.seasons:
        print(f"Season {season}")
        for st in types:
            t = get_cached(raw, season, "T", st)
            if len(t):
                teams.append(transform_team(t, season))
            if not a.no_players:
                p = get_cached(raw, season, "P", st)
                if len(p):
                    players.append(transform_player(p))

    if not teams:
        sys.exit("No team data downloaded.")
    tg = pd.concat(teams).sort_values(["date", "game_id"])
    tg.to_csv(out / "team_games.csv", index=False)
    print(f"Wrote {len(tg)} team-game rows ({tg['game_id'].nunique()} games) to {out / 'team_games.csv'}")
    if players:
        pg = pd.concat(players).sort_values(["date", "team"])
        pg.to_csv(out / "player_games.csv", index=False)
        print(f"Wrote {len(pg)} player-game rows to {out / 'player_games.csv'}")
    print("\nNext: python build_training_csv.py team_games.csv player_games.csv")


if __name__ == "__main__":
    main()
