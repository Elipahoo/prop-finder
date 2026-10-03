"""
NFL prop edge finder: ESPN game logs + FanDuel lines (via The Odds API).

For each FanDuel player prop it:
  1. pulls the player's ESPN game log (last few seasons),
  2. blends recent form with the player's history vs. THIS opponent,
  3. converts the projection to P(over) and compares to FanDuel's de-vigged odds,
  4. ranks the biggest gaps (edges).

Setup:   pip install requests
         export ODDS_API_KEY=your_key      (https://the-odds-api.com)
Run:     python prop_edge_finder.py --min-edge 0.05

NOTE: ESPN endpoints are unofficial and can change. If parsing breaks, run with
--debug to print the raw labels/keys and adjust the parsing functions.
"""
import argparse, functools, math, os, statistics as st
import requests

ODDS_KEY = os.environ.get("ODDS_API_KEY")
ODDS = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
ESPN_SEARCH = "https://site.web.api.espn.com/apis/common/v3/search/v3"
ESPN_ATHLETE = "https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{id}"
ESPN_GAMELOG = ESPN_ATHLETE + "/gamelog"
ESPN_TEAMS = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"

# market -> (anchor label that starts the stat group, target label inside the group)
# ESPN reuses labels like "YDS" across passing/rushing/receiving, so we anchor.
MARKETS = {
    "player_pass_yds": ("CMP", "YDS"),
    "player_pass_tds": ("CMP", "TD"),
    "player_rush_yds": ("CAR", "YDS"),
    "player_reception_yds": ("REC", "YDS"),
    "player_receptions": ("REC", "REC"),
}
SEASONS_BACK = 4
RECENT_N = 10         # games used for "recent form"
DECAY = 0.8           # each older game counts 80% as much as the one after it
OPP_PRIOR_K = 12      # higher = trust opponent history less (6 games vs a team ~ 33% weight)


def get(url, **params):
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


# ---------- ESPN ----------
def norm(name):
    """Lowercase, drop punctuation and Jr/III-type suffixes so names match across sites."""
    n = name.lower().replace(".", "").replace("'", "").replace("-", " ")
    return " ".join(p for p in n.split() if p not in ("jr", "sr", "ii", "iii", "iv", "v"))


def build_roster_index():
    """Returns (team name -> abbreviation, normalized player name -> [(espn_id, team name)])."""
    teams = get(ESPN_TEAMS)["sports"][0]["leagues"][0]["teams"]
    abbr, index = {}, {}
    for t in teams:
        t = t["team"]
        abbr[t["displayName"]] = t["abbreviation"]
        roster = get(f"{ESPN_TEAMS}/{t['id']}/roster")
        for group in roster.get("athletes", []):
            for p in group.get("items", []):
                index.setdefault(norm(p["fullName"]), []).append((p["id"], t["displayName"]))
    return abbr, index


@functools.lru_cache(maxsize=None)
def gamelog_raw(pid, season):
    try:
        return get(ESPN_GAMELOG.format(id=pid), season=season)
    except requests.HTTPError:
        return None


def to_num(s):
    try:
        return float(str(s).replace(",", "").split("/")[0])
    except ValueError:
        return None


def game_log(pid, market, debug=False):
    """Return list of (date, opp_abbr, value) oldest -> newest."""
    anchor, target = MARKETS[market]
    year = 2026  # adjust season logic as needed (NFL season = start year)
    rows = []
    for season in range(year - SEASONS_BACK + 1, year + 1):
        d = gamelog_raw(pid, season)
        if not d:
            continue
        labels = d.get("labels", [])
        if debug:
            print("labels:", labels)
        if anchor not in labels:
            continue
        i = labels.index(anchor)
        j = labels.index(target, i)
        meta = d.get("events", {})
        for stype in d.get("seasonTypes", []):
            if "preseason" in stype.get("displayName", "").lower():
                continue  # starters barely play in preseason; it skews recent form
            for cat in stype.get("categories", []):
                for e in cat.get("events", []):
                    m = meta.get(e["eventId"])
                    v = to_num(e["stats"][j]) if m else None
                    if v is not None:
                        rows.append((m["gameDate"], m["opponent"]["abbreviation"], v))
    return sorted(rows)


# ---------- model ----------
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


COUNT_MARKETS = {"player_receptions", "player_pass_tds"}


def poisson_cdf(k, mu):
    """P(X <= k) for a count stat with average mu."""
    mu = max(mu, 0.05)
    return sum(math.exp(-mu) * mu ** i / math.factorial(i) for i in range(int(k) + 1))


def defender_adjustment(pid, opp_abbr):
    """HOOK: return a multiplier (e.g. 0.93 = tough matchup) from defender-level data
    (PFF/NGS shadow coverage, CB-WR matchups, pass-rush win rate). Not in ESPN."""
    return 1.0


def project(rows, opp):
    vals = [v for _, _, v in rows]
    if len(vals) < 5:
        return None
    recent = vals[-RECENT_N:][::-1]  # newest game first
    weights = [DECAY ** i for i in range(len(recent))]
    mu_recent = sum(x * w for x, w in zip(recent, weights)) / sum(weights)
    vs_opp = [v for _, o, v in rows if o == opp]
    n = len(vs_opp)
    mu_opp = st.mean(vs_opp) if n else mu_recent
    # shrink opponent history toward recent form; few games = mostly recent form
    mu = (n * mu_opp + OPP_PRIOR_K * mu_recent) / (n + OPP_PRIOR_K)
    sd = max(st.pstdev(vals[-16:]), 0.15 * mu, 0.5)
    return mu, sd, n


def implied(american):
    return 100 / (american + 100) if american > 0 else -american / (-american + 100)


def payout(american):
    return american / 100 if american > 0 else 100 / -american


# ---------- odds ----------
def fanduel_props(markets):
    events = get(f"{ODDS}/events", apiKey=ODDS_KEY)
    for ev in events:
        d = get(f"{ODDS}/events/{ev['id']}/odds", apiKey=ODDS_KEY, regions="us",
                bookmakers="fanduel", markets=",".join(markets), oddsFormat="american")
        for bk in d.get("bookmakers", []):
            for mk in bk["markets"]:
                lines = {}
                for o in mk["outcomes"]:
                    lines.setdefault((o["description"], o["point"]), {})[o["name"]] = o["price"]
                for (player, point), sides in lines.items():
                    if "Over" in sides and "Under" in sides:
                        yield ev, mk["key"], player, point, sides


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-edge", type=float, default=0.05)
    ap.add_argument("--max-gap", type=float, default=0.35,
                    help="skip props where projection differs from the line by more than this "
                         "fraction (usually means injury/role change the model can't see)")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    if not ODDS_KEY:
        raise SystemExit("Set ODDS_API_KEY first.")

    print("Loading NFL rosters from ESPN...")
    abbr, index = build_roster_index()
    results, missed, skipped = [], set(), 0
    print("Loading FanDuel props and player history (this can take a few minutes)...")
    for ev, market, player, line, px in fanduel_props(MARKETS):
        try:
            # match by name AND by being on one of the two teams in this game
            cands = [c for c in index.get(norm(player), [])
                     if c[1] in (ev["home_team"], ev["away_team"])]
            if not cands:
                missed.add(player)
                continue
            pid, team = cands[0]
            opp_name = ev["away_team"] if team == ev["home_team"] else ev["home_team"]
            opp = abbr.get(opp_name)
            proj = project(game_log(pid, market, a.debug), opp)
            if not proj:
                continue
        except Exception as e:  # keep going on bad lookups
            if a.debug:
                print("skip", player, e)
            continue

        mu, sd, n_opp = proj
        mu *= defender_adjustment(pid, opp)
        # A huge gap between the line and our projection almost always means the book
        # knows about an injury / backup starting / role change. Don't treat it as an edge.
        if abs(mu - line) / max(line, 1.0) > a.max_gap:
            skipped += 1
            continue
        if market in COUNT_MARKETS:
            p_over = 1 - poisson_cdf(int(line), mu)
        else:
            p_over = 1 - norm_cdf((line - mu) / sd)
        io, iu = implied(px["Over"]), implied(px["Under"])
        fair_over = io / (io + iu)  # remove the vig
        for side, p_model, fair, price in (("Over", p_over, fair_over, px["Over"]),
                                           ("Under", 1 - p_over, 1 - fair_over, px["Under"])):
            edge = p_model - fair
            ev_per_dollar = p_model * payout(price) - (1 - p_model)
            if edge >= a.min_edge:
                results.append((edge, ev_per_dollar, player, market, side, line, price, mu, n_opp, opp))

    results.sort(reverse=True)
    print(f"{'edge':>6} {'EV/$':>6}  player | market | pick line (odds) | proj | games vs opp")
    for edge, ev_, pl, mk, side, line, price, mu, n, opp in results:
        flag = "  <-- CHECK INJURY/ROLE NEWS" if edge > 0.20 else ""
        print(f"{edge:6.1%} {ev_:6.2f}  {pl} | {mk} | {side} {line} ({price:+d}) | {mu:.1f} | {n} vs {opp}{flag}")
    print(f"\n{len(results)} edges found. {skipped} props skipped (line too far from projection, "
          f"likely injury/role change). {len(missed)} players couldn't be matched to an ESPN roster.")
    if a.debug and missed:
        print("Unmatched:", ", ".join(sorted(missed)))


if __name__ == "__main__":
    main()
