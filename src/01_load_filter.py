"""
src/01_load_filter.py
Script 1: Ingest raw Cricsheet IPL match CSVs, apply filtering criteria,
extract chase-level metadata, cross-check targets, and save chases.csv.
"""
import csv
from pathlib import Path
from collections import Counter
import pandas as pd
import config

def parse_match(csv_path: Path):
    """
    Parses a single Cricsheet match CSV file.
    Returns:
        (status, match_data, drop_reason)
    """
    match_id = csv_path.stem
    info_dict = {}
    winner = None
    outcome = None
    method = None
    date_str = None
    venue_str = None
    season_raw = None
    target_runs_info = None

    innings_1_balls = []
    innings_2_balls = []

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            row_type = row[0]
            if row_type == "info":
                key = row[1]
                if key == "winner" and len(row) > 2:
                    winner = row[2]
                elif key == "outcome" and len(row) > 2:
                    outcome = row[2]
                elif key == "method" and len(row) > 2:
                    method = row[2]
                elif key == "date" and len(row) > 2:
                    if date_str is None:
                        date_str = row[2]
                elif key == "venue" and len(row) > 2:
                    venue_str = row[2]
                elif key == "season" and len(row) > 2:
                    season_raw = row[2]
                elif key == "target_runs" and len(row) > 3:
                    try:
                        target_runs_info = int(row[3])
                    except ValueError:
                        pass
                info_dict[key] = row[2:]
            elif row_type == "ball":
                inn = row[1]
                if inn == "1":
                    innings_1_balls.append(row)
                elif inn == "2":
                    innings_2_balls.append(row)

    # Filtering Criteria:
    # 1. Must have info,winner
    if not winner:
        return "DROP", None, "missing_winner"

    # 2. Must NOT have info,outcome (e.g. no result, tie)
    if outcome is not None:
        return "DROP", None, f"abnormal_outcome_{outcome}"

    # 3. Must NOT have info,method (e.g. D/L)
    if method is not None:
        return "DROP", None, f"rain_rule_{method}"

    # 4. Must have deliveries in both innings 1 and innings 2
    if len(innings_1_balls) == 0:
        return "DROP", None, "no_innings_1_deliveries"
    if len(innings_2_balls) == 0:
        return "DROP", None, "no_innings_2_deliveries"

    # Season parsing: using date year (e.g. 2008 for 2007/08 season) to represent calendar tournament year
    if date_str:
        season_year = int(date_str.split("/")[0].split("-")[0])
    elif season_raw:
        season_year = int(season_raw.split("/")[0])
    else:
        season_year = None

    # Chasing team is batting team of innings 2
    chasing_team = innings_2_balls[0][3]

    # Calculate innings 1 total runs: runs_off_bat + extras
    inn1_total = sum(int(b[7]) + int(b[8]) for b in innings_1_balls)
    expected_target = inn1_total + 1

    target = target_runs_info if target_runs_info is not None else expected_target
    target_mismatch = (target_runs_info is not None and target_runs_info != expected_target)

    chase_won = 1 if (winner == chasing_team) else 0

    match_record = {
        "match_id": match_id,
        "season": season_year,
        "date": date_str,
        "venue": venue_str,
        "chasing_team": chasing_team,
        "target": target,
        "expected_target": expected_target,
        "target_mismatch": target_mismatch,
        "inn1_runs": inn1_total,
        "winner": winner,
        "chase_won": chase_won,
    }

    return "KEEP", match_record, None

def main():
    print("=" * 70)
    print("SCRIPT 1: Ingesting and Filtering IPL Match CSVs")
    print("=" * 70)

    raw_files = sorted(config.RAW_DATA_DIR.glob("*.csv"))
    total_files = len(raw_files)
    print(f"Total CSV files found in {config.RAW_DATA_DIR}: {total_files}")

    kept_matches = []
    drop_reasons = Counter()
    mismatches = []

    for fpath in raw_files:
        status, data, reason = parse_match(fpath)
        if status == "KEEP":
            kept_matches.append(data)
            if data["target_mismatch"]:
                mismatches.append(data)
        else:
            drop_reasons[reason] += 1

    df_chases = pd.DataFrame(kept_matches)

    # Save to data/processed/chases.csv
    out_path = config.PROCESSED_DATA_DIR / "chases.csv"
    df_chases.to_csv(out_path, index=False)
    print(f"\nSaved filtered chases to: {out_path}")

    # Summary Report
    print("\n--- Summary Statistics ---")
    print(f"Files read: {total_files}")
    print(f"Files kept: {len(df_chases)} ({len(df_chases)/total_files*100:.1f}%)")
    print(f"Files dropped: {sum(drop_reasons.values())}")
    for reason, count in sorted(drop_reasons.items(), key=lambda x: -x[1]):
        print(f"  - {reason}: {count}")

    print("\n--- Target Cross-Check ---")
    if mismatches:
        print(f"Found {len(mismatches)} target mismatches between info.target_runs and (inn1_total + 1):")
        for m in mismatches[:5]:
            print(f"  Match {m['match_id']} ({m['season']}): target_runs={m['target']}, inn1_total+1={m['expected_target']}")
        if len(mismatches) > 5:
            print(f"  ... and {len(mismatches) - 5} more.")
    else:
        print("All kept matches had target == inn1_total + 1 (100% match).")

    print("\n--- Matches Per Season ---")
    season_counts = df_chases["season"].value_counts().sort_index()
    for s, c in season_counts.items():
        chase_wins = df_chases[df_chases["season"] == s]["chase_won"].sum()
        win_rate = chase_wins / c * 100
        print(f"  Season {s}: {c:3d} matches | Chase Won: {chase_wins:2d} ({win_rate:5.1f}%)")

    overall_wins = df_chases["chase_won"].sum()
    overall_win_rate = overall_wins / len(df_chases) * 100
    print(f"\nOverall Chase Win Rate: {overall_wins}/{len(df_chases)} ({overall_win_rate:.2f}%)")
    print("=" * 70)

if __name__ == "__main__":
    main()
