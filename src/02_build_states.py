"""
src/02_build_states.py
Script 2: Replay 2nd innings ball-by-ball, compute over-by-over match state
(wickets bucket, pressure bucket), construct transitions table and over_states.csv.
"""
import csv
from pathlib import Path
import pandas as pd
import numpy as np
import config

def replay_chase(match_row, raw_dir: Path):
    """
    Replays the 2nd innings ball-by-ball for a single match.
    Returns:
        over_records: list of dicts for each completed over + final absorption row
        transitions: list of dicts (from_state, to_state, over_from)
        trace_summary: dict of trace information for sanity checking
    """
    match_id = str(match_row["match_id"])
    season = int(match_row["season"])
    target = int(match_row["target"])
    chase_won = int(match_row["chase_won"])
    csv_path = raw_dir / f"{match_id}.csv"

    innings_2_balls = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0] == "ball" and row[1] == "2":
                innings_2_balls.append(row)

    runs = 0
    wickets = 0
    legal_balls = 0
    over_records = []
    transitions = []

    absorbed = False
    absorbing_state = None
    absorption_over = None
    absorption_balls_left = None
    absorption_runs_needed = None

    last_transient_state = None
    last_transient_over = 0

    # Ball by ball replay
    for ball_idx, row in enumerate(innings_2_balls):
        # Delivery legality: wides (row[9]) and noballs (row[10]) do NOT count as legal balls
        is_wide = bool(len(row) > 9 and row[9])
        is_noball = bool(len(row) > 10 and row[10])
        is_legal = (not is_wide) and (not is_noball)

        # Runs on this ball: runs_off_bat + extras
        runs_off_bat = int(row[7])
        extras = int(row[8])
        ball_runs = runs_off_bat + extras
        runs += ball_runs

        if is_legal:
            legal_balls += 1

        # Wicket dismissal
        is_wicket = bool(len(row) > 14 and row[14])
        if is_wicket:
            wickets += 1

        # Check immediate absorption conditions on this ball
        if runs >= target:
            absorbed = True
            absorbing_state = "WIN"
            absorption_over = (legal_balls - 1) // 6 + 1 if legal_balls > 0 else 1
            break
        elif wickets >= 10:
            absorbed = True
            absorbing_state = "LOSS"
            absorption_over = (legal_balls - 1) // 6 + 1 if legal_balls > 0 else 1
            break
        elif legal_balls >= 120:
            absorbed = True
            absorbing_state = "WIN" if runs >= target else "LOSS"
            absorption_over = 20
            break

        # Check if an over has just completed (e.g. legal_balls = 6, 12, ..., 114)
        if is_legal and (legal_balls % 6 == 0):
            over_num = legal_balls // 6
            if over_num <= 19:
                balls_left = 120 - legal_balls
                runs_needed = target - runs
                crr = runs / (legal_balls / 6.0)
                rrr = runs_needed / (balls_left / 6.0)
                pressure = rrr - crr
                state_name = config.get_state(wickets, pressure)

                rec = {
                    "match_id": match_id,
                    "season": season,
                    "over": over_num,
                    "runs": runs,
                    "wickets": wickets,
                    "balls_left": balls_left,
                    "runs_needed": runs_needed,
                    "CRR": round(crr, 3),
                    "RRR": round(rrr, 3),
                    "pressure": round(pressure, 3),
                    "state": state_name,
                }
                over_records.append(rec)

                # Record transition from previous completed over if applicable
                if last_transient_state is not None:
                    transitions.append({
                        "match_id": match_id,
                        "season": season,
                        "from_state": last_transient_state,
                        "to_state": state_name,
                        "over_from": last_transient_over,
                    })

                last_transient_state = state_name
                last_transient_over = over_num

    # In case the loop ended without explicitly triggering an absorption condition
    if not absorbed:
        absorbed = True
        absorbing_state = "WIN" if runs >= target else "LOSS"
        absorption_over = min(20, (legal_balls - 1) // 6 + 1 if legal_balls > 0 else 1)

    # Transition to absorbing state
    if last_transient_state is not None:
        transitions.append({
            "match_id": match_id,
            "season": season,
            "from_state": last_transient_state,
            "to_state": absorbing_state,
            "over_from": last_transient_over,
        })

    # Final absorbing state record for over_states.csv
    balls_left_final = max(0, 120 - legal_balls)
    runs_needed_final = max(0, target - runs)
    crr_final = round(runs / (legal_balls / 6.0), 3) if legal_balls > 0 else 0.0

    absorbing_record = {
        "match_id": match_id,
        "season": season,
        "over": absorption_over,
        "runs": runs,
        "wickets": wickets,
        "balls_left": balls_left_final,
        "runs_needed": runs_needed_final,
        "CRR": crr_final,
        "RRR": np.nan,
        "pressure": np.nan,
        "state": absorbing_state,
    }
    over_records.append(absorbing_record)

    trace_summary = {
        "match_id": match_id,
        "season": season,
        "target": target,
        "final_runs": runs,
        "final_wickets": wickets,
        "final_legal_balls": legal_balls,
        "absorbing_state": absorbing_state,
        "absorption_over": absorption_over,
        "expected_win": bool(chase_won == 1),
        "actual_win": bool(absorbing_state == "WIN"),
        "negative_balls_left": (balls_left_final < 0),
        "overs_completed": len(over_records) - 1,
    }

    return over_records, transitions, trace_summary

def main():
    print("=" * 70)
    print("SCRIPT 2: Building Match Over States and Transitions")
    print("=" * 70)

    chases_path = config.PROCESSED_DATA_DIR / "chases.csv"
    df_chases = pd.read_csv(chases_path)
    print(f"Loaded {len(df_chases)} matches from {chases_path}")

    all_over_records = []
    all_transitions = []
    traces = []
    mismatch_count = 0
    negative_balls_count = 0

    for _, row in df_chases.iterrows():
        recs, trans, trace = replay_chase(row, config.RAW_DATA_DIR)
        all_over_records.extend(recs)
        all_transitions.extend(trans)
        traces.append(trace)

        if trace["expected_win"] != trace["actual_win"]:
            mismatch_count += 1
            print(f"ERROR: Mismatch in match {trace['match_id']}: expected {trace['expected_win']} got {trace['actual_win']}")

        if trace["negative_balls_left"]:
            negative_balls_count += 1

    df_over_states = pd.DataFrame(all_over_records)
    df_transitions = pd.DataFrame(all_transitions)

    # Save to CSV
    over_states_file = config.PROCESSED_DATA_DIR / "over_states.csv"
    transitions_file = config.PROCESSED_DATA_DIR / "transitions.csv"

    df_over_states.to_csv(over_states_file, index=False)
    df_transitions.to_csv(transitions_file, index=False)

    print(f"Saved {len(df_over_states)} over-state observations to: {over_states_file}")
    print(f"Saved {len(df_transitions)} state transitions to: {transitions_file}")

    print("\n--- Sanity Checks ---")
    print(f"Total matches processed: {len(traces)}")
    print(f"Matches with negative balls_left: {negative_balls_count} (PASS: 0 expected)")
    print(f"Absorption outcome vs chase_won mismatches: {mismatch_count} (PASS: 0 expected)")
    
    # Check that every match has exactly one absorbing state
    absorbing_rows = df_over_states[df_over_states["state"].isin(config.ABSORBING_STATES)]
    match_absorbing_counts = absorbing_rows.groupby("match_id").size()
    single_absorbing = (match_absorbing_counts == 1).all() and len(match_absorbing_counts) == len(df_chases)
    print(f"Every match has exactly one absorbing state: {single_absorbing} (PASS: True expected)")

    print("\n--- Sample Hand-Traced Chases ---")
    sample_indices = [0, len(traces) // 2, len(traces) - 1]
    for idx in sample_indices:
        tr = traces[idx]
        mid = tr["match_id"]
        sub_states = df_over_states[df_over_states["match_id"] == mid]
        print(f"\nMatch ID: {mid} (Season {tr['season']}) | Target: {tr['target']} | Final: {tr['final_runs']}/{tr['final_wickets']} in {tr['final_legal_balls']} legal balls | Result: {tr['absorbing_state']}")
        for _, s in sub_states.iterrows():
            if s["state"] in config.ABSORBING_STATES:
                print(f"  -> End (Over {s['over']}): State={s['state']}, Runs={s['runs']}, Wkts={s['wickets']}, BallsLeft={s['balls_left']}, RunsNeeded={s['runs_needed']}")
            else:
                print(f"  Over {int(s['over']):2d}: State={s['state']:10s} | Runs={int(s['runs']):3d}, Wkts={int(s['wickets']):1d}, Needed={int(s['runs_needed']):3d}, CRR={s['CRR']:5.2f}, RRR={s['RRR']:5.2f}, Press={s['pressure']:+5.2f}")

    print("\n--- Observations and Outgoing Transitions Per State ---")
    # Transient state observations count
    transient_obs = df_over_states[df_over_states["state"].isin(config.TRANSIENT_STATES)]["state"].value_counts()
    
    # Outgoing transitions count per state
    outgoing_counts = df_transitions["from_state"].value_counts()

    state_summary_rows = []
    sparse_states = []

    for st in config.TRANSIENT_STATES:
        obs = transient_obs.get(st, 0)
        out_trans = outgoing_counts.get(st, 0)
        win_trans = len(df_transitions[(df_transitions["from_state"] == st) & (df_transitions["to_state"] == "WIN")])
        loss_trans = len(df_transitions[(df_transitions["from_state"] == st) & (df_transitions["to_state"] == "LOSS")])
        transient_to = out_trans - win_trans - loss_trans

        state_summary_rows.append({
            "State": st,
            "Observations": obs,
            "Outgoing Transitions": out_trans,
            "To Transient": transient_to,
            "To WIN": win_trans,
            "To LOSS": loss_trans,
        })

        if out_trans < 30:
            sparse_states.append((st, out_trans))

    summary_df = pd.DataFrame(state_summary_rows)
    print(summary_df.to_string(index=False))

    print("\n--- State Sparsity Check (< 30 Outgoing Transitions) ---")
    if sparse_states:
        print(f"FLAGGED: {len(sparse_states)} states have fewer than 30 outgoing transitions:")
        for st, cnt in sparse_states:
            print(f"  - {st}: {cnt} transitions")
        print("\nNote: Recommendation / Potential merge options:")
        print("  If needed, P4 and P5 could be merged into P_High (pressure >= 2.0),")
        print("  or W_High with extreme pressure could be combined.")
        print("  Configurable cutoffs are maintained in src/config.py.")
    else:
        print("All 15 transient states have >= 30 outgoing transitions. No merge required!")

    print("=" * 70)

if __name__ == "__main__":
    main()
