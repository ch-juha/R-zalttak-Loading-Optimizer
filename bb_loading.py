"""black box (Rin || Cin) 부하가 연결된 1단 CS 증폭기의 로딩 특성을 측정"""
import argparse
import circuit_simulator
import csv
from datetime import datetime
import itertools
import math
from pathlib import Path
import re
import shutil
import subprocess

DEFAULT_NGSPICE = r"C:\ngspice\Spice64\bin\ngspice_con.exe"
# 각 case에선 측정, 최적화 끝날 때까지 부하 조건을 고정
RIN_OHM = [1000, 1500, 2200, 3300, 4700, 6800, 10000, 22000, 47000, 100000]
CIN_F = [10e-12, 100e-12, 1e-9, 10e-9, 47e-9]
# 1단 설계 변수. 지정하지 않으면 Baseline 값 사용
DESIGN_KEYS = ("rd_ohm", "vg_v", "cl_f")
VTH = 0.7  # 넷리스트 모델 VTO와 동일 (동작영역 판별용)
LAMBDA = 0.01  # 넷리스트 모델 LAMBDA와 동일 (손계산 검증용)
HAND_TOL_PCT = 1.0  # 손계산과 ngspice 결과의 허용 오차 


def cases():
    return [{"case": "baseline", "group": "baseline", "rd_ohm": 5000,
             "cl_f": 1e-12, "vg_v": 1.2}]

def netlist(row, bb=None):
    text = circuit_simulator.netlist(row)
    return text if bb is None else attach_black_box(text, bb)


def attach_black_box(text, bb):
    """넷리스트 out1에 동일한 블랙박스 부하 연결"""
    block = (f"XBB out1 BLACKBOX\n.subckt BLACKBOX inp\n"
             f"RIN inp 0 {bb['rin_ohm']:.12g}\nCIN inp 0 {bb['cin_f']:.12g}\n.ends BLACKBOX")
    return text.replace(".control", block + "\n.control", 1)


def read_number(log, name):
    pattern = rf"^\s*{re.escape(name)}\s*=\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"
    match = re.search(pattern, log, re.MULTILINE | re.IGNORECASE)
    if not match:
        raise ValueError(f"Missing result: {name}; inspect this case's log")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise ValueError(f"Non-finite result: {name}")
    return value


def find_exe(argument=None):
    if argument:
        candidate = Path(argument).expanduser()
    else:
        located = shutil.which("ngspice_con") or shutil.which("ngspice")
        candidate = Path(located) if located else Path(DEFAULT_NGSPICE)
    if not candidate.is_file():
        raise SystemExit('ngspice not found. Use --ngspice "C:\\...\\ngspice_con.exe"')
    return candidate.resolve()


def operating_region(vgs, vds):
    if vgs <= VTH + 1e-9:
        return "cutoff"
    return "saturation" if vds >= vgs - VTH else "triode"


def simulate(exe, output, row, bb=None, iteration=1):
    return circuit_simulator.simulate(exe, output, row, iteration, netlist_text=netlist(row, bb))



def _label(rin, cin):
    cap = f"{round(cin * 1e12):g}p" if cin < 1e-9 else f"{round(cin * 1e9, 6):g}n"
    return f"R{rin / 1e3:g}k_C{cap}"


# 기준 출력저항을 기준으로 부하 세기를 구분
def rin_class(rin):
    return "strong" if rin <= 3300 else "moderate" if rin <= 10000 else "weak"


def bb_cases():
    return [dict(case_id=f"BB{i:02}", bb_case=_label(r, c), rin_ohm=r, cin_f=c, rin_class=rin_class(r))
            for i, (r, c) in enumerate(itertools.product(RIN_OHM, CIN_F), 1)]


def validate_cases(rows):
    expected = set(itertools.product(RIN_OHM, CIN_F))
    pairs = [(row["rin_ohm"], row["cin_f"]) for row in rows]
    problems = []
    if len(rows) != len(expected):
        problems.append(f"{len(rows)} cases, expected {len(expected)}")
    for key in ("case_id", "bb_case"):
        if len({row[key] for row in rows}) != len(rows):
            problems.append(f"duplicate {key}")
    if len(set(pairs)) != len(pairs):
        problems.append("duplicate (Rin, Cin) pair")
    if expected - set(pairs):
        problems.append(f"missing pairs {sorted(expected - set(pairs))}")
    if set(pairs) - expected:
        problems.append(f"unexpected pairs {sorted(set(pairs) - expected)}")
    if problems:
        raise SystemExit("Case table invalid: " + "; ".join(problems))
    return len(rows)


def find_case(name):
    for bb in bb_cases():
        if name.lower() in (bb["case_id"].lower(), bb["bb_case"].lower()):
            return bb
    raise SystemExit(f"Unknown case {name}; run with --list to see case IDs")




def stage1_row(design=None, case="stage1"):
    unknown = set(design or {}) - set(DESIGN_KEYS)
    if unknown:
        raise ValueError(f"Not a stage-1 design variable: {sorted(unknown)}")
    return dict(cases()[0], **(design or {}), case=case)  # 팀 Baseline: RD=5k, CL=1p, VG=1.2


def _pct(value, reference):
    return (value - reference) / reference * 100 if reference else math.nan


# 측정 동작점으로 근사이득과 극점 계산. saturation region에서만 비교
def hand_check(row, bb, values):
    gds = LAMBDA * values["id_dc"] / (1 + LAMBDA * values["vds1"])
    conductance = 1 / row["rd_ohm"] + gds + (1 / bb["rin_ohm"] if bb else 0)
    cap = (row["cl_f"] or 0) + (bb["cin_f"] if bb else 0)
    mid = values["gm_s"] / conductance
    pole = conductance / (2 * math.pi * cap) if cap else math.inf
    at = {f: mid / math.sqrt(1 + (f / pole) ** 2) for f in (1e3, 100e3)}
    check = dict(hand_gain_mid=mid, hand_pole_hz=pole,
                 hand_gain_1k=at[1e3], hand_gain_100k=at[100e3],
                 hand_err_1k_pct=_pct(at[1e3], values["gain_1k"]),
                 hand_err_100k_pct=_pct(at[100e3], values["gain_100k"]))
    valid = values["region1"] == "saturation"
    worst = max(abs(check["hand_err_1k_pct"]), abs(check["hand_err_100k_pct"]))
    check.update(hand_valid=valid,
                 hand_note=(("agree" if worst < HAND_TOL_PCT else "mismatch") if valid
                            else f"not_applicable_{values['region1']}"))
    return check


def metrics(target, values):
    gain = values["gain_1k"]
    return dict(target_gain_1k=target,
                error_abs=abs(gain - target),
                error_pct=abs(gain - target) / target * 100,
                retention_1k_pct=gain / target * 100,
                loading_loss_pct=(target - gain) / target * 100,
                retention_100k_pct=values["gain_100k"] / target * 100)


# Target Gain = Baseline 단독(부하 없음) 1 kHz |Av|
def measure_target(exe, output):
    row = stage1_row(case="target")
    try:
        values = simulate(exe, output, row)
    except (OSError, subprocess.TimeoutExpired, ValueError, RuntimeError) as error:
        raise SystemExit(f"Baseline simulation failed, no target: {error}")
    target = abs(values["gain_1k"])
    record = dict({k: row[k] for k in DESIGN_KEYS}, case_id="baseline", bb_case="none", status="ok",
                  **values, **metrics(target, values), **hand_check(row, None, values))
    return target, record


# 시뮬레이션 실패도 결과 표에 남기기 위하여 status로 처리
def evaluate(exe, output, bb, target, design=None, tag="base", iteration=1):
    row = stage1_row(design, f"{bb['case_id']}_{tag}")
    stem = f"{row['case']}_cs_{iteration:02}"  # simulate()의 파일명 규칙
    record = dict(bb, **{k: row[k] for k in DESIGN_KEYS}, target_gain_1k=target,
                  status="failed", error="", netlist=stem + ".cir", log=stem + ".log")
    try:
        values = simulate(exe, output, row, bb, iteration)
    except (OSError, subprocess.TimeoutExpired, ValueError, RuntimeError) as error:
        record["error"] = str(error)
        return record
    record.update(values, **metrics(target, values), **hand_check(row, bb, values), status="ok")
    return record


def evaluate_source_follower(exe, output, bb, target, design, *, rs2_ohm):
    row = stage1_row(design, f"{bb['case_id']}_sf")
    row["rs2_ohm"] = rs2_ohm
    record = {**row, "status": "failed"}
    try:
        text = attach_black_box(circuit_simulator.netlist(row, rs2_ohm=rs2_ohm), bb)
        values = circuit_simulator.simulate(exe, output, row, netlist_text=text, rs2_ohm=rs2_ohm)
    except (OSError, subprocess.TimeoutExpired, ValueError, RuntimeError) as error:
        return dict(record, error=str(error))
    record.update(values, **metrics(target, values), status="ok")
    return record


COLUMNS = ["case_id", "bb_case", "rin_ohm", "cin_f", "rin_class", "rd_ohm", "vg_v", "cl_f",
           "target_gain_1k", "gain_1k", "gain_100k", "error_abs", "error_pct",
           "retention_1k_pct", "loading_loss_pct", "retention_100k_pct",
           "vout_dc", "vgs1", "vds1", "vdsat1", "id_dc", "gm_s", "sat_margin1_v", "region1",
           "hand_valid", "hand_note", "hand_gain_mid", "hand_gain_1k", "hand_gain_100k",
           "hand_pole_hz", "hand_err_1k_pct", "hand_err_100k_pct",
           "status", "error", "netlist", "log"]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(records):
    ok = [r for r in records if r["status"] == "ok"]
    regions = {}
    for r in ok:
        regions.setdefault(r["region1"], []).append(r["case_id"])
    notes = {}
    for r in ok:
        notes[r["hand_note"]] = notes.get(r["hand_note"], 0) + 1
    sat = [r for r in ok if r["hand_valid"]]
    print(f"Simulated: {len(records)}, ok: {len(ok)}, failed: {len(records) - len(ok)}")
    for region, ids in sorted(regions.items()):
        print(f"  {region}: {len(ids)}" + ("" if region == "saturation" else f" -> {', '.join(ids)}"))
    print(f"  hand check: {notes}")
    if sat:
        worst = max(sat, key=lambda r: max(abs(r["hand_err_1k_pct"]), abs(r["hand_err_100k_pct"])))
        print(f"  worst saturation hand error: {worst['case_id']} "
              f"1k={worst['hand_err_1k_pct']:+.4f}%  100k={worst['hand_err_100k_pct']:+.4f}%")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ngspice", default=DEFAULT_NGSPICE, help="Full path to ngspice_con.exe")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--case", help="One black-box case: ID (BB33) or label (R10k_C1n)")
    mode.add_argument("--all", action="store_true", help="All Rin x Cin cases")
    mode.add_argument("--list", action="store_true", help="Print and validate the case table only")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "bb_results")
    args = parser.parse_args(argv)
    case_table = bb_cases()  
    print(f"Case table OK: {validate_cases(case_table)} cases "
          f"({len(RIN_OHM)} Rin x {len(CIN_F)} Cin, unique IDs and pairs)")
    if args.list:
        for bb in case_table:
            print(f'{bb["case_id"]}  {bb["bb_case"]:13} Rin={bb["rin_ohm"]:>7g} Ohm  '
                  f'Cin={bb["cin_f"]:.3g} F  {bb["rin_class"]}')
        return 0
    exe = find_exe(args.ngspice)
    # ngspice는 긴 경로에서 로그를 만들지 못할 수 있어 짧은 ASCII경로를 사용
    output = args.out.resolve() / datetime.now().strftime("loading_%Y%m%d_%H%M%S")
    if not str(output).isascii():
        parser.error("Use an ASCII --out path, e.g. C:\\ngspice_project\\bb_results")
    output.mkdir(parents=True)

    target, baseline = measure_target(exe, output)
    write_csv(output / "baseline.csv", [baseline])
    print(f"Target |Av| @1kHz (baseline, unloaded) = {target:.6g}  [{baseline['region1']}, "
          f"margin {baseline['sat_margin1_v']:.3f} V, hand {baseline['hand_note']}]")
    selected = case_table if args.all else [find_case(args.case)]
    records = []
    for bb in selected:
        record = evaluate(exe, output, bb, target)
        records.append(record)
        if record["status"] == "ok":
            print(f'{bb["case_id"]} {bb["bb_case"]:13} gain_1k={record["gain_1k"]:7.4f} '
                  f'gain_100k={record["gain_100k"]:7.4f} loss={record["loading_loss_pct"]:6.2f}% '
                  f'{record["region1"]:10} margin={record["sat_margin1_v"]:+.3f} V '
                  f'hand={record["hand_note"]}', flush=True)
        else:
            print(f'{bb["case_id"]} {bb["bb_case"]}: failed ({record["error"]})', flush=True)
    write_csv(output / "loading.csv", records)
    if args.all:
        validate_cases(records)
    summarize(records)
    print(f"Results: {output / 'loading.csv'}\nBaseline: {output / 'baseline.csv'}")
    return 0 if all(r["status"] == "ok" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
