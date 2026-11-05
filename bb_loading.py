"""Stage-1 CS amp + black box (Rin || Cin, DC-coupled at out1). Standalone (no run_cs.py needed)."""
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
# 외부 부하 조건 (블랙박스): 한 Case 안에서는 최적화가 끝날 때까지 고정
RIN_OHM = [1000, 1500, 2200, 3300, 4700, 6800, 10000, 22000, 47000, 100000]
CIN_F = [10e-12, 100e-12, 1e-9, 10e-9, 47e-9]
# 1단 설계 변수 (향후 BO 탐색 대상), 지정하지 않으면 Baseline 값 사용
DESIGN_KEYS = ("rd_ohm", "vg_v", "cl_f")
VTH = 0.7  # 넷리스트 모델 VTO와 동일 (동작영역 판별용)
LAMBDA = 0.01  # 넷리스트 모델 LAMBDA와 동일 (손계산 검증용)
HAND_TOL_PCT = 1.0  # Saturation에서 손계산-ngspice 일치 판정 기준


# ---- run_cs.py에서 옮겨온 부분 (회로, 모델, 해석 명령, 측정 방식 동일) ----

# Baseline 값 정의 (run_cs.cases()[0]과 동일)
def cases():
    return [{"case": "baseline", "group": "baseline", "rd_ohm": 5000,
             "cl_f": 1e-12, "vg_v": 1.2}]


# 단일 CS Amp 넷리스트 (run_cs.netlist()와 동일), bb가 있으면 out1에 블랙박스 subckt 추가
def netlist(row, bb=None):
    text = circuit_simulator.netlist(row)
    return text if bb is None else attach_black_box(text, bb)


def attach_black_box(text, bb):
    """Attach the same black-box load to the netlist's out1 node."""
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


# ngspice 경로: 인자 > PATH의 ngspice_con/ngspice > DEFAULT_NGSPICE
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


# ngspice 실행 + 파싱 (run_cs.simulate(stage="cs")와 동일, 파일명 규칙 {case}_cs_{iteration:02})
def simulate(exe, output, row, bb=None, iteration=1):
    return circuit_simulator.simulate(exe, output, row, iteration, netlist_text=netlist(row, bb))


# ---- 블랙박스 Case 정의 ----

def _label(rin, cin):
    cap = f"{round(cin * 1e12):g}p" if cin < 1e-9 else f"{round(cin * 1e9, 6):g}n"
    return f"R{rin / 1e3:g}k_C{cap}"


# Baseline Rout(RD||ro) ~ 4.88 kOhm 기준 팀 분류
def rin_class(rin):
    return "strong" if rin <= 3300 else "moderate" if rin <= 10000 else "weak"


# 블랙박스 Case 정의: Rin x Cin 전체 조합, Rin 바깥 루프 순서로 BB01..BB50
def bb_cases():
    return [dict(case_id=f"BB{i:02}", bb_case=_label(r, c), rin_ohm=r, cin_f=c, rin_class=rin_class(r))
            for i, (r, c) in enumerate(itertools.product(RIN_OHM, CIN_F), 1)]


# 개수, ID 중복, 조합 중복/누락 검사 (실행 결과 rows에도 그대로 사용 가능)
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


# ---- 측정, 지표, 손계산 ----

# 1단 회로 row: Baseline에서 설계 변수만 덮어씀 (Rin/Cin은 여기 들어오지 않음)
def stage1_row(design=None, case="stage1"):
    unknown = set(design or {}) - set(DESIGN_KEYS)
    if unknown:
        raise ValueError(f"Not a stage-1 design variable: {sorted(unknown)}")
    return dict(cases()[0], **(design or {}), case=case)  # 팀 Baseline: RD=5k, CL=1p, VG=1.2


def _pct(value, reference):
    return (value - reference) / reference * 100 if reference else math.nan


# 측정 동작점으로 계산한 손계산 값: |Av| = gm (RD||ro||Rin) / sqrt(1 + (f/f_p)^2),
# f_p = 1/(2 pi (RD||ro||Rin)(CL+Cin)). Saturation 가정이므로 다른 영역은 비교 불가로 표시
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


# Target Gain = Baseline 단독(부하 없음) 1 kHz |Av|, 이후 변경 금지
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


# BO 등에서 재사용할 단일 평가: 고정 부하 bb + 1단 설계 design -> 측정/지표 record (예외 대신 status)
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
    case_table = bb_cases()  # cases()(Baseline 정의)와 이름이 겹치지 않도록 구분
    print(f"Case table OK: {validate_cases(case_table)} cases "
          f"({len(RIN_OHM)} Rin x {len(CIN_F)} Cin, unique IDs and pairs)")
    if args.list:
        for bb in case_table:
            print(f'{bb["case_id"]}  {bb["bb_case"]:13} Rin={bb["rin_ohm"]:>7g} Ohm  '
                  f'Cin={bb["cin_f"]:.3g} F  {bb["rin_class"]}')
        return 0
    exe = find_exe(args.ngspice)
    # ngspice는 긴(>MAX_PATH) 경로에서 로그를 못 만들기 때문에 짧은 ASCII 경로 유지
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
        validate_cases(records)  # 결과 파일에도 50개 조합이 빠짐없이 들어갔는지 재확인
    summarize(records)
    print(f"Results: {output / 'loading.csv'}\nBaseline: {output / 'baseline.csv'}")
    return 0 if all(r["status"] == "ok" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
