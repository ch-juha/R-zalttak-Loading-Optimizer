"""기준 1단 CS 회로의 넷리스트 생성과 ngspice 실행을 담당

기준 조건: VDD=5V, RD=5 kΩ, VG=1.2 V, CL=1 pF
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import shutil
import subprocess


def baseline_row(design=None, case="baseline"):
    # 기준 회로에 요청한 설계값을 반영한 새 사전을 반환
    design = design or {}
    if set(design) - {"rd_ohm", "vg_v", "cl_f"}:
        raise ValueError("Unknown stage-1 design variable")
    row = dict(case=case, group="baseline", rd_ohm=5000., vg_v=1.2, cl_f=1e-12)
    row.update(design)
    for key in ("rd_ohm", "vg_v", "cl_f"):
        value = row[key]
        if key == "cl_f" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid {key}: {value}")
    if row["rd_ohm"] <= 0 or not 0 <= row["vg_v"] <= 5 or (row["cl_f"] is not None and row["cl_f"] <= 0):
        raise ValueError("Require RD>0, 0<=VG<=5, CL>0 or None")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", case):
        raise ValueError("Case name must contain only ASCII letters, digits, _ or -")
    return row


def netlist(row, *, rs2_ohm=None):
    """1단 CS 회로와 OP/AC 측정 명령을 ngspice 텍스트로 생성"""
    loads = []
    if row["cl_f"] is not None:
        loads.append(f'CL out1 0 {row["cl_f"]:.12g}')
    drain = "out1" if rs2_ohm is None else "mid"
    sf_devices = ""
    sf_measurements = ""
    if rs2_ohm is not None:
        if not math.isfinite(rs2_ohm) or rs2_ohm <= 0:
            raise ValueError("RS2 must be finite and positive")
        sf_devices = f"M2 vdd mid out1 out1 NM W=100u L=1u\nRS2 out1 0 {rs2_ohm:.12g}\n"
        sf_measurements = """let vds1 = v(mid)
let vds2 = v(vdd) - v(out1)
let vgs2 = v(mid) - v(out1)
let id2_dc = @m2[id]
let gm2_s = @m2[gm]
print vds1 vds2 vgs2 id2_dc gm2_s
"""
    return f"""* Automated CS experiment: {row['case']}
.temp 27
VDD vdd 0 DC 5
VIN in 0 DC {row['vg_v']:.12g} AC 1
RD vdd {drain} {row['rd_ohm']:.12g}
M1 {drain} in 0 0 NM W=20u L=1u
{sf_devices}.model NM NMOS (LEVEL=1 VTO=0.7 KP=200u LAMBDA=0.01)
{chr(10).join(loads)}
.control
set noaskquit
set numdgt=12
op
let vout_dc = v(out1)
let id_dc = @m1[id]
let gm_s = @m1[gm]
print vout_dc id_dc gm_s
{sf_measurements}ac dec 100 1 100Meg
meas ac gain_1k FIND vm(out1) AT=1k
meas ac gain_100k FIND vm(out1) AT=100k
quit 0
.endc
.end
"""

def read_number(log, name):
    # ngspice 로그에서 유한한 측정값을 읽음
    pattern = rf"^\s*{re.escape(name)}\s*=\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"
    match = re.search(pattern, log, re.MULTILINE | re.IGNORECASE)
    if not match:
        raise ValueError(f"Missing result: {name}; inspect this case's log")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise ValueError(f"Non-finite result: {name}")
    return value

def find_exe(argument):
    # 명시 경로, PATH, 기본 경로 순으로 ngspice 실행 파일을 찾음.
    if argument:
        candidate = Path(argument).expanduser()
    else:
        located = shutil.which("ngspice_con") or shutil.which("ngspice")
        known = Path.home() / "OneDrive" / "바탕 화면" / "ngspice-47_64" / "Spice64" / "bin" / "ngspice_con.exe"
        candidate = Path(located) if located else known
    if not candidate.is_file():
        raise SystemExit('ngspice not found. Use --ngspice "C:\\...\\ngspice_con.exe"')
    return candidate.resolve()

def operating_region(vgs, vds):
    # LEVEL=1 모델의 Vth=0.7 V로 cutoff/triode/saturation을 분류
    if vgs <= 0.7 + 1e-9:
        return "cutoff"
    return "saturation" if vds >= vgs - 0.7 else "triode"


def simulate(exe, output, row, iteration=1, *, netlist_text=None, timeout=60, rs2_ohm=None):
    # 회로 한 개를 ngspice로 실행하고 측정 결과를 반환
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline_row({k: row[k] for k in ("rd_ohm", "vg_v", "cl_f")}, row["case"])
    stage = "cs" if rs2_ohm is None else "two_stage"
    stem = f"{row['case']}_{stage}_{iteration:02}"
    cir = output / (stem + ".cir")
    log_path = output / (stem + ".log")
    # 이전 로그 재사용하거나 덮어쓰지 않도록 파일명 확인
    if cir.exists() or log_path.exists():
        raise ValueError(f"Simulation files already exist: {stem}")
    cir.write_text(netlist(row, rs2_ohm=rs2_ohm) if netlist_text is None else netlist_text, encoding="ascii")
    try:
        result = subprocess.run([str(exe), "-n", "-b", "-o", log_path.name, cir.name],
                                cwd=output, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        with log_path.open("ab") as handle:
            handle.write(b"\nSimulation timeout\n" + (error.stdout or b"") + (error.stderr or b""))
        raise
    # 실행 오류 확인할 수 있도록 stdout/stderr도 별도 저장
    (output / (stem + ".process.log")).write_bytes(result.stdout + b"\n" + result.stderr)
    if not log_path.exists():
        log_path.write_bytes(result.stdout + b"\n" + result.stderr)
    if result.returncode:
        raise RuntimeError(f"ngspice exit {result.returncode}; inspect {log_path.name}")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    values = {name: read_number(log, name) for name in
              ("vout_dc", "id_dc", "gm_s", "gain_1k", "gain_100k")}

    values.update(vds1=values["vout_dc"], vgs1=row["vg_v"], vdsat1=row["vg_v"] - .7)
    if rs2_ohm is not None:
        values.update({name: read_number(log, name) for name in
                       ("vds1", "vds2", "vgs2", "id2_dc", "gm2_s")})
        values.update(vdsat2=values["vgs2"] - .7,
                      region2=operating_region(values["vgs2"], values["vds2"]))
        values["sat_margin2_v"] = values["vds2"] - values["vdsat2"]
    values.update(region1=operating_region(values["vgs1"], values["vds1"]),
                  sat_margin1_v=values["vds1"] - values["vdsat1"],
                  netlist=cir.name, log=log_path.name)
    return values


def main(argv=None):
    # 직접 실행 시 무부하 기준 회로를 측정
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ngspice", help="ngspice 실행 파일 경로")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "baseline_results")
    args = parser.parse_args(argv)
    exe = find_exe(args.ngspice)
    folder = args.out.resolve() / datetime.now().strftime("baseline_%Y%m%d_%H%M%S_%f")
    if not str(folder).isascii() or len(str(folder)) > 180:
        parser.error("Use a short ASCII output path")
    row = baseline_row()
    values = simulate(exe, folder, row)
    record = dict(row, **values)
    (folder / "baseline.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"Results: {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
