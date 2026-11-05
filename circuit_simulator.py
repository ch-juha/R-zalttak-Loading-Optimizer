"""기준 CS 회로 하나와 ngspice 실행을 담당하는 공통 모듈입니다.

여러 RD/CL/VG 조합을 생성하는 스윕 기능은 포함하지 않습니다.
기준 회로: VDD=5 V, RD=5 kΩ, VG=1.2 V, CL=1 pF, W/L=20u/1u.
블랙박스 연결은 bb_loading.py, RD/VG 탐색은 bo_opt.py가 담당합니다.
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
    """기준 회로를 새 사전으로 반환하고 요청한 설계값만 덮어씁니다.

    호출마다 독립 사전을 만들어 BO 후보가 원래 기준값을 바꾸지 않게 합니다.
    cl_f=None은 커패시터 미장착이며 최적화기에서는 기본 1 pF를 유지합니다.
    """
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


# 단일 CS 회로의 소자·모델·측정 조건은 기존 코드와 동일합니다.
def netlist(row, *, rs2_ohm=None):
    """1단 CS 회로와 OP/AC 측정 명령을 ngspice 텍스트로 생성합니다.

    AC 입력 크기는 1 V이므로 vm(out1)이 곧 이득 크기(V/V)입니다.
    gain_1k와 gain_100k는 각각 1 kHz, 100 kHz에서 측정합니다.
    """
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
    """ngspice 로그의 '측정이름 = 숫자'를 읽고 유한한 실수인지 확인합니다."""
    pattern = rf"^\s*{re.escape(name)}\s*=\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"
    match = re.search(pattern, log, re.MULTILINE | re.IGNORECASE)
    if not match:
        raise ValueError(f"Missing result: {name}; inspect this case's log")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise ValueError(f"Non-finite result: {name}")
    return value

def find_exe(argument):
    """명시 경로 → PATH → 기존 PC 설치 경로 순으로 ngspice 파일을 찾습니다."""
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
    """현재 LEVEL=1 모델의 Vth=0.7 V로 cutoff/triode/saturation을 분류합니다."""
    if vgs <= 0.7 + 1e-9:
        return "cutoff"
    return "saturation" if vds >= vgs - 0.7 else "triode"


def simulate(exe, output, row, iteration=1, *, netlist_text=None, timeout=60, rs2_ohm=None):
    """회로 한 개를 ngspice로 실행하고 측정 결과를 반환합니다.

    netlist_text가 있으면 해당 회로를 실행하고, 없으면 기준 CS 회로를 생성합니다.
    블랙박스 연결을 위해 전역 함수를 임시 교체할 필요가 없습니다.
    실패는 예외로 전달하며 bb_loading.evaluate()가 실패 레코드로 변환합니다.
    """
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # 파일명으로 사용하는 이름과 실제 설계값을 실행 전에 검증합니다.
    baseline_row({k: row[k] for k in ("rd_ohm", "vg_v", "cl_f")}, row["case"])
    stage = "cs" if rs2_ohm is None else "two_stage"
    stem = f"{row['case']}_{stage}_{iteration:02}"
    cir = output / (stem + ".cir")
    log_path = output / (stem + ".log")
    # 이전 로그를 새 측정으로 오인하지 않고 기존 결과도 덮어쓰지 않습니다.
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
    # stdout/stderr도 별도 보존하여 ngspice의 실행 단계 오류를 확인할 수 있게 합니다.
    (output / (stem + ".process.log")).write_bytes(result.stdout + b"\n" + result.stderr)
    if not log_path.exists():
        log_path.write_bytes(result.stdout + b"\n" + result.stderr)
    if result.returncode:
        raise RuntimeError(f"ngspice exit {result.returncode}; inspect {log_path.name}")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    values = {name: read_number(log, name) for name in
              ("vout_dc", "id_dc", "gm_s", "gain_1k", "gain_100k")}
    # 소스가 접지된 현재 CS 모델에서는 VDS=Vout, VGS=VG입니다.
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
    """직접 실행하면 블랙박스 없는 기준 회로 하나만 측정합니다."""
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
