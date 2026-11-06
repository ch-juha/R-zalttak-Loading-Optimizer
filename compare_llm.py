"""같은 Case·시드·평가 예산에서 BO 단독과 LLM+BO를 비교
"""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path

import bo_opt
from bb_loading import DEFAULT_NGSPICE, find_case

ROWS = [("best_error_pct", "최종 Target 오차 (%)"),
        ("first_hit_1pct", "오차 <1% 첫 도달 (평가 번호)"),
        ("first_hit_0p1pct", "오차 <0.1% 첫 도달 (평가 번호)"),
        ("best_iteration", "최적점 발견 (평가 번호)"),
        ("best_rd_ohm", "최적 RD (Ohm)"), ("best_vg_v", "최적 VG (V)"),
        ("best_gain_1k", "최적 Gain @1 kHz (V/V)"), ("best_gain_100k", "최적 Gain @100 kHz (V/V)"),
        ("best_region1", "최적점 동작 영역"), ("best_sat_margin1_v", "최적점 Saturation margin (V)"),
        ("n_evaluations", "평가 횟수"), ("n_simulations_total", "ngspice 실행 수 (Target 포함)"),
        ("elapsed_s", "Case 실행 시간 (s)"), ("llm_elapsed_s", "LLM 단계 시간 (s)"),
        ("llm_status", "LLM 상태")]


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def run(label, root, case, seed, ngspice, extra):
    folder = root / label
    folder.mkdir()
    bo_opt.main(["--case", case, "--seed", str(seed), "--ngspice", ngspice, "--out", str(folder)] + extra)
    return next(p for p in folder.iterdir() if p.is_dir())


def fmt(value):
    if value in ("", None):
        return "미도달"
    try:
        number = float(value)
    except ValueError:
        return value
    return f"{number:.6g}" if not number.is_integer() else str(int(number))


def main(argv=None):
    bo_opt.load_project_env()  
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default="BB42")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--llm", choices=["required", "auto"], default="required",
                        help="required (default): stop if the LLM fails, so the result is never BO vs BO")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL"),
                        help="OpenAI model (default: OPENAI_MODEL from .env or the environment)")
    parser.add_argument("--llm-file", type=Path, help="Offline test with a stored mock response (not an LLM)")
    parser.add_argument("--ngspice", default=DEFAULT_NGSPICE)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "bb_results")
    args = parser.parse_args(argv)
    if not args.llm_file and (not args.model or not os.environ.get("OPENAI_API_KEY")):
        parser.error("A real LLM run needs OPENAI_API_KEY and OPENAI_MODEL in the project .env (see .env.example); "
                     "use --llm-file for an offline mock test")
    bb = find_case(args.case)
    kind = "mock" if args.llm_file else "openai"
    root = args.out.resolve() / datetime.now().strftime(f"llm_compare_{bb['case_id']}_{kind}_%Y%m%d_%H%M%S")
    root.mkdir(parents=True)

    bo_dir = run("bo_only", root, bb["case_id"], args.seed, args.ngspice, [])
    extra = ["--llm", args.llm] + (["--llm-file", str(args.llm_file)] if args.llm_file else ["--model", args.model])
    try:
        llm_dir = run("llm_bo", root, bb["case_id"], args.seed, args.ngspice, extra)
    except RuntimeError as error:  # required 모드에서 LLM 실패시 비교 중단
        print(f"\nLLM+BO stopped (--llm {args.llm}): {error}\nBO-only result kept in {bo_dir}")
        return 3

    summaries = {"bo_only": read_rows(bo_dir / f'{bo_opt.result_prefix("off")}_summary.csv')[0], "llm_bo": read_rows(llm_dir / f'{bo_opt.result_prefix(args.llm)}_summary.csv')[0]}
    histories = {"bo_only": read_rows(bo_dir / f'{bo_opt.result_prefix("off")}_history.csv'), "llm_bo": read_rows(llm_dir / f'{bo_opt.result_prefix(args.llm)}_history.csv')}
    llm = read_json(llm_dir / "llm.json")
    response = read_json(llm_dir / "llm_response.json")

    with (root / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "label", "bo_only", "llm_bo"])
        for key, label in ROWS:
            writer.writerow([key, label, summaries["bo_only"].get(key, ""), summaries["llm_bo"].get(key, "")])
    
    with (root / "convergence.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration"] + [f"{name}_{col}" for name in histories
                                         for col in ("phase", "rd_ohm", "vg_v", "error_pct", "best_error_pct_so_far")])
        for i in range(max(len(h) for h in histories.values())):
            row = [i + 1]
            for history in histories.values():
                record = history[i] if i < len(history) else {}
                row += [record.get(c, "") for c in ("phase", "rd_ohm", "vg_v", "error_pct", "best_error_pct_so_far")]
            writer.writerow(row)

    lines = [f"Case {bb['case_id']} (Rin/Cin fixed, not sent to the LLM), seed {args.seed}, "
             f"budget {summaries['bo_only']['n_evaluations']} evaluations each",
             f"LLM source: {llm.get('source', '')}" + ("  [MOCK FILE - NOT AN LLM RESULT]" if args.llm_file else
                                                       f"  model {response.get('model', args.model)}"),
             "", f"{'지표':34} {'BO 단독':>14} {'LLM+BO':>14}"]
    lines += [f"{label:34} {fmt(summaries['bo_only'].get(key)):>14} {fmt(summaries['llm_bo'].get(key)):>14}"
              for key, label in ROWS]
    if llm.get("status") == "accepted":
        lines += ["", f"LLM 진단: {llm.get('diagnosis', '')}",
                  f"LLM 추천 범위: RD {llm['bounds']['rd_ohm']} Ohm, VG {llm['bounds']['vg_v']} V",
                  f"LLM 근거: {llm.get('rationale', '')}", "LLM 초기 후보의 실제 ngspice 결과:"]
        lines += [f"  #{r['iteration']} RD={float(r['rd_ohm']):.1f} VG={float(r['vg_v']):.4f} -> "
                  f"gain_1k={float(r['gain_1k']):.4f} err={float(r['error_pct']):.3f}% "
                  f"margin={float(r['sat_margin1_v']):+.3f} V {r['region1']}"
                  for r in histories["llm_bo"] if r["phase"] == "init_llm" and r["status"] == "ok"]
        if response.get("usage"):
            usage = response["usage"]
            lines.append(f"토큰: input {usage.get('input_tokens')}, output {usage.get('output_tokens')}, "
                         f"response_id {response.get('response_id')}")
    else:
        lines += ["", f"WARNING: LLM advice not used (status {llm.get('status')}: {llm.get('error', '')}); "
                      "the LLM+BO column is the same as BO only."]
    lines += ["", f"BO only: {bo_dir}", f"LLM+BO : {llm_dir}"]
    text = "\n".join(lines)
    (root / "comparison.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + "=" * 70 + "\n" + text + f"\n\nSaved: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
