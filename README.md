# LLM-Assisted Interstage Loading Analysis and Signal Loss Optimization Using Equivalent Circuit Models

**Team R zalttak (R잘딱)**  
2026 Ewha Engineering Capstone Design Contest

## 1. Project Overview

본 프로젝트는 아날로그 증폭기에서 발생하는 **Loading Effect(부하 효과)**를 분석하고, LLM(Large Language Model)과 Bayesian Optimization(BO)을 활용하여 신호 전달 손실을 최소화하는 자동 최적화 시스템을 개발하는 것을 목표로 합니다.

Common-Source(CS) Amplifier에 서로 다른 입력 임피던스를 갖는 블랙박스 회로가 연결되었을 때 발생하는 Gain 감소를 분석하고, 기존 목표 Gain에 가까워지도록 회로 파라미터를 최적화합니다.

## 2. System Architecture

1. **Baseline Circuit:** 기준 CS Amplifier의 Target Gain 측정
2. **Black-Box Loading:** 50개 Rin–Cin 조합으로 Loading Effect 시뮬레이션
3. **LLM Analysis:** 회로 동작 상태 분석 및 RD·VG 초기 탐색값 제안
4. **Bayesian Optimization:** ngspice와 연동하여 RD·VG 최적화
5. **Performance Evaluation:** BO Only와 LLM-Enhanced BO 결과 비교
6. **Post-BO Analysis:** LLM이 Source Follower 추가 여부를 판단하고 추가 회로를 시뮬레이션하여 평가

현재 버전의 Source Follower 기능은 구조 추가 후 1회 평가 방식이며, SF 파라미터에 대한 별도 BO 최적화는 포함하지 않습니다.

## 3. Development Environment

- Language: Python
- Circuit Simulator: ngspice
- Optimization: Bayesian Optimization (Gaussian Process + Expected Improvement)
- LLM: OpenAI API
- Libraries: NumPy, SciPy, scikit-learn, OpenAI, Pydantic, python-dotenv

## 4. Project Structure

```text
R-zalttak-Loading-Optimizer/
├── bb_loading.py          # Black-box loading simulation
├── circuit_simulator.py   # SPICE netlist generation and simulation
├── bo_opt.py              # BO and LLM-enhanced optimization
├── llm_optimizer.py       # LLM analysis and circuit advice
├── compare_llm.py         # BO-only vs. LLM+BO comparison
├── requirements.txt       # Python dependencies
├── .env.example           # API configuration template
├── .gitignore
└── README.md
```

## 5. Installation

### Prerequisites

- Python 3.10 or later
- ngspice installed
- OpenAI API key (required for LLM features)

### Install Dependencies

```bash
py -m pip install -r requirements.txt
```

### Configure Environment Variables

Copy `.env.example` to `.env` and enter your own OpenAI API key.

```dotenv
OPENAI_API_KEY=YOUR_API_KEY_HERE
OPENAI_MODEL=gpt-6.1-sol
```

Ensure that `.env` is not committed to GitHub.

The default ngspice executable path is `C:\ngspice\Spice64\bin\ngspice_con.exe`. If ngspice is installed elsewhere, supply the appropriate `--ngspice` path.

## 6. Usage

### Black-Box Case List

```bash
py bb_loading.py --list
```

### Run All 50 Loading Cases

```bash
py bb_loading.py --all
```

### Bayesian Optimization Only

```bash
py bo_opt.py --case BB42 --llm off
```

### LLM-Enhanced Bayesian Optimization

```bash
py bo_opt.py --case BB42 --llm required
```

### Compare BO Only and LLM-Enhanced BO

```bash
py compare_llm.py --case BB42
```

### Check LLM Connection

```bash
py bo_opt.py --check-llm
```

Simulation results, optimization histories, and performance summaries are saved under `bb_results/`.

## 7. Optimization Objective

The primary objective is to minimize the relative gain error at 1 kHz:

\[
\text{Gain Error}(\%) =
\frac{|A_{v,\mathrm{optimized}}-A_{v,\mathrm{target}}|}
{|A_{v,\mathrm{target}}|}\times100
\]

The optimization also considers MOSFET saturation conditions, with a required saturation margin of 0.05 V.

The black-box load parameters Rin and Cin remain fixed within each test case.

## 8. Notes

- The project uses 50 predefined black-box loading conditions.
- BO-only and LLM-enhanced optimization can be compared under the same simulation budget.
- The external black-box parameters are not optimized.
- OpenAI API usage may incur charges.
- ngspice must be installed separately on each user's computer.

## 9. Team

**R zalttak (R잘딱)**

Ewha Womans University  
2026 Engineering Capstone Design Contest
