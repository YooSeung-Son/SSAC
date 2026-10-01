# Beyond Under Pressure: Measuring Context-Adjusted Clutch Skill in Men's Professional Tennis

MIT Sloan Sports Analytics Conference 2027 연구 논문 공모 초록(v3)의 코드, 결과, LaTeX 원고입니다.
설계는 「설득력을 위해 필요한 결과 설계 (v5)」의 R1–R10을 따릅니다.

- 초록: [`latex/main.pdf`](latex/main.pdf) (1페이지, 제목 포함 466단어)
- 그림 1 / 표 1: [`latex/figure1.pdf`](latex/figure1.pdf), [`latex/table1.tex`](latex/table1.tex)
- 그림과 표를 같은 높이로 나란히 놓는 블록: [`latex/figtab.tex`](latex/figtab.tex)

## 결과 요약 (R1–R10)

| | 결과 | 설계 기준 |
|---|---|---|
| R1 큰 포인트가 승부를 가른다 | 포인트 기대 대비 승패 잔차와 상관: 상위 15% 레버리지 포인트 0.82, 나머지 −0.14. 포인트를 덜 따고 이긴 경기 5.3% | 충족 |
| R2 공식 구성요소는 대부분 운 | 반분 r: 타이브레이크 −0.02, 결정 세트 0.06. 신뢰도 0.5까지 119–612경기 | 충족 |
| R3 공식 지표는 실력을 다시 센다 | 포인트 점유율과 r = 0.53. 실력 제거 후 반분 r 0.08 → 0.04 | 충족 |
| R4 공식 지표는 예측력이 없다 | 반대: BP 결과 예측에 공식 BP 비율이 소폭 정보 추가(log loss −0.0008), UPR+는 추가 없음 | 미충족 (초록 제외) |
| R5 압박 능력은 실재한다 | 분산 1.73배(귀무 최대 1.41), 반분 r 0.26(귀무 최대 0.20), 세트 기준선 0.25, 진짜 SD 0.43%p | 충족 |
| R6 UPR+ > 공식 UPR | 일관성 0.26 vs 0.14(차이 CI −0.04–0.31), 예측 1.01 vs 0.48(차이 CI 0.01–1.16) | 예측은 충족, 일관성은 방향만 |
| R7 다음 시즌 승률 | SD당 +1.0%p/경기 [0.6, 1.4]; Q5−Q1 3.8%p ≈ 시즌 1.8승; 과거 포인트 대비 승리 통제 후 +0.6%p | 충족 (단조 증가는 아님) |
| R8 다음 해 랭킹 상승 | SD당 −4.3% [−11.6, +3.6] | 미충족 |
| R9 51–150위 숨은 선수 | 랭킹 −5.8%(n.s.), 상·하위 절반 차이 없음 | 미충족 |
| R10 견고성 | 반분 r: 하드 0.12, 클레이 0.27, 잔디 0.04(21명) | 미충족 |

정의
- **레버리지**: 마르코프 모델로 계산한, 그 포인트를 이길 때와 질 때의 매치 승률 차이 (`src/markov.py`).
- **UPR+**: 레버리지 가중 포인트 성적 − 같은 경기에서 그 선수의 서브/리턴 수준. 투어 공통 스코어 상태 효과
  (예: 서버가 BP에서 약해짐)는 미리 제거하고, 표본 크기에 맞춰 경험적 베이즈로 축소.
- **포인트 대비 승리**: 실제 승패 − 그 경기의 서브/리턴 포인트 승률로 계산한 마르코프 기대 승률.

## 재현

```bash
pip install -r requirements.txt
scripts/get_data.sh          # 원자료 다운로드 (~450 MB, 저장소에는 포함하지 않음)
./run_all.sh                 # 약 4분: out/*.json, latex/figure1.pdf, latex/main.pdf
```

| 단계 | 스크립트 | 산출 |
|---|---|---|
| Elo·롤링 서브/리턴 (Sackmann 전 경기) | `src/prep.py` | `out/long_all.parquet` |
| 차팅 포인트 레버리지, 매치 기대 승률, 선수 ID | `src/mcp_prep.py` | `out/mcp_points.parquet`, `out/mcp_matches.parquet` |
| R1, R5(보정 전·순열 귀무) | `src/mcp_clutch.py` | `out/mcp_clutch.json` |
| R5(스코어 효과 제거·시뮬레이션 귀무), UPR+ | `src/mcp_clutch2.py` | `out/mcp_clutch2.json` |
| Sackmann 경기별 마르코프 기대 승률 | `src/sack_markov.py` | `out/sack_markov.parquet` |
| R2, R3, R6–R9 | `src/panel.py` | `out/panel.json` |
| R4 | `src/r4_points.py` | `out/r4_points.json` |
| 표 1, 그림 1 수치 | `src/table_fig_v3.py` | `out/table1_v3.json`, `out/fig1_v3.json` |
| 초록 속 견고성 수치 | `src/r7_robustness.py` | `out/r7_robustness.json` |
| 그림 1 | `src/fig_quint.py W H` | `latex/figure1.pdf` |

그림 크기 `W H`(pt)는 LaTeX가 `latex/main.log`에 찍는 `FIGBOX:` 값입니다. 템플릿이 바뀌면 그 값으로 다시 그리면 됩니다.
그림과 표의 높이 맞춤은 다시 그리지 않아도 유지됩니다.

## 데이터와 범위

- Match Charting Project (남자): GitHub 공개분 7,518경기, 128만 포인트. 웹사이트 집계(10,136경기)보다 적습니다.
- Jeff Sackmann ATP 경기·랭킹·선수 파일: 원 저장소(`JeffSackmann/tennis_atp`)가 내려가 있어 미러
  `Aneeshers/tennis-sackmann-archive`를 씁니다.
- 설계안의 "현재 active 선수만" 필터는 적용하지 않았습니다. 성적이 떨어져 은퇴한 선수가 빠지면 R7–R9가
  좋게 보이는 생존 편향이 생기기 때문입니다.

## 라이선스와 출처

두 데이터 모두 CC BY-NC-SA 4.0입니다. 원자료는 이 저장소에 넣지 않았고, 이 저장소의 파생 결과도 같은 조건을 따릅니다.

- *Crowdsourced shot-by-shot professional tennis data* by The Tennis Abstract Match Charting Project,
  <http://www.tennisabstract.com/charting/meta.html>, <https://github.com/JeffSackmann/tennis_MatchChartingProject>
- Jeff Sackmann / Tennis Abstract, ATP match and ranking data (`tennis_atp`)
