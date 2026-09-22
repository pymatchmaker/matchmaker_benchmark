# skf_imm — SKF + IMM + measure graph with synthesised-score chroma emission (2026-09-21/22)

소스: `matchmaker-laurenceyoon/matchmaker/prob/skf_imm.py` (`--method skf_imm`, `reference: score_audio`, 30 fps chroma).
검증 폴더: `output/skf_imm_valid20_20260921/` (protocol/에 각 버전 소스 사본과 sha256), 146·folded: `output/skf_imm_v10_20260922/`.

## 모델 (v10)

- 상태: `(chord k, age a, route)` 빔 ≤200 (Jiang & Raphael SKF). chord는 지속시간 모델 `L_k ~ N(l_k t, σ_ε²+l_k²σ_t²)`로 끝남(σ_ε=0.05t, σ_η=0.01 l t — SKF 원 논문 값).
- 템포: 가설마다 CV/CA/ZV IMM 뱅크. 모드 전이 = `IMMMotionModels.M_play/M_pause`(악보 sojourn에서 유도, 논문 IMM 층과 동일 객체). CA는 Wiener-acceleration(drift 상태, 노이즈는 σ_η에서 유도). ZV는 악보 pause 구간(fermata·쉼표)에서만 진입, hazard는 자체 sojourn(평균 1박), Kalman 갱신 없음.
- 방출: 합성 악보 오디오의 chroma 프레임을 chord별로 묶고, L1 정규화 live chroma의 multinomial cross-entropy를 프레임에 대해 softmin(log-mean-exp) 풀링. rest = 악보의 무음 프레임(없으면 평탄 chroma). ZV의 음향 = (chord + rest)/2, CV/CA = chord.
- 구조: 마디 끝 chord에서 `ScoreGraph.transition_distribution`의 점프 edge로 전진 질량 분배(prior 0.5), route 기록. 곡 끝은 흡수 상태(마지막 chord에서 점프 가능).
- 시작 무음: soft_oltw와 공유하는 `SILENCE_PEAKINESS` 게이트.
- 튜닝 상수 없음. 인자: modes, max_hypotheses, σ 스케일.

## valid20 (soft_oltw+IMM 19/20, SKF 13/20)

| 버전 | 변경 | TR | max\|err\|≤2b | 최대이탈 중앙값 |
|---|---|---:|---:|---:|
| v1 | SKF 템플릿 + ad-hoc 상수 7개, skip 5% | 17 | 5 | 4.5b |
| v3/v4 | 원리적 재설계 + 세그먼트 병합/PDA skip | 10 | — | — |
| no-skip | v4, skip 없음 | 14 | 9 | 1.17b |
| v6 | + rest 템플릿(평탄 스펙트럼), ZV sojourn | 14 | 11 | 1.57b |
| v8 | chroma multinomial 방출 + 평탄 rest, ZV 어디서나 | 16 | 7 | 3.0b |
| v9 | v8 − rest, 시작 무음 게이트 | 14 | 5 | 3.9b |
| **v10** | v8 + score-gated ZV + 게이트 | **16** | **11** | **1.15b** |

교훈: (1) chord skip은 σ_ε∝템포라 짧은 음 1개/2개를 못 가르고 동일화음 반복에서 racing → 제거. (2) 무음은 chroma로 표현 불가(에너지 소실); rest 항은 도움이 되나 ZV는 악보 pause로 게이팅해야 반복음에서 정지하지 않음. (3) 누적 DP 비용을 SKF 방출로 쓰면 과거 증거 이중 계산 + 정규화(racing)/비정규화(lag) 딜레마 → 프레임 풀링만 채택. (4) 남은 실패는 밀집 패시지에서 뒤처진 뒤 복구 불가(SKF 고유).

## 최종 v11 결과 (`output/skf_imm_v11_20260922/`, 2026-09-22 00:42–01:04, workers 3/4, OPENBLAS/OMP threads 1)

v11 = v10 + 프레임이 없는 짧은 chord에 onset 프레임 배정(K.281 1악장: 1335 chord / 3006 프레임에서 빔 붕괴 수정) + 우도를 빔 도달 chord 기준 정규화.

| 설정 | skf_imm v11 | SoftOLTW+IMM (Table 1/2) | SKF |
|---|---:|---:|---:|
| 146 unfolded TR | **114/146 (78.1%)**: ASAP 21/32, Batik 27/30, Vienna 66/84 | 135/146: 31/26/78 | 89/146 |
| 146, 30s창 중앙값 ≤0.5b | 111 | 124 | 89 |
| 146, 전 구간 max\|err\|≤2b | **72** | 48 | 73 |
| 146 성공곡 최대이탈 중앙값 / p95 | **1.59b / 0.49b** | 2.85b / 0.82b | 1.22b / 0.51b |
| 146 성공곡 1b 이상 앞섬 / 뒤처짐 onset | 0.74% / **1.41%** | 1.02% / 3.42% | 0.53% / 2.31% |
| folded Batik 30 TR (2b) | **27/30** | 27/30 (legacy hsoltw) | — |
| folded 성공곡 MeanAE / ≤0.5b / ≤1b | **0.357 / 96.9 / 98.8** | 0.592 / 90.0 / 96.0 | — |
| folded 실패곡 | K.281-3, K.284-3, K.331-1 | K.281, K.283, K.331 | — |
| valid20 TR | 16/20 | 19/20 | 13/20 |
| RTF (146) | 0.08–0.09 | 0.17 | 0.23 |

146곡 실패 45곡 중 36곡은 밀집 패시지에서 뒤처진 뒤 복구 불가(SKF 고유). 세 방법 공통 성공 83곡.
