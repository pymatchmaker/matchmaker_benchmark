# IMM와 score following: 문헌 조사 및 구현 판단

2026-09-20. 이 문서는 실험 근거를 코드 밖에 보관한다. 원문을 확인한 논문과 초록·서지 중심으로 확인한 논문을 구별한다. 다른 분야의 결과를 이 벤치마크의 성능 보장으로 해석하지 않는다.

## 현재 판단

IMM은 운동 모델의 불확실성을 처리한다. 관측이 어느 악보 위치에서 발생했는지의 불확실성은 별도 문제다. 누적 DP가 이미 선택한 잘못된 위치를 매 프레임 독립 Gaussian 관측처럼 넣으면, 표준 IMM 수식을 정확하게 구현해도 그 오류를 반복 학습한다. 여러 dynamics를 추가하는 것만으로 데이터 연관 오류가 해결되지는 않는다.

악보로 관측 불확실성을 예측하고 system prediction에 더 의존한다는 방향은 타당하다. 다만 pitch-class가 일정한 구간의 길이는 관측 분산의 대용량이지, 보정된 센서 오차 분산은 아니다. 같은 음의 반복 타격은 pitch-class가 같아도 리듬 정보를 제공한다. Op.38 도입부는 이 구별이 특히 중요하다.

사용자가 보고한 SKF의 **모든 방법의 공통 tracked subset에서도 낮은 오차**는 단순한 쉬운 곡 선택 효과로 설명할 수 없다. SKF의 관측·템포 갱신·위치 출력 구조를 비교해야 한다. 현재 코드에서 확인한 차이는 다음과 같다.

1. 위치·age 가설마다 템포 Gaussian을 유지한다. 다른 위치 가설의 템포를 즉시 하나로 합치지 않는다.
2. 같은 chord에 머무르는 동안 tempo KF를 반복 갱신하지 않는다. 다음 onset으로 진행할 때 경과시간을 관측한다.
3. duration의 conditional survival probability로 stay/advance를 정한다.
4. 선택한 chord 안에서는 age와 tempo로 연속 위치를 출력한다.
5. chroma 대신 raw spectrum과 합성 spectral template을 사용한다. 따라서 정확도 차이를 tempo model만의 효과로 단정할 수 없다.

현 SKF 코드에도 논문과의 차이가 있다. chord를 새로 시작하는 음들만으로 구성하며 지속 중인 음과 note offset은 충분히 반영하지 않는다. 또한 `onset_beat / 4`를 whole-note 단위로 사용한다. Op.38의 6/8박자에서는 `onset_quarter`와 `onset_beat`가 두 배 차이 난다. 단위 수정 효과는 별도 validation 실험으로 확인한다.

## 논문별 근거

| 문헌 | 확인 범위 | 해당 분야의 처리 방식과 이 작업에 주는 근거 |
|---|---|---|
| [Blom & Bar-Shalom, 1988, The interacting multiple model algorithm for systems with Markovian switching coefficients](https://doi.org/10.1109/9.1299) | 원 논문 자료 | interaction, 모델별 필터, likelihood 기반 mode posterior, moment matching이 핵심이다. 관측 모델의 적합성까지 자동으로 보장하지 않는다. |
| [Li & Jilkov, 2005, Survey of maneuvering target tracking, Part V](https://ieeexplore.ieee.org/abstract/document/1561886) | 저자 preprint | 모델 집합과 switching 설계를 다룬다. measurement-origin uncertainty는 이 survey의 범위 밖이다. dynamics와 association을 구분해야 한다. |
| [Kirubarajan & Bar-Shalom, 2004, Probabilistic data association techniques for target tracking in clutter](https://doi.org/10.1109/JPROC.2003.823149) | 초록·서지 | 단일 관측을 확정하기보다 가능한 관측의 association probability를 사용한다. 낮은 detection probability와 clutter는 IMM과 별개로 처리한다. |
| [Bar-Shalom, Daum & Huang, 2009, The probabilistic data association filter](https://ieeexplore.ieee.org/document/5338565/) | 저자 자료 | 후보 평균뿐 아니라 association spread가 covariance에 들어간다. missed detection 가설도 필요하다. 단순 acoustic softmax와 정식 PDA는 같지 않다. |
| [Chen & Tugnait, 2001, IMM/JPDA filtering and fixed-lag smoothing](https://doi.org/10.1016/S0005-1098(00)00158-8) | 초록·서지 | maneuver와 association을 함께 처리한다. fixed-lag smoothing의 이익에는 지연이 있으므로 즉시 출력하는 online follower와 구분해야 한다. |
| [Multisensor tracking using IMMPDA with simultaneous measurement update, 2005](https://ieeexplore.ieee.org/document/1541458/) | 초록·서지 | 센서 관측의 결합 likelihood와 association을 함께 계산한다. 독립이 아닌 posterior를 새 관측으로 반복 결합하는 설계와 구분된다. |
| [Kirubarajan et al., 2000, Ground target tracking with variable structure IMM estimator](https://ieeexplore.ieee.org/document/826310/) | 저자 서지·초록 | 지도 정보를 이용해 적절한 모델 구조를 선택한다. 악보로 pause mode의 가능성을 제한하는 발상과 연결되지만, 악보가 실제 정지를 강제하는 관측은 아니다. |
| [Kirubarajan & Bar-Shalom, 2003, Tracking evasive move-stop-move targets with a GMTI radar](https://doi.org/10.1109/TAES.2003.1238762) | 초록·서지 | 정지로 관측이 사라지는 상황을 VS-IMM으로 다룬다. 관측 부재와 계속 같은 위치가 측정되는 경우를 구분한다. threshold 선택이 있는 방법이며 무조정 방법은 아니다. |
| [Lukeš & Říha, 2021, Variable structure IMM for highly aperiodic sensors with poor measurement precision](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/rsn2.12102) | 원문 | 불규칙한 시간 간격에 맞춘 transition과 센서 geometry에 따른 정확도 정보를 사용한다. 관측이 나쁠 때 복잡한 maneuver model이 항상 도움이 되지는 않는다. score-derived observability의 가장 직접적인 타 분야 근거다. |
| [Li & Jia, Kullback–Leibler divergence for IMM estimation with random matrices](https://arxiv.org/abs/1411.1284) | preprint·출판 정보 | unknown measurement covariance를 random matrix와 변분 추론으로 추정한다. covariance adaptation은 지속적인 위치 bias나 잘못된 association을 해결하는 것과 다르다. |
| [Robust IMM based on Student's t-distribution, 2019](https://doi.org/10.3390/s19224830) | 원문 검색 자료 | heavy-tailed 관측을 robust likelihood로 처리한다. 일시적인 outlier에는 맞지만 반복되는 잘못된 악보 위치를 정상 위치로 복원하는 근거는 아니다. 추가 prior와 추론 설정이 필요하다. |
| [Youn et al., 2021, A Novel Multiple-Model Adaptive Kalman Filter for an Unknown Measurement Loss Probability](https://macsphere.mcmaster.ca/items/96ddab82-83d1-47f5-8afa-6670431af9db) | 저자 원문 | 상태와 관측 유실 확률을 함께 추론한다. 정보 없는 프레임에서 prediction만 유지하는 방법에 근거를 제공하지만, score silence가 곧 유실임을 뜻하지는 않는다. |
| [Visina et al., 2020, Track-to-Track Fusion Using Inside Information From Local IMM Estimators](https://isif.org/files/isif/2024-01/06-052019-0018R2_LR.pdf) | 원문 | 이미 필터링된 track을 결합할 때 correlation과 모델별 정보를 고려한다. recurrent DP endpoint의 정보 재사용 문제와 관련된 유추이며 해당 식을 그대로 적용할 수는 없다. |
| [Acar & Orguner, 2021, Decorrelation of Previously Communicated Information for an IMM Filter](https://doi.org/10.1109/TAES.2020.3018275) | 저자 초록 | 이전에 공유한 정보를 다시 결합하는 중복을 제거한다. DP posterior를 매 프레임 fresh sensor reading으로 취급하는 가정에 주의를 준다. |
| [Labeled RFS-Based Track-Before-Detect for Multiple Maneuvering Targets, 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4721749/) | 원문 검색 자료 | hard detection 전에 raw observation과 여러 dynamics를 결합한다. score following의 acoustic evidence를 위치로 조기 축약하지 않는 방향과 관련된다. 전체 multi-target RFS 구조를 도입할 필요는 없다. |
| [A multi-rate multiple model track-before-detect particle filter, 2009](https://doi.org/10.1016/j.mcm.2008.02.009) | 초록·서지 | 저신호 상황에서 detection 이전 관측을 축적한다. 계산량과 모델 복잡도가 증가하므로 최소 변경의 첫 선택은 아니다. |
| [Adaptive IMM for underwater tracking with randomly delayed measurements, 2023](https://www.sciencedirect.com/science/article/pii/S0029801823013173) | 초록·서론 | 관측 지연을 상태와 관측 모델에 명시한다. DP와 feature latency를 단순 분산 확대만으로 대신할 수 없다는 관련 사례다. |
| [IMM with a Maximum Correntropy Criterion for GPS Navigation, 2023](https://doi.org/10.3390/app13031782) | 출판사 자료 | multipath 등 비정상 관측을 robust criterion으로 완화한다. kernel width 같은 추가 선택이 필요하므로 이번 최소 변경 목표의 우선순위는 낮다. |
| [Synergistic Dual-Adaptive IMM Filtering, 2026](https://doi.org/10.1016/j.phycom.2026.103340) | 9월 공개 preproof·초록 | robust noise adaptation과 transition adaptation을 함께 사용한다. 여러 조절 장치가 들어가며 일반적인 전역 수렴 보장으로 해석할 수 없다. 단순한 첫 구현의 근거로 채택하지 않는다. |
| [Pulford, A Survey of Manoeuvring Target Tracking Methods](https://arxiv.org/abs/1503.07828) | 초록 | maneuver detection, association, smoothing을 포함한 넓은 비교 맥락이다. arXiv 업로드 연도와 원 출판 연도를 혼동하지 않는다. |
| [Jiang & Raphael, 2020, Score Following with Hidden Tempo Using a Switching State-Space Model](https://archives.ismir.net/ismir2020/paper/000159.pdf) | 원문 | duration survival, onset 전환 때의 KF, 위치·age별 가설이 직접 관련된다. 논문 자체도 noise scale과 beam 크기를 설정한다. 기존 SKF의 성공을 무조정 IMM의 성공으로 동일시하지 않는다. |
| [Cemgil et al., On tempo tracking: Tempogram Representation and Kalman filtering](https://www.mcg.uva.nl/mcg-2023/papers/mmm-27.pdf) | 원문 | tempo를 잠재변수로 두고 시간 관측과 연결한다. 매 프레임의 정체한 score position을 tempo 0의 독립 증거로 사용하는 것과 구별된다. |
| [Loumponias & Tsaklidis, Kalman filtering with censored measurements](https://arxiv.org/abs/2002.08597) | 원문 자료 | censored 관측의 조건부 Gaussian moment를 계산한다. 우리가 시도한 유한 harmonic interval 관측은 별도 응용 가정이며, 원 논문이 score interval의 타당성을 입증하지 않는다. |
| [Yang, Zhu & He, 2021, IMM Robust Cardinality Balance Multi-Bernoulli Filter with Interval Measurement](https://doi.org/10.1049/cje.2021.08.009) | 출판사 원문 자료 | interval likelihood와 IMM을 결합한다. multi-target detection/clutter 추론은 현재 단일 score follower의 요구보다 크다. |
| [Target Tracking in Boost Stage with Quantized Measurements, 2020](https://www.jstage.jst.go.jp/article/tjsass/63/5/63_T-19-62/_pdf) | 원문 검색 자료 | quantization 구간을 관측 정보로 사용한다. 구간 관측을 중심점 하나로 축약하지 않는 관련 사례다. |
| [IEEE AESS, Filter Design for Radar Tracking of Maneuvering Targets, 2023](https://ieee-aess.org/presentation/webinar/filter-design-radar-tracking-maneuvering-targets) | 공식 강연 개요 | 센서 오차와 maneuver scale에 근거한 Q 설계를 강조한다. 현재 한 박 동안 위치 분산을 R로 맞춘 선택도 검증해야 할 모델 가정이다. |

## 검증한 후속 실험

전부 validation 20곡, 기존 `test_audio.py --no-plots`를 사용했다. 수치는 각 실행의 `summary_tracked.json`에서 읽었다. 새 scalar parameter 탐색은 하지 않았다. architecture 선택 역시 validation 선택이라는 점은 동일하다.

| 방법 | Tracked | Tracked beat MAE | ≤0.5b | ≤1b |
|---|---:|---:|---:|---:|
| SoftOLTW | 19/20 | 0.2306 | 89.01% | 95.64% |
| 현 standard IMM | 19/20 | 0.2274 | 88.69% | 95.23% |
| harmonic interval 관측 | 19/20 | 0.2291 | 88.85% | 95.31% |
| interval 관측 + 예측 corridor | 19/20 | 0.2290 | 88.86% | 95.32% |
| 현재 프레임 acoustic candidate mixture | 0/20 | — | — | — |

interval 방식은 recovery를 입증하지 못했다. acoustic mixture 실패는 이번 likelihood scale·association·corridor 구성의 실패다. missed-detection 가설과 보정된 acoustic likelihood를 갖춘 정식 IMM-PDA 전체가 부적합하다는 결론은 아니다. 어느 방식도 개선 근거 없이 본 구현에 추가하지 않는다.

## 다음 변경의 기준

반복된 동일 pitch의 onset을 보존하고, 새 timing evidence가 생길 때 tempo를 갱신하는지 확인한다. SKF의 단위 문제는 먼저 고립된 비교로 확인한다. 성능을 위한 예외 규칙, 특정 곡 분기, full146 점수에 따른 parameter 선택은 추가하지 않는다. 문헌에 등장한다는 이유만으로 robust loss, adaptive transition, particle association을 모두 쌓지 않는다.

최종 목표는 baseline보다 최소 3곡 많은 130/146 이상, 낮은 tracked MAE, 높은 두 tolerance coverage다. 이 단계에서는 달성하지 못했다. 이후 상관 관측 오차 모델로 130/146을 달성했으며 최종 결과는 `imm-review-20260920.md`에 기록했다.

## 추가 조사: ICASSP, 신호처리, map matching

| 문헌 | 확인 범위 | 적용 판단 |
|---|---|---|
| [Popescu & Zeljkovic, ICASSP 1998, Kalman filtering of colored noise for speech enhancement](https://doi.org/10.1109/ICASSP.1998.675435) | 저자 원문 검색 자료 | speech와 noise를 각각 AR process로 둔다. 관측 오차의 시간 구조를 상태에 포함한다는 신호처리 쪽 선행 근거다. speech enhancement의 SNR 결과를 score-following 성능으로 옮기지는 않는다. |
| [Wang & Brookes, ICASSP 2014, Speech enhancement using a modulation domain Kalman filter post-processor with a Gaussian mixture noise model](https://dihana.cps.unizar.es/proceedings/ICASSP/2014/papers/p7074-wang.pdf) | 원문 검색 자료, 직접 PDF 열기는 실패 | 전처리된 신호의 잔여 오차가 Gaussian이 아니고 frame overlap으로 상관되어 있음을 모델링한다. 이미 추정된 DP 위치를 raw independent sensor reading으로 취급하지 말아야 한다는 직접적인 유추가 가능하다. 현재 후보는 이 논문의 GMM을 복제하지 않고 AR(1) 오차 상태 하나만 사용한다. |
| [Chakrabarty et al., ICASSP 2014, Extended Kalman filter with probabilistic data association for multiple non-concurrent speaker localization in reverberant environments](https://doi.org/10.1109/ICASSP.2014.6855047) | 원문 검색 자료·저자 초록 | 잔향 환경에서 측정 하나를 선택하는 대신 여러 narrowband DOA 후보를 PDA로 결합한다. 여러 acoustic 위치 후보를 보존하는 방향의 근거다. 이번 실패한 acoustic mixture는 보정된 likelihood와 missed-detection model이 부족하므로 이 논문의 PDA와 동일하다고 주장하지 않는다. |
| [Mansour & Waters, ICASSP 2013, Map-assisted Kalman filtering](https://doi.org/10.1109/ICASSP.2013.6638250) | 저자 공개 원문 텍스트 | 지도 제약을 measurement로 넣어 상태와 covariance를 함께 갱신한다. KF 출력 위치만 사후에 clipping하면 covariance가 일치하지 않는다는 지적이 중요하다. 악보의 허용 위치·진행 방향을 사용할 경우에도 상태만 강제 수정하지 않아야 한다. |
| [Newson & Krumm, SIGSPATIAL 2009, Hidden Markov map matching through noise and sparseness](https://www.ismll.uni-hildesheim.de/lehre/semSpatial-10s/script/6.pdf) | 원문 | geometry에 가까운 도로 하나만 고르지 않고 route transition과 observation likelihood를 함께 사용한다. noise scale과 transition scale은 ground truth로 추정한 parameter이며 무조정 방법이 아니다. 이 작업에서 동일한 calibration을 한다면 validation 20곡으로 제한해야 한다. |
| [Murphy, Pao & Yuen, Map matching when the map is wrong](https://arxiv.org/html/1809.09755v2) | 원문, 2019 v2 preprint | on-road HMM과 off-road KF를 semi-interacting 구조로 결합한다. 잘못된 지도 제약이 독립적인 추정기까지 오염시키지 않게 한다. 원고의 해당 인용 연도·최종 출판 정보는 별도로 확인해야 한다. DP와 KF가 서로의 결론을 순환적으로 확신하는 문제에 관련되지만, score feature와 GPS는 관측 구조가 달라 그대로 이식할 수 없다. |
| [Shmaliy et al., IET Signal Processing 2020, Kalman and UFIR state estimation with coloured measurement noise using backward Euler method](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-spr.2019.0166) | 출판사 원문 | colored measurement noise를 처리하는 state augmentation과 measurement differencing을 다룬다. 현재 후보의 추가 오차 상태는 전자의 표준 구조를 따른다. 악보에서 correlation scale을 정하는 부분은 우리의 모델 가정이다. |
| [Adaptive Gaussian mixture filter for Markovian jump nonlinear systems with colored measurement noises, 2018](https://www.sciencedirect.com/science/article/abs/pii/S0019057818302155) | 출판사 초록 | Markov switching과 autoregressive measurement noise를 함께 다룬다. 운동 모드와 관측 오차 dynamics를 분리하는 선행 사례다. 비선형·적응형 전체 방법은 최소 변경 목표상 도입하지 않는다. |
| [Generalized bias compensated pseudolinear Kalman filter for colored noisy bearings-only measurements](https://www.sciencedirect.com/science/article/pii/S0165168421003686) | 출판사 초록·서론 | 첫 필터의 출력을 두 번째 필터 입력으로 쓰는 cascaded 구조에서 colored noise가 발생하는 사례를 다룬다. SoftOLTW 후단 KF의 독립 관측 가정을 검토하는 데 관련된다. |
| [Passive underwater tracking with unknown measurement noise statistics, Digital Signal Processing 2024](https://doi.org/10.1016/j.dsp.2024.104648) | 출판사 초록 | 관측 noise의 covariance뿐 아니라 mean도 미지수로 추정한다. 지속적인 위치 오차를 R 확대만으로 처리하는 한계를 뒷받침하는 관련 모델이다. |
| [Adaptive Choice of Process Noise Covariance in Kalman Filter Using Measurement Matrices, TCST 2024](https://doi.org/10.1109/TCST.2023.3339732) | 출판사 초록 | innovation 크기에 무조건 반응하지 않고 measurement matrix를 이용해 불필요한 covariance inflation을 줄인다. 악보의 관측 가능성과 모델 오차를 분리해야 한다는 방향에 관련된다. |
| [Outlier-Robust Centralized and Distributed Variational Bayesian Moving Horizon Estimation, TSP 2025](https://doi.org/10.1109/TSP.2025.3572912) | 출판사 초록 | Bernoulli-Gaussian outlier model과 noise covariance를 moving horizon에서 함께 추정한다. robust observation의 근거지만 window·prior·추론 복잡도가 추가되어 현재 작은 수정의 우선순위는 낮다. |

### 현재 검증 후보: 상관된 관측 오차

상태를 `[p, v, a, b]`로 두고 `z = p + b + e`를 관측식으로 사용한다. `p,v,a`의 CV/CA/ZV 모델과 IMM interaction·mode evidence·moment matching은 유지한다. `b`는 DP 위치 오차의 지속 성분이고 `e`는 기존 분산 R=5의 white component다.

악보 pitch-class support 구간의 길이를 L reference frames라고 하면, `Var(b)=L²/12`, `rho=exp(-1/L)`, `b_next=rho*b+eta`, `Var(eta)=Var(b)*(1-rho²)`로 설정했다. L=0이면 rho와 bias process variance는 0이다. 이 scale 지정은 악보로 얻은 **가정**이지 문헌에서 유일하게 도출되는 값이 아니다. 같은 pitch class 구간을 합쳐도 원래 음향 특징이나 DP의 attack 정보는 삭제되지 않는다. 여기서는 그 구간을 독립적인 위치 오차의 분산 대신 지속 오차의 scale로 사용한다.

validation 결과는 다음과 같다. 모든 오차는 tracked-only이며 정렬율은 기존 평가기의 `≤` 정의를 따른다.

| 방법 | TR | Mean error | MedAE | ≤0.5b | ≤1b |
|---|---:|---:|---:|---:|---:|
| SoftOLTW | **95% (19/20)** | 0.2306 | 0.1000 | 89.01% | 95.64% |
| **IMM — 상관 관측 오차** | **95% (19/20)** | **0.2155** | **0.0757** | **89.67%** | **95.65%** |

1b 차이는 0.01 percentage point로 매우 작다. 통계적 유의성을 주장하지 않는다. 500-step covariance·probability 검사와 scalar augmented-KF 대조 검사를 통과했다. 전체 평가 전 소스와 선택 이유를 `output/imm_colored_full_146_20260920/protocol`에 고정했다. 이 validation 결과만으로 전체 성능 목표 달성을 판단하지 않았다. 이후 고정된 구현의 전체 평가에서 130/146을 확인했다.

이전 관측 정보량 후보는 두 정렬율이 baseline보다 낮았는데도 MAE 감소만 보고 전체 평가로 보낸 판단이 잘못이었다. 해당 실행은 65곡에서 중단했고, `output/imm_information_full_146_20260920/stopped.json`과 `partial_comparison_tracked.json`에 기록했다. 이후 전체 평가 진입에는 TR 유지·개선, 낮은 mean error, 높은 두 정렬율을 함께 요구한다.

## 130곡 버전 고정 이후의 목표

구현 저장소의 `faf4f33`을 다음 실험의 기준으로 고정한다. 전체 146곡에서 133곡 이상 성공하면서 tracked-only mean beat error 및 두 정렬율의 개선을 확인하는 것이 다음 목표다. validation 20곡에서만 모델과 parameter를 선택하며 전체 평가 결과로 수치를 조절하지 않는다.

map matching의 후보 경로 유지 및 재진입 구조는 검토할 가치가 있다. 현재 IMM은 DP의 단일 위치 출력을 관측하므로 잘못된 DP 경로 자체를 복구하지는 않는다. 다만 Murphy 등의 독립 off-road KF와 달리 이 KF는 DP 관측에 의존한다. 이를 독립 추정기로 간주해 DP와 다시 결합하면 같은 음향 증거를 중복 사용한다. 다음 실험에서는 후보 경로의 관측 likelihood와 tempo transition을 구분하고, 잘못된 후보가 나머지 후보를 오염시키지 않는 최소 구조를 우선 검토한다.

이 항목은 다음 실험의 설계 방향이며 구현 또는 성능 향상이 검증되었다는 뜻은 아니다. 기존 colored-error IMM을 대조군으로 유지하고, 실패했던 단순 acoustic mixture 및 feedback 실험과 구별되는 관측 모델을 먼저 명시한다.

## 직접적인 KF–DP 결합 선행연구 추가 조사

2026-09-20, 130곡 버전 커밋 이후 조사. DP는 DTW뿐 아니라 Viterbi 경로 탐색도 포함하지만, 두 알고리즘을 같은 것으로 취급하지 않는다. 아래의 적용 판단은 우리의 설계 제안이며 해당 논문이 score following에서 이를 검증한 것은 아니다.

| 연구 | 결합 방식 | 확인 범위와 현재 구현에 대한 의미 |
|---|---|---|
| [Jang, 2021, The Optical Tracking Method of Flight Target using Kalman Filter with DTW](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002736087) | 기준 비행 궤적과 실제 관측 궤적을 DTW로 정렬하고, 그 결과에서 얻은 각속도 등을 KF 관측으로 사용한다. | KCI에 등록된 저자 초록 확인, 수식 원문 미확보. 기준 악보와 연주를 정렬한 결과를 KF에 넣는 현재 구조와 직접적인 유사성이 있다. 관측 상관 처리까지 같다고 주장할 수는 없다. DOI 10.12673/jant.2021.25.3.217. |
| [Huang, Xue & Guo, 2012, Penalty Dynamic Programming Algorithm for Dim Targets Detection in Sensor Systems](https://pmc.ncbi.nlm.nih.gov/articles/PMC3355457/) | 추적 추정값으로 DP merit function의 penalty를 구성한다. 원문의 tracking block은 IMM 및 PDA/MHT를 사용한다. | 논문 본문 검색 텍스트와 저자 공개 원문 Section 4.2 확인. 직접 PDF 접근 실패. IMM과 DP의 피드백 결합 자체에 선행 사례가 있다. 여러 표적의 association과 추가 threshold까지 현재 문제에 가져올 필요는 없다. |
| [Pavlovic, Rehg & MacCormick, CVPR 2000, Impact of Dynamic Model Learning on Classification of Human Motion](https://users.cecs.anu.edu.au/~hartley/LearningPapers/Learning/Pavlovic-CVPR00.pdf) | SLDS의 approximate Viterbi에서 전이 후보마다 Kalman prediction/update를 계산한다. innovation likelihood와 discrete transition probability로 경로 점수를 갱신하고, 현재 상태별 최선 predecessor의 mean/covariance를 보존한다. | 원문 Section 3.1, 식 4와 의사코드 확인. 후보별 연속 상태를 유지하는 직접적인 구현 패턴이다. 상태별 하나의 survivor를 남기는 것은 근사이며 전역 최적성을 보장하지 않는다. 원문의 최종 backtracking/RTS smoothing은 온라인 평가에 그대로 사용할 수 없다. |
| [Jiang & Raphael, ISMIR 2020, Score Following with Hidden Tempo Using a Switching State-Space Model](https://archives.ismir.net/ismir2020/paper/000159.pdf) | note-index/age 경로마다 Gaussian tempo를 유지한다. stay/advance 확률은 tempo와 duration에서 계산하고, advance 때만 해당 경로의 KF를 업데이트한다. 음향 likelihood는 경로 확률에 곱한다. | 원문 Sections 3–4 재확인. DTW 후처리가 아니라 hybrid state-space의 경로 추론이다. 현재 SKF의 장점과 연결되는 가장 직접적인 score-following 근거다. 단일 전역 tempo를 모든 후보에 공유하지 않는 점이 중요하다. |
| [A Markov-Switching Model Approach to Heart Sound Segmentation and Classification, 2018 preprint](https://arxiv.org/pdf/1809.03395) | SKF 출력과 duration-dependent Viterbi를 결합해 상태 전환과 체류시간을 추론한다. | 원문 식 8, Algorithm 3 확인. duration prior의 유용한 관련 사례지만 offline backtracking을 사용한다. 누적 관측을 사용한 SKF posterior를 연속 emission처럼 곱하는 구조여서, 그대로 이식하면 증거 중복 문제를 해결했다는 근거가 되지 않는다. |
| [Murphy, Pao & Yuen, 2019 v2, Map matching when the map is wrong](https://arxiv.org/html/1809.09755v2) | 독립적인 off-road KF가 on-road HMM 후보를 생성할 수 있는 semi-interacting 구조다. | 원문 Section 3.1 확인. 우리 KF는 DP 결과를 관측하므로 이 논문의 독립 fallback 조건을 만족하지 않는다. 단순히 DP와 KF 출력을 혼합하는 구현의 근거로 쓰면 안 된다. |

추가 서지 후보: Yue 등의 ICCCAS 2010 논문 `A Kalman filtering-based dynamic programming track-before-detect algorithm for turn target` (DOI 10.1109/ICCCAS.2010.5581958)은 KF prediction으로 DP transition step을 조절한다고 저자 초록에 기술한다. IEEE 직접 본문은 확보하지 못했으므로 세부 수식의 구현 근거로 사용하지 않는다. 2024년 [Wu 등의 RD-plane DP-TBD 논문](https://www.mdpi.com/2072-4292/16/14/2639) 참고문헌에서 관련 2010·2016·2019 연구의 서지를 확인했다. 이 2024 논문 자체를 KF–DP 결합 연구라고 단정하지 않는다.

### 다음 실험 설계에 주는 구체적인 영향

현재 구조는 단일 DP 위치를 상관 오차 IMM에 입력한다. 후보 경로가 틀렸을 때 다른 후보로 돌아갈 수 있는 구조는 별도로 필요하다. 우선순위는 Pavlovic의 후보별 Kalman 상태와 Jiang–Raphael의 경로 조건부 tempo를 참고해, 음향 경로 후보와 그 후보에 조건화된 tempo 분포를 함께 유지하는 방식이다. 동일한 악보 위치에 도착해도 tempo 이력이 다르면 미래의 전이 분포가 다르므로 위치만으로 후보를 합치는 것은 근사다.

피드백 자체가 확률적으로 잘못된 것은 아니다. 과거 관측에서 얻은 prediction과 현재 프레임의 likelihood를 결합하는 것은 정상적인 Bayesian filtering이다. 문제가 되는 것은 현재 관측으로 이미 선택·보정한 결과를 다시 독립 관측으로 넣거나, 누적 DP 비용을 매 프레임의 새로운 likelihood로 해석하는 경우다. 후보의 이전 점수, 현재 음향 likelihood, 해당 후보의 tempo transition을 분리해 각 항이 한 번씩 사용되도록 설계해야 한다.

기존 softmin recurrence를 Viterbi max로 바꾸면 경로 합산과 최선 경로 선택이라는 추론 목표까지 달라진다. 최소 변경 실험에서도 이 차이를 명시하고, Kalman 상태를 soft mixture로 합칠 경우 between-mean covariance를 포함해야 한다. 상관 오차 상태는 DP-derived position 관측을 계속 쓸 때 유지할 근거가 있으며, raw acoustic likelihood로 관측 모델을 바꾸면 자동으로 같은 오차 모델을 붙이지 않는다.

이 조사 시점에는 새 구현이나 benchmark 결과가 없었다. 기준 커밋은 matchmaker `faf4f33`, benchmark `0938290`이다. 다음 목표는 133/146 이상이며 모델 선택은 validation 20곡으로 제한한다.


## KF–DP 조사 이후 구현 및 검증 결과

Pavlovic의 경로별 연속 상태 유지와 Jiang–Raphael의 경로 조건부 tempo를 참고해 `KalmanPathLattice`를 구현했다. 위치 후보마다 등속도 KF를 유지하고, 들어오는 경로별 prediction likelihood와 acoustic cost로 soft weight를 계산한다. 각 Kalman posterior를 between-mean covariance까지 포함해 Gaussian 하나로 합친다. 이는 표준 IMM의 운동 모델 혼합과 구별되는 경로 가설 혼합이며, 원문의 코드를 직접 이식한 것은 아니다. 최종 출력에는 기존 CV/CA/ZV 상관 관측 오차 IMM이 이어진다.

처음의 hard-survivor 및 단일 emission 방식은 validation 중단 기준에 걸렸다. 통과한 구성은 지나간 악보 프레임의 acoustic cost와 DTW 길이 정규화를 보존한다. 따라서 개선은 단순히 KF를 추가한 효과가 아니라 경로 추론 구성과 결합된 결과다. score-clock 관측 상관 후보는 validation에서 미세한 개선 뒤 full 평가 66곡에서 손실을 보여 중단했고 채택하지 않았다. 어떤 full 결과로도 숫자 parameter를 조정하지 않았다.

선택된 모델은 validation 19/20, beat MAE 0.1994, MedAE 0.0731, ≤0.5b 90.30%, ≤1b 95.92%였다. 고정 후 full146에서 134/146, 0.2680, 0.0840, 88.13%, 95.00%를 얻었다. SoftOLTW는 127/146, 0.2938, 0.1000, 86.57%, 94.33%다. 기존 checkpoint는 실험 이력으로 보관하며 원고의 주 비교는 SoftOLTW와 최종 방법으로 한다. 원고에는 경로 추론의 변경도 명시해야 하며, 모든 개선을 출력 IMM만의 효과로 해석하지 않는다.
