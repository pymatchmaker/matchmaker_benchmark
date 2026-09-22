# Table 1 SoftOLTW + IMM (135/146) — 소스 커밋 기록 (2026-09-21)

## 결론

**`matchmaker-laurenceyoon` 커밋 `fd517737bc58eec1421cb96fcf9577996c3dd7bd`(HEAD, 2026-09-21 20:55, "Fix repeat boundary transitions in hierarchical score following")의 `--method soft_oltw` 기본값이 원고 Table 1의 SoftOLTW + IMM 행(135/146, MeanAE 0.2555 b)을 생성한 모델이다.**

`soft_oltw` 경로에 관여하는 파일(`dp/oltw_soft.py`, `dp/oltw_imm.py`, `prob/imm.py`, `methods.yaml`, `registry.py`, `matchmaker.py`, `features/audio.py`)은 `54af681`(19:25)과 `fd51773`에서 동일하므로, 두 커밋은 이 목적상 등가다. `fd51773`은 `oltw_hierarchical.py`와 테스트만 바꿨다.

## 생성 계보

| 단계 | 소스 | 비고 |
|---|---|---|
| Table 1 경로 생성 | 고정 패키지 `/tmp/imm-joint-path-20260921` (base `9aa221b` + `matchmaker/prob/imm_path.py` sha256 `ee9a2aa9…`, `methods.yaml` `e11fc51a…`), 메서드 이름 `joint_imm` | `output/imm_acoustic_redesign_20260921/joint_path/{asap,batik,vienna}` → `output/table1_joint_imm_20260921/softoltw`가 심링크 |
| 저장소 통합 | `dc96e3d` (15:20) `joint_imm` as `dp/oltw_imm.py` | 150 synthetic step 및 두 예제에서 고정 후보와 비트 동일 (`joint_path/integration_equivalence.json`) |
| 기본 모델로 승격 | `63089b9` (15:52) `softoltw` = `IMMOnlineTimeWarping` | 구 `oltw_soft`/`softoltw_no_imm` 스펙 제거 |
| 이름 확정 | `54af681` (19:25) `soft_oltw` = `dp:SoftOnlineTimeWarping` | IMM 팔로워 통합 |
| 현재 HEAD | `fd51773` (20:55) | hierarchical만 변경 |

## 재현 증거

1. `output/final_imm_hierarchical_reproduction_20260921` (17:52 시작): 146곡 전부 경로·정답이 Table 1 경로와 원소 단위 동일 (`verification.json: linear_exact_trajectories = 146`). 당시 작업 트리는 `63089b9` + `54af681`로 향하는 미커밋 변경 상태.
2. HEAD `fd51773` 직접 실행 (2026-09-21 22:00 경, 이 문서 작성 시점): Existing IMM ↔ Joint IMM 사이에 결과가 바뀐 결정적 5곡 **ASAP 17, Vienna 27, 28, 37, 75** + Vienna 1 모두 Table 1 `wp_*.tsv`와 **바이트 동일**.
3. HEAD `fd51773` 146곡 전체 soft_oltw 재실행 (2026-09-21 21:58–22:33, `--workers 4 --no-plots`): `output/softoltw_head_fd51773_146_20260921/` — **146/146 `wp_*.tsv` 바이트 동일, 135/146 추적 성공** (`verification.json`).

## 참고

- 벤치마크 실행 명령: `HOMEBREW_PREFIX=/opt/homebrew python matchmaker_eval/test_audio.py --dataset {vienna,asap,batik} --method soft_oltw --no-plots --workers N --output-dir …`
  (`HOMEBREW_PREFIX`는 pyfluidsynth가 `libfluidsynth.dylib`를 찾는 데 필요. 대화형 zsh에는 설정되어 있으나 비대화형 셸에는 없음.)
- 같은 날 `hierarchical_soft_oltw`를 146곡 unfolded 악보에 돌린 결과: **75/146** (`output/hierarchical_linear_146_20260921`). 선형 악보에서 soft_oltw와 동일해야 하는 래퍼가 9초 이후부터 앞서 달림(racing). soft_oltw 자체는 무관.
