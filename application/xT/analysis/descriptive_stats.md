# xT Model Performance Report (J03WR9)

## Event Prediction Accuracy

### Pass/Control Event Metrics

| Threshold | Precision | Recall | F1-Score | GT Count | Pred Count |
| --------- | --------- | ------ | -------- | -------- | ---------- |
| 0.4s      | 0.4773    | 0.6350 | 0.5450   | 1693     | 2246       |
| 0.6s      | 0.5730    | 0.7620 | 0.6541   | 1693     | 2246       |

---

### GT Event Type별 상세 분석 (Threshold 0.6s)

#### Pass 계열 GT 타입

조건 제거로 인해 모든 Pass 계열 이벤트의 감지율이 크게 향상되었습니다.

| GT Type        | Total | Recall    | Matched | Precision | F1     | 비고          |
| -------------- | ----- | --------- | ------- | --------- | ------ | ------------- |
| **pass**       | 793   | **85.0%** | 674     | 42.4%     | 56.6%  |               |
| clearance      | 35    | **48.6%** | 17      | 58.6%     | 53.1%  | (기존 11%) 🔺 |
| corner_crossed | 6     | 100.0%    | 6       | 100.0%    | 100.0% |               |
| cross          | 25    | **72.0%** | 18      | 40.0%     | 51.4%  | (기존 20%) 🔺 |
| goalkick       | 16    | **81.2%** | 13      | 54.2%     | 65.0%  |               |
| shot           | 25    | **52.0%** | 13      | 61.9%     | 56.5%  | (기존 0%) 🔺  |
| throw_in       | 31    | 22.6%     | 7       | 20.6%     | 21.5%  |               |

#### Control 계열 GT 타입 (범위 확장)

| GT Type          | Total | Recall    | Matched | Precision | F1    | 비고      |
| ---------------- | ----- | --------- | ------- | --------- | ----- | --------- |
| **control**      | 606   | **77.6%** | 470     | 37.7%     | 50.7% |           |
| **bad_touch**    | 44    | **43.2%** | 19      | 38.3%     | 40.6% | (신규) ✅ |
| **dispossessed** | 13    | **30.8%** | 4       | 10.0%     | 15.1% | (신규) ✅ |
| interception     | 59    | 49.2%     | 29      | 38.7%     | 43.3% |           |
| tackle           | 7     | 0.0%      | 0       | 0.0%      | 0.0%  |           |

---

### 종합 메트릭

| 카테고리         | Total GT | Matched | Recall    | Precision | F1    |
| ---------------- | -------- | ------- | --------- | --------- | ----- |
| **Pass 계열**    | 933      | 751     | **80.5%** | 42.9%     | 56.0% |
| **Control 계열** | 730      | 522     | **71.5%** | 37.0%     | 48.8% |
| **전체**         | 1663     | 1273    | **76.5%** | 56.6%     | 65.1% |

> ✅ **최종 결론**: 모델은 실패한 패스(크로스 실패 등)나 실패한 컨트롤(터치 미스 등)도 **"동작 시도 자체"를 매우 잘 인식**하고 있습니다.  
> 전체 GT의 **76.5%**를 커버하는 높은 감지율을 보입니다.

---

## xT Comparison (Episode-Level)

Comparison of Total xT generated per episode (Actual vs Predicted).

- **Episodes Compared**: 91
- **Total xT (Actual)**: 7.8293
- **Total xT (Predicted)**: 6.9154
- **MAE per Episode**: 0.0410

---

## Feature Analysis & Improvements

### Top 3 Sequences (Final Corrected Logic)

- **Predicted Top 3 (Now Valid Attacking Plays)**:
  1.  **Away P2 (Rank 1)**: `pass` at Frame 114021 (Val 0.3042). _Valid Attacking Pass._
  2.  **Home P2 (Rank 2)**: `pass` at Frame 143907 (Val 0.2001). _Valid Attacking Pass._
  3.  **Home P2 (Rank 3)**: `pass` at Frame 143877 (Val 0.1398). _Valid Attacking Pass._

- **Actual Top 3 (Goal Filtered)**:
  - _Note: Previously high xT events (0.99, 0.88) were verified as Goals/Shots. These have been filtered out to show build-up play._
  1.  **Away P2 (Rank 1)**: `corner_crossed` (Val 0.3470). _Valid Set Piece Opportunity._
  2.  **Home P1 (Rank 2)**: `pass` at X=99.4 (Val 0.2549). _Valid Attacking Pass._
  3.  **Home P1 (Rank 3)**: `freekick_crossed` (Val 0.1338). _Valid Set Piece._

### xT Filtering and Player-Level Correlation

**Filtering Rules** (Applied during xT calculation):

- Events are **excluded** from xT calculation if:
  - Next event is **10+ seconds later**
  - Next event is in a **different episode**

**Player-Level Correlation** (Penalty Mode):

- **Correlation**: **0.6191**
- **N (Players)**: 31
- **MAE**: 0.2918
- **Interpretation**: The model captures individual player contributions reasonably well when aggregated across all valid (within-episode, <10s gap) events. Higher correlation than episode-level analysis indicates that player-level aggregation smooths out some of the timing/context mismatches seen in individual episodes.
