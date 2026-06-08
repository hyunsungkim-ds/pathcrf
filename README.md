<div align="center">
	<h1>
		PathCRF
	</h1>
</div>

Source code for the paper [PathCRF: Ball-Free Sports Event Detection via Possession Path Inference from Player Trajectories](https://arxiv.org/abs/2602.12080) by Hyunsung Kim et al., [KDD 2026](https://kdd2026.kdd.org).

## Introduction
**PathCRF** is a framework for detecting on-ball events solely from player trajectories in soccer. It models player trajectories as a fully connected dynamic graph and formulates event detection as the problem of selecting exactly one edge corresponding to the current possession state at each time step. It encodes dynamic interactions between players over time via a Set Attention-based backbone, while a Conditional Random Field (CRF) enforces logical consistency across the resulting edge sequence.
<p align="center">
  <img src="docs/overview.png" />
</p>

The animation below shows the PathCRF output applied to an action sequence, where the framework infers ball possession states per time step from player trajectories alone. At each moment, PathCRF predicts that either a player is in possession (highlighted), or the ball is traveling from one player to another (shown as an arrow between them). Whenever this state changes, the framework detects an on-ball event and logs it to the table on the right.

For reference, we overlay the actual ball trajectory as a semi-transparent path, so you can see how closely the predicted possession states track the real ball movement, even though the framework has no access to it.
<p align="center">
  <img src="docs/J03WR9_EP002_T140.gif" width="800" />
</p>

## Data Preparation and Preprocessing
- Uses [Sportec Open DFL Dataset (Bassek et al., 2025)](https://www.nature.com/articles/s41597-025-04505-y) and [kloppy](https://kloppy.pysport.org) package.
- First, to download and synchronize the event and tracking data, follow `tutorial.ipynb` of [ELASTIC (Kim et al., 2025)](https://github.com/hyunsungkim-ds/elastic.git). This will create the synchronized event and tracking data files into the designative paths.
- Place the synchronized event and tracking data files into `data/sportec/event_synced` and `data/sportec/tracking_parquet`, respectively.
- Running `python datatools/preprocess.py` merges event-based ground-truth possession into tracking data and saves the preprocessed result into `tracking_processed`. Ground-truth event data is saved in `event_processed`.

```bash
python datatools/preprocess.py
```

## Model Training
For reproducibility, this repository already contains the trained models listed in Section 3.2 as follows:
- Non-CRF: `saved/100`
- Static Dense CRF: `saved/110`
- Static Masked CRF: `saved/120`
- Dynamic Dense CRF: `saved/130`
- Dynamic Masked CRF: `saved/140`

To train models on your own, run `*.sh` files in `scripts` (e.g., `bash scripts/ballradar_crf.sh`). Be sure to change `--trial` in the files to avoid overwriting.

## Model Inference and Evaluation
Follow `tutorial.ipynb` step by step to reproduce inference and evaluation.

Example outputs:
### Event recall under varying tolerance thresholds (Fig. 6)
![Event recall](docs/event_recall.png)

### Ground-truth and model-detected event sequences (Fig. 2)
<p align="left">
  <img src="docs/events_true.png" width="100%" />
  <img src="docs/events_140.png" width="98%" />
</p>

## Practical Applications
Following `tutorial.ipynb`, you can reproduce visualizations comparing the downstream analysis metrics derived from ground-truth and model-detected events.

Example outputs:
### Team-level or player-level event heatmaps (Fig. 3)
![Team heatmaps](docs/heatmap_home.png)
![Player heatmaps](docs/heatmap_home_29.png)
  
### Timeline of the home team's possession shares (Fig. 4)
![Possession timeline](docs/home_poss.png)

### Pass networks (Fig. 5)
![Pass networks](docs/passmap_home.png)

## Citation
If you use this code in your research, please consider citing the following paper:
```
@inproceedings{kim2026pathcrf,
  author       = {Hyunsung Kim and
                  Kunhee Lee and
                  Sangwoo Seo and
                  Sang-Ki Ko and
                  Jinsung Yoon and
                  Chanyoung Park},
  title        = {{PathCRF}: Ball-Free Sports Event Detection via Possession Path Inference from Player Trajectories},
  booktitle    = {Proceedings of the 32nd {ACM} {SIGKDD} Conference on Knowledge Discovery and Data Mining},
  year         = {2026},
  doi          = {10.1145/3770855.3818152},
}
```