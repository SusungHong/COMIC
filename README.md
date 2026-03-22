# COMIC: Agentic Sketch Comedy Generation
[![Project Page](https://img.shields.io/badge/Project_Page-Visit-blue)](https://susunghong.github.io/COMIC)
[![arXiv](https://img.shields.io/badge/arXiv-2603.11048-b31b1b)](https://arxiv.org/abs/2603.11048)

## :clapper: Comedy Video Results

![Results](https://arxiv.org/html/2603.11048v1/x3.png)

See our [project page](https://susunghong.github.io/COMIC) for video results.

## :bar_chart: Comedy Evaluation

Automated evaluation comparing method videos against YouTube reference videos using critics aligned with viewer engagement patterns. Reports win rate and global-normalized inter/intra diversity per critic set (global best, channel best) and combined. Please see the paper for details.

### :wrench: Setup

```bash
pip install -r video_eval/requirements.txt
```

Set your Gemini API key via environment variable or a keys file:
```bash
export GEMINI_API_KEY=your_key_here
# or place keys (one per line) in video_eval/gemini_api_keys.txt
```

### :file_folder: Data Structure

```
data/
  test/
    middle.txt          # reference video IDs + YouTube URLs (50 videos, 5 channels)
  critics/
    global_best.json    # 1 critic evaluating all channels
    channel_best.json   # 5 channel-specialized critics
  videos/
    ours/*.mp4          # generated videos per method (.mp4)
    veo/*.mp4
    sora/*.mp4
    vgot/*.mp4
    ma/*.mp4
```

### :rocket: Usage

```bash
python video_eval/evaluate_videos.py
```

### :outbox_tray: Output

Results are saved to `video_eval/evaluations/`:
- **CSV**: per-critic per-channel win/loss data
- **JSON**: aggregated metrics (win rate, global-normalized inter/intra diversity)

### :art: Assets

All character and background assets used in the paper are included in `data/assets/`. For the evaluation shown in the paper, we generate the videos in a single pass using the character specification (`data/assets/characters/character_Haedol.json`).

## :memo: TODO

- [x] Comedy video results
- [x] YouTube-aligned comedy critics
- [x] Automated evaluation code
- [ ] Inference code

## :scroll: Citation

```bibtex
@article{hong2026comic,
  title={COMIC: Agentic Sketch Comedy Generation},
  author={Hong, Susung and Curless, Brian and Kemelmacher-Shlizerman, Ira and Seitz, Steve},
  journal={arXiv preprint arXiv:2603.11048},
  year={2026}
}
```
