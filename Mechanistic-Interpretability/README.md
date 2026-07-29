# Mechanistic Interpretability for IVModel

Interpretability tooling for the `IVModel` trained in this repo (`gpt2/iv_model.py`) — a GPT-2-architecture
causal decoder repurposed for numeric options-implied-volatility time-series forecasting instead
of text. Applied to the two trained checkpoints on the Hugging Face Hub:
[`Pratham007xo/iv-forecast-constituent-124m`](https://huggingface.co/Pratham007xo/iv-forecast-constituent-124m)
(per-stock next-day IV forecasting) and
[`Pratham007xo/iv-forecast-index-124m`](https://huggingface.co/Pratham007xo/iv-forecast-index-124m)
(synthetic equal-weighted S&P 500 index-level forecasting).

This toolkit was adapted from an earlier GPT-2/TransformerLens-based mechanistic-interpretability
project. Since `IVModel` has no vocab or embedding table (continuous feature vectors in,
continuous regression output out — no tokenizer at all), it can't be loaded via
[TransformerLens](https://transformerlensorg.github.io/TransformerLens/)'s `from_pretrained`
registry, which only supports a fixed set of known text architectures. Instead, this toolkit
hooks directly into the real, unmodified `IVModel`/`GPTDecoderBlock` code (see `hooks.py`),
guaranteeing numerical fidelity to the actual deployed model.

## Analyses

- **Attention pattern visualization** (`visualize_attention.py`) — heatmaps of what each
  attention head attends to, for a given input window.
- **QK circuit analysis** (`qk_circuit_analysis.py`) — query-key interaction strength per
  layer/head, i.e. how much each day's query "matches" each other day's key, independent of the
  learned value vectors.
- **Fixed-lag attention detection** (`fixed_lag_detection.py`) — GPT-2's "induction heads"
  detect literal token repetition (`A B ... A -> B`), which has no direct analog in continuous
  time-series data. This is reframed as: does a given head consistently attend to a *fixed*
  earlier timestep (e.g. "yesterday", "5 trading days ago") more than a uniform-attention
  baseline would predict? That's a genuinely meaningful question for volatility data — it
  reveals whether the model learned lag- or seasonality-based attention patterns. The original
  tool's fake evaluation against hardcoded dummy labels has been dropped entirely (there's no
  ground truth for this); it's replaced with a principled uniform-causal-attention baseline
  comparison instead.

## Usage

```bash
pip install -r requirements.txt
```

```python
from load_iv_model import load_iv_checkpoint
from sample_windows import load_sample_windows
from hooks import run_with_cache, verify_hooks
from visualize_attention import plot_attention_heatmap_static
from qk_circuit_analysis import plot_qk_interactions_static
from fixed_lag_detection import compute_lag_profile, plot_lag_heatmap

model, scaler = load_iv_checkpoint('Pratham007xo/iv-forecast-constituent-124m')
windows = load_sample_windows('options_iv_data/valid.npz', n=200)  # produced by scripts/prepare_options_iv.py
features = torch.from_numpy(windows).float()

verify_hooks(model, features[:4])  # sanity-check the hooks against this checkpoint
plot_attention_heatmap_static(model, features[:1], layer=0, head=0)
plot_qk_interactions_static(model, features[:1], layer=0)
lag_df = compute_lag_profile(model, features)
plot_lag_heatmap(lag_df)
```

See `iv_interpretability_colab.ipynb` for a full, runnable walkthrough on both checkpoints
(clones this repo, prepares the datasets, downloads both checkpoints, and runs all three
analyses).

## Known limitation: timestep labels

`train.npz`/`valid.npz` (produced by `scripts/prepare_options_iv.py` /
`prepare_options_iv_index.py`) don't retain per-window date/ticker metadata, so heatmap axes are
currently labeled with relative day-offsets (`t-31 ... t-1, t`) rather than actual calendar
dates or ticker symbols. Enhancing the data-prep scripts to emit a parallel metadata file would
allow richer, dated visualizations — left as a follow-up, out of scope here.

## License

See `LICENSE`.
