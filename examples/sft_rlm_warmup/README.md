# OpenEnv REPL warmup on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jucamohedano/trl/blob/sft-rlm-warmup/examples/sft_rlm_warmup/sft_rlm_warmup.ipynb)

Open [the notebook](sft_rlm_warmup.ipynb) in Colab and select a T4 GPU runtime
when one is available. Upload `laguna-alphaxiv-selective.train.jsonl` and
`laguna-alphaxiv-selective.validation.jsonl` when prompted. The default run
checks both splits against an 8,192-token limit, performs one QLoRA update on
Qwen3.5-4B using the longest training conversation, and saves a downloadable
adapter. Set `SMOKE_RUN = False` to train a full epoch and evaluate the
validation split. Measuring HotpotQA transfer uses a separate held-out sample.

The Qwen3.5-4B smoke run applied one optimizer update on a Tesla T4 using a
5,668-token training conversation. The loss was `0.686594`, with `9.48 GiB`
peak allocated and `9.96 GiB` peak reserved GPU memory. The longest validation
conversation was 7,500 tokens. The smoke run checked validation tokenization
and labels. The full run evaluates validation loss.

To regenerate the smoke input from the published datasets, use an environment
with `pyarrow`, `smolagents`, and OpenEnv installed. Set `OPENENV_CHECKOUT` to
the OpenEnv checkout that provides `envs/repl_env`. The task parquet files
take about 512 MiB to download.

```bash
hf download alphaXiv/sft-traces-arxivqa-multi hf_dataset.jsonl --type dataset \
  --revision 0d33e91d6d420395a77472c313949cc124964a2d --local-dir /tmp/laguna-alphaxiv-multi
hf download alphaXiv/multi-paper-v1 --type dataset --include 'data/train-*.parquet' \
  --revision c438622b1311a16db1c4f366b4998b3b8e4fbe11 --local-dir /tmp/laguna-alphaxiv-tasks
PYTHONPATH="$OPENENV_CHECKOUT/src:$OPENENV_CHECKOUT/envs" \
  python examples/sft_rlm_warmup/replay_alphaxiv_parents.py \
  --traces /tmp/laguna-alphaxiv-multi/hf_dataset.jsonl \
  --tasks /tmp/laguna-alphaxiv-tasks/data --output /tmp/laguna-alphaxiv-replayed.jsonl
```

By default the replay adapts two [public AlphaXiv parent traces](https://huggingface.co/datasets/alphaXiv/sft-traces-arxivqa-multi),
executes the translated code in OpenEnv, and supplies scripted child replies
from the verified evidence. The exported JSONL contains the parent messages.

## Audit the full parent release

Run the converter in a TRL environment with PyArrow and the same OpenEnv
checkout. Download the tokenizer and config files without model weights.

```bash
hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --include '*.json' --include '*.jinja' --exclude '*.index.json' \
  --local-dir /tmp/laguna-warmup-qwen35-4b-tokenizer
PYTHONPATH="$OPENENV_CHECKOUT/src:$OPENENV_CHECKOUT/envs" \
  python examples/sft_rlm_warmup/replay_alphaxiv_parents.py --all \
  --traces /tmp/laguna-alphaxiv-multi/hf_dataset.jsonl \
  --tasks /tmp/laguna-alphaxiv-tasks/data \
  --tokenizer /tmp/laguna-warmup-qwen35-4b-tokenizer \
  --output /tmp/laguna-alphaxiv-selective.jsonl
```

The converter writes a per-trajectory report and separate train and validation
JSONL files. It keeps each source-paper group together. Replay success and
reference evidence coverage are recorded separately. Children return scripted
reference passages from the papers they receive, and a direct teacher answer
is flagged if its excerpts cannot be located verbatim in the public papers.
Token counts use TRL's training template, and the converter verifies that the
assistant loss mask covers the parent code turns. Check the report before
training. A successful replay can still contain partial evidence or exceed
the notebook's sequence limit.
