# OpenEnv REPL warmup on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jucamohedano/trl/blob/sft-rlm-warmup/examples/sft_rlm_warmup/sft_rlm_warmup.ipynb)

Open [the notebook](sft_rlm_warmup.ipynb) in Colab and select a T4 GPU runtime
when one is available. Upload `laguna-alphaxiv-replayed.jsonl` when prompted.
The default run checks assistant-only loss, performs one LoRA update on
Qwen3-0.6B, and saves a downloadable adapter. The two input conversations
exercise the training path. A HotpotQA evaluation requires a separate,
held-out sample and a warmup dataset large enough to measure transfer.

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

The replay adapts two [public AlphaXiv parent traces](https://huggingface.co/datasets/alphaXiv/sft-traces-arxivqa-multi),
executes the translated code in OpenEnv, and supplies scripted child replies
from the verified evidence. The exported JSONL contains the parent messages.
