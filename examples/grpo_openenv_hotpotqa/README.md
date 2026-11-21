# Train an RLM on HotpotQA with GRPO

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jucamohedano/trl/blob/grpo-openenv-hotpotqa/examples/grpo_openenv_hotpotqa/grpo_openenv_hotpotqa.ipynb)

The [notebook](grpo_openenv_hotpotqa.ipynb) starts from Daniel Kim and Rehaan
Ahmad’s [public Qwen3.5-4B SFT checkpoint](https://huggingface.co/alphaXiv/evidence-multi-rlm-sft-4b).
The root reads candidate Wikipedia articles in an OpenEnv Python REPL and can
delegate evidence checks to child REPLs backed by frozen Qwen3.5-4B. GRPO trains
a LoRA adapter on the root using normalized exact match on its final answer.
The prompts adapt the evidence-selection workflow described in the authors’
[research blog post](https://www.alphaxiv.org/blog/reinforcement-learning-for-rlms)
to multi-hop HotpotQA questions.

## Run in Colab

Select a T4 runtime and run the cells in order. Set `QUICK_RUN = True` in the
first code cell to train on one question with four rollouts and evaluate on one
held-out question. The full run uses 20 training and 20 held-out questions. The
saved run took about 92 minutes for training, with evaluation afterward.

The notebook shows the training curves in Trackio and saves the LoRA adapter
in the run directory. To also upload it, set `HUB_MODEL_ID` to
`your-username/your-model` and enter a Hugging Face token with write access.

The saved notebook contains 10 paired held-out results. Base Qwen3.5-4B scored
**4/10** and the SFT + GRPO root scored **7/10** under the same REPL harness.
