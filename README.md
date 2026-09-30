# KhmerAI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rinabuoy/KhmerAI/blob/main/GemmaX_MiLMMT_Translation_Demo.ipynb)

A hands-on Google Colab demo of **multilingual machine translation with a focus on Khmer (ភាសាខ្មែរ)**, using the open translation LLMs from [xiaomi-research/gemmax](https://github.com/xiaomi-research/gemmax), which are built on Google's Gemma models.

| Family | Languages | Sizes | Base model |
|---|---|---|---|
| **MiLMMT-46** (latest, v1.0) | 46 | 1B, 4B, 12B | Gemma 3 |
| **GemmaX2-28** | 28 | 2B, 9B | Gemma 2 |

Both families support Khmer, as well as Thai, Lao, Vietnamese, Burmese and many other languages.

## What the notebook covers

1. GPU check and dependency install
2. Loading a MiLMMT / GemmaX2 checkpoint, with automatic choice of bf16, fp32 or 4-bit NF4 based on your GPU
3. The official translation prompt format
4. Single-sentence, one-to-many and batched document translation (e.g. English → Khmer)
5. A reference-free quality check: round-trip translation (EN → X → EN) scored with chrF
6. An interactive translator widget
7. *(Optional)* high-throughput inference with [vLLM](https://github.com/vllm-project/vllm)
8. *(Optional)* the upstream repo's training data formats (CPT / SFT / RL) and its SFT↔RL linear weight-merge script

## Quick start

1. Click the **Open in Colab** badge above.
2. In Colab, choose `Runtime → Change runtime type → GPU`.
3. Run the cells from top to bottom. The default model is `MiLMMT-46-1B-v1.0` (~2 GB download).

### Hardware guide

| Model | Free T4 (16 GB) | L4 / A100 |
|---|---|---|
| MiLMMT-46-1B | ✅ fp32 | ✅ bf16 |
| MiLMMT-46-4B | ✅ 4-bit | ✅ bf16 |
| MiLMMT-46-12B | ❌ | ✅ (4-bit on L4) |

Gemma 3 is numerically unstable in fp16, so on GPUs without bf16 support (such as the T4) the notebook uses fp32 for small models and 4-bit quantization for larger ones.

### Running locally

Any machine with a CUDA GPU works:

```bash
pip install "transformers>=4.50" accelerate bitsandbytes sacrebleu ipywidgets matplotlib jupyter
jupyter notebook GemmaX_MiLMMT_Translation_Demo.ipynb
```

Skip or adapt the Colab-specific bits (`!nvidia-smi`, `#@param` forms, `/content` paths).

## Prompt format

The models were trained on this exact template; use the language names exactly as the notebook lists them:

```text
Translate this from English to Khmer:
English: The weather in Phnom Penh is hot and humid today.
Khmer:
```

Minimal example with 🤗 Transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "xiaomi-research/MiLMMT-46-1B-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

prompt = "Translate this from English to Khmer:\nEnglish: Angkor Wat is a temple complex in Cambodia.\nKhmer:"
inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
```

## Troubleshooting

- **Empty or garbled output on a T4:** the model is probably running in fp16. Use fp32 or 4-bit instead, as the notebook does.
- **Output in the wrong language:** make sure the language name matches the list exactly (e.g. `Chinese (Simplified)` for MiLMMT, `Chinese` for GemmaX2).
- **Long inputs:** translate sentence by sentence or paragraph by paragraph; the models were trained on sentence-level pairs.
- **Out of memory:** enable `FORCE_4BIT`, lower `batch_size`, or pick a smaller model.
- **`*-Pretrain` checkpoints** are not translation models; use the `v0.1`, `v0.2` or `v1.0` releases.

## References

- Upstream repo: https://github.com/xiaomi-research/gemmax
- MiLMMT-v1.0: *Reference-Free Post-Training of Open LLMs for Multilingual MT* (arXiv:2608.10812)
- MiLMMT-v0.1: *Scaling Model and Data for Multilingual MT with Open LLMs* (arXiv:2602.11961)
- GemmaX2: *Multilingual MT with Open LLMs at Practical Scale* (NAACL 2025)

Model weights are subject to the licenses published with them on Hugging Face, including the Gemma Terms of Use.
