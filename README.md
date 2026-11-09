# Financial LLM Fine-Tuning & Adaptation

QLoRA fine-tuning of **Qwen2.5-1.5B-Instruct** on financial Q&A data (LoRA via PEFT, 4-bit NF4 via BitsAndBytes, TRL SFTTrainer), deployed as a Gradio app on Hugging Face Spaces.

> The final README with real metrics, loss curve and live links is generated automatically by the notebook after training. See `notebooks/finance_llm_finetune.ipynb`.

## Structure
```
notebooks/finance_llm_finetune.ipynb   # one-click: train, push adapter, deploy Space, generate results
app/                                   # Gradio app deployed to Hugging Face Spaces
outputs/                               # loss curve and metrics
```
