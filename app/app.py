import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
_here = os.path.dirname(os.path.abspath(__file__))
ADAPTER = os.environ.get("ADAPTER_REPO") or open(os.path.join(_here, "adapter_repo.txt")).read().strip()
SYSTEM = "You are a helpful financial assistant."

tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()
model.eval()


def answer(question):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


demo = gr.Interface(
    fn=answer,
    inputs=gr.Textbox(lines=4, label="Ask a finance question"),
    outputs=gr.Textbox(label="Answer"),
    title="Finance LLM (Qwen2.5-1.5B + LoRA, 4-bit QLoRA fine-tuned)",
    description="Educational demo only. Not financial advice. Running on free CPU, so answers take 20-60 seconds.",
    examples=[
        "What is the difference between stocks and bonds?",
        "How does compound interest work?",
        "What is a P/E ratio and why does it matter?",
    ],
)

if __name__ == "__main__":
    demo.launch()
