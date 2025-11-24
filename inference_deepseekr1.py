import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -------------------------------
# 1. Load base model and LoRA adapter
# -------------------------------

base_model = "deepseek-ai/deepseek-llm-7b-chat"
adapter_path = "./deepseek7bchat-lora-final"  # 👈 Your saved LoRA folder

print("🔧 Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16
)

print("🔗 Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter_path)

print("🔤 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

model.eval()

# -------------------------------
# 2. Inference Function
# -------------------------------

def chat(prompt: str, max_new_tokens=200):
    input_text = f"""You are a helpful assistant.

### User:
{prompt}

### Assistant:
"""
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    # Strip the prompt portion from the response
    return decoded.split("### Assistant:")[-1].strip()


# -------------------------------
# 3. Run a Sample Chat
# -------------------------------

user_input = "Explain how transformers work in simple terms."
response = chat(user_input)

print("\n🧠 Assistant:", response)
