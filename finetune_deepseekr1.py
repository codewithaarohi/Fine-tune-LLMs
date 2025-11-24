import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Trainer, TrainingArguments 
from peft import LoraConfig, get_peft_model 
from datasets import load_dataset
from tqdm import tqdm  

# -------------------------------
# 1. Load Model and Tokenizer
# -------------------------------
model_name = "deepseek-ai/deepseek-llm-7b-chat"  

print("✅ Loading model and tokenizer...")
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

if hasattr(config, "quantization_config"):
    delattr(config, "quantization_config")


# Loads the 7B chat model on GPU 0, Uses float16 for memory efficiency, Loads the tokenizer too
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    device_map={"": "cuda:0"},
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# -------------------------------
# 2. Load and Preprocess Dataset
# -------------------------------
dataset = load_dataset("tatsu-lab/alpaca")["train"]

# Format chat into prompt + completion
def format_chat_batch(batch):
    results = []
    for instruction, response in zip(batch["instruction"], batch["output"]):
        prompt = f"""You are a helpful assistant.

### User:
{instruction}

### Assistant:
{response}
"""
        tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
        results.append(tokenized)
    return {
        "input_ids": [r["input_ids"] for r in results],
        "attention_mask": [r["attention_mask"] for r in results],
        "labels": [r["input_ids"] for r in results]
    }


# Applies the formatter to the entire dataset with multiprocessing
print("🔁 Tokenizing with progress...")
tokenized_dataset = dataset.map(
    format_chat_batch,
    batched=True,
    batch_size=32,
    num_proc=4,
    remove_columns=dataset.column_names,
    desc="🔄 Tokenizing "
)

# -------------------------------
# 3. Apply LoRA (PEFT)
# -------------------------------
'''LoRA modifies just a few layers:
q_proj, v_proj in the attention blocks
Reduces GPU memory + training time significantly'''

print("✅ Applying LoRA configuration...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Injects LoRA modules into the base model
model = get_peft_model(model, lora_config)

# -------------------------------
# 4. Training Setup
# -------------------------------
# Logs every 50 steps, saves every 500
# Uses fp16 training (faster + lighter)
training_args = TrainingArguments(
    output_dir="./deepseek7bchat-lora-alpaca",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    # evaluation_strategy="no",   ← REMOVE THIS
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# -------------------------------
# 5. Start Fine-Tuning
# -------------------------------

print("🚀 Starting fine-tuning...")
# trainer.train()  # fromscratch

trainer.train(resume_from_checkpoint="./deepseek7bchat-lora-alpaca/checkpoint-4500")


# -------------------------------
# 6. Save Fine-Tuned Model
# -------------------------------

print("💾 Saving the fine-tuned LoRA adapter...")

#Saves the LoRA adapter weights
trainer.save_model("./deepseek7bchat-lora--final")

#Saves the tokenizer files for future use
tokenizer.save_pretrained("./deepseek7bchat-lora--final")
print("✅ Training & saving complete!")

