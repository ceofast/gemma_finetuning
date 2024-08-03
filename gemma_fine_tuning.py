import torch
from trl import SFTTrainer
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments, TextStreamer

# Model configuration
max_seq_length = 2048
dtype = None
load_in_4_bit = True


# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
  model_name="unsloth/gemma-2b-2b",
  max_seq_length=max_seq_length,
  dtype=dtype,
  load_in_4bit=load_in_4bit,
)


# Configure the PEFT model
model = FastLanguageModel.get_peft_model(
  model,
  r=16,
  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
  lora_alpha=16,
  lora_droupout=0,
  bias="none",
  use_gradient_checkpointing="unsloth",
  random_state=3047,
  use_rslora=False,
  loftq_config=None,
)

# Define prompt formatting function
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
  instructions = examples["instruction"]
  inputs = examples["input"]
  outputs = examples["output"]
  texts = []
  for instruction, input, output in zip(instructions, inputs, outputs):
    text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
    texts.append(text)
  return {"text": texts}


# Load and process dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)


# Setup training arguments and trainer
trainer = SFTTrainer(
  model=model,
  tokenizer=tokenizer,
  train_dataset=dataset,
  dataset_text_field="text",
  max_seq_length=max_seq_length,
  dataset_num_proc=2,
  packing=False,
  args=TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_step=4,
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    fp16=not FastLanguageModel.is_bfloat16_supported(),
    bf16=FastLanguageModel.is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
  ),
)


# Train the model
trainer_stats = trainer.train()


# Inference example
FastLanguageModel.for_inference(model)
inputs = tokenizer(
  [
    alpaca_prompt.format(
      "Continune the fibonacci sequence.",
      "1, 1, 2, 3, 5, 8",
      "",
    )
  ],
  return_tensors="pt",
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
print(tokenizer.batch_decode(outputs))


# Inference with streaming
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)

# Save the model and tokenizer
model.save_pretrained("lora_model")
tokenizer.save_pretrainer("lora_model")
