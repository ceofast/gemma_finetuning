# gemma_finetuning

This script demonstrates how to fine-tune a pre-trained language model using the Unsloth framework. The script loads a model, configures it for training with LoRA (Low-Rank Adaptation), trains it on a custom dataset, and performs inference to generate text based on a given prompt. The model and tokenizer can be saved locally after training.

## Dependencies

Ensure you have the following Python libraries installed:
```bash
pip install torch trl datasets unsloth transformers
```

## Script Breakdown

1. **Importing Required Libraries**

```bash
import torch
from trl import SFTTrainer
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments, TextStreamer
```

This section imports the necessary libraries. torch is used for tensor operations, trl and transformers provide utilities for training and inference, datasets is used for loading and processing datasets, and unsloth is used for handling model operations.


2. **Model Configuration**

```bash
max_seq_length = 2048
dtype = None
load_in_4bit = True
```

Here, we define the model configuration parameters. `max_seq_length` sets the maximum sequence length for the model, `dtype` allows for specifying the data type (e.g., `float16` for mixed precision), and `load_in_4bit` enables loading the model in 4-bit precision to reduce memory usage.


3. **Loading the Model and Tokenizer**

```bash
model, tokenizer = FastLanguageModel.from_pretrained(
  model_name="unsloth/gemma-2b-2b",
  max_seq_length=max_seq_length,
  dtype=dtype,
  load_in_4bit=load_in_4bit,
)
```

This section loads a pre-trained model and its corresponding tokenizer using the `from_pretrained` method from the `FastLanguageModel` class in the `unsloth` library. The model is specified by its name, and the configuration parameters are passed during loading.


4. **Configuring the PEFT Model**

```bash
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
```

This section configures the model for Parameter-Efficient Fine-Tuning (PEFT) using LoRA. It specifies the LoRA parameters like rank (`r`), alpha (`lora_alpha`), and which modules to target for adaptation. Gradient checkpointing is enabled to save memory during training.


5. **Defining the Prompt Formatting Function**

```bash
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token
```

The `alpaca_prompt`defines a template for formatting training examples. The `EOS_TOKEN` is added to mark the end of a sequence.

```bash
def formatting_prompts_func(examples):
  instructions = examples["instruction"]
  inputs = examples["input"]
  outputs = examples["output"]
  texts = []
  for instruction, input, output in zip(instructions, inputs, outputs):
    text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
    texts.append(text)
  return {"text": texts}
```

The `formatting_prompts_func` function applies the `alpaca_prompt` template to each example in the dataset.


6. **Loading and Processing the Dataset**
```bash
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)
```

This section loads a dataset from the Hugging Face Hub and applies the `formatting_prompts_func` to each example to prepare the data for training.


7. **Setting Up Training Arguments and Trainer**

```bash
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
```

This section sets up the training arguments and initializes the `SFTTrainer` for fine-tuning. The training arguments control various aspects of the training process, including batch size, learning rate, and optimization strategy.


8. **Training the Model**

```bash
trainer_stats = trainer.train()
```

The model is trained on the processed dataset using the `trainer.train()` method.

9. **Inference Example**

```bash
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
```

This section demonstrates how to use the trained model for text generation based on a sample input. The `generate` method generates a sequence of tokens, and the output is decoded into text.


10. **Inference with Streaming**

```bash
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)
```

This section demonstrates how to perform inference with streaming, where the generated text is streamed as it is being produced.

11. **Saving the Model and Tokenizer**

```bash
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
```

Finally, the trained model and tokenizer are saved locally for future use.

This script is a complete workflow for fine-tuning a language model using the Unsloth framework and the LoRA technique. It includes steps for data preparation, model configuration, training, and inference. The script is designed to run in an environment with GPU support for efficient processing.
