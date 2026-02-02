# Gemma-3 Accounting Analytics for Thai Auto Parts

This project focuses on fine-tuning the **Gemma-3** (specifically `google/gemma-3n-E4B-it`) model to handle accounting and analytics tasks tailored for the Thai auto parts industry. The model is trained to understand and generate Python code for various business operations such as profit/loss calculation, VAT analysis, inventory management, and customer segmentation.

## 🚀 Project Overview

The goal of this project is to create a specialized assistant for auto parts shop owners and accountants in Thailand. It leverages the power of Gemma-3 and fine-tunes it on a dataset specifically designed for the nuances of the auto parts business, including Thai language support for domain-specific queries.

### Key Capabilities
- **Profit & Loss Analysis**: Automatically calculate gains and losses from sales data.
- **VAT Calculation**: Calculate 7% Value Added Tax (VAT) for various transactions.
- **Customer Segmentation**: Identify VIP customers based on purchase history and total spent.
- **Inventory Management**: Track stock levels and generate alerts for low inventory.
- **Shipping Cost Calculation**: Determine shipping fees based on weight and distance.
- **Data Visualization**: Generate Python code using Matplotlib and Pandas to visualize business trends.

## 📊 Dataset

The model was trained on the `thai_autoparts_ai_generated (12).json` dataset, which consists of **7,906 conversational pairs**. These pairs cover a wide range of accounting scenarios, from simple dictionary-based calculations to complex database queries using SQLAlchemy.

## 🛠️ Training Details

The fine-tuning process was performed using **Parameter-Efficient Fine-Tuning (PEFT)** with **LoRA**.

- **Base Model**: `google/gemma-3n-E4B-it`
- **Hardware**: NVIDIA H100 (SDPA Mode)
- **Frameworks**: `transformers`, `peft`, `trl`, `datasets`

### LoRA Configuration
- **Rank (r)**: 64
- **Alpha**: 32
- **Dropout**: 0.05
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

### Hyperparameters
- **Learning Rate**: 1e-4
- **Epochs**: 1
- **Batch Size**: 8 (with Gradient Accumulation Steps: 4)
- **Optimizer**: `adamw_torch_fused`
- **Precision**: `bf16`
- **Max Length**: 4096 tokens

## 💻 How to Use

You can load the fine-tuned model using the `transformers` and `peft` libraries:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_id = "google/gemma-3n-E4B-it"
adapter_id = "Phonsiri/gemma-3n-E4B-accounting-analytics"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
model = PeftModel.from_pretrained(model, adapter_id)

# Sample Query
prompt = "<start_of_turn>user\nเขียนโปรแกรม Python คำนวณกำไรจากการขายอะไหล่ โดยมีราคาซื้อ 500 บาท และราคาขาย 700 บาท<end_of_turn>\n<start_of_turn>model\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 📝 License
This project follows the licensing terms of the Gemma model.
