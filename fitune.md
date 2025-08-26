# 🚀 模型微调、量化与部署完整指南

本指南涵盖了从模型微调到生产部署的完整流程，包括 LoRA 微调、GGUF 量化、以及 Ollama 部署，仅为操作，不包含理论解析。

---

## 📋 目录

- [模型微调](#-模型微调)
- [模型量化](#-模型量化)  
- [Ollama 部署](#-ollama-部署)
- [VLLM 部署](#-vllm-部署)

---

## 🔧 模型微调

### 调试配置

首先需要在训练器中添加调试代码来查看数据预处理效果：

**编辑文件：** `src/llamafactory/train/sft/trainer.py`

在以下代码块后添加调试逻辑：

```python
if is_transformers_version_greater_than("4.46"):
    kwargs["processing_class"] = kwargs.pop("tokenizer")
else:
    self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

super().__init__(**kwargs)

# ✨ 添加调试代码，打印输入tokens，输入模型的字符串和labels
import os
print("\n=== 🔍 调试：查看预处理后的样本 ===")
train_dataset = kwargs.get('train_dataset')
if train_dataset:
    for i in range(min(10, len(train_dataset))):
        sample = train_dataset[i]
        print(f"\n📝 样本 {i+1}:")
        print("🔤 Input tokens:", sample['input_ids'][:100])
        print("📄 Decoded text:", self.processing_class.batch_decode([sample['input_ids']], skip_special_tokens=False)[0])
        if 'labels' in sample:
            print("🏷️ Labels:", sample['labels'])
print("=== ✅ 调试结束 ===\n")
```

### 数据预处理调试

执行以下命令进行数据预处理调试：

```bash
**编辑文件并新增训练任务glaive_toolcall_en_demo指定对应的训练数据集路径和格式模板等：** `src/data/dataset_info.json`
export MODEL_PATH=/root/autodl-tmp/model/Qwen/Qwen2.5-7B-Instruct

python src/train.py \
    --stage sft \
    --model_name_or_path $MODEL_PATH \
    --dataset glaive_toolcall_en_demo \
    --template qwen \
    --cutoff_len 2048 \
    --do_train false
```

**调试输出示例：**

<details>
<summary>点击查看完整调试输出</summary>

```
样本 1:
Input tokens: [151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 382, 2, 13852, 271, 2610, 1231, 1618, 825, 476, 803, 5746, 311, 7789, 448, 279, 1196, 3239, 382, 2610, 525, 3897, 448, 729, 32628, 2878, 366, 15918, 1472, 15918, 29, 11874, 9492, 510, 27, 15918, 397, 4913, 1313, 788, 330, 1688, 497, 330, 1688, 788, 5212, 606, 788, 330, 1836, 7080, 8923, 497, 330, 4684, 788, 330, 5890, 369, 18627, 3118, 389, 13966, 497, 330, 13786, 788, 5212, 1313, 788, 330, 1700, 497, 330, 13193, 788, 5212, 38120, 788, 5212, 1313, 788]

Decoded text: <|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search_recipes", "description": "Search for recipes based on ingredients", "parameters": {"type": "object", "properties": {"ingredients": {"type": "array", "items": {"type": "string"}, "description": "The ingredients to search for"}}, "required": ["ingredients"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
Hi, I have some ingredients and I want to cook something. Can you help me find a recipe?<|im_end|>
<|im_start|>assistant
Of course! I can help you with that. Please tell me what ingredients you have.<|im_end|>
<|im_start|>user
I have chicken, bell peppers, and rice.<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "search_recipes", "arguments": {"ingredients": ["chicken", "bell peppers", "rice"]}}
</tool_call><|im_end|>

Labels: [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 2124, 3308, 0, 358, 646, 1492, 498, 448, 429, 13, 5209, 3291, 752, 1128, 13966, 498, 614, 13, 151645, 198, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 151657, 198, 4913, 606, 788, 330, 1836, 7080, 8923, 497, 330, 16370, 788, 5212, 38120, 788, 4383, 331, 9692, 497, 330, 17250, 57473, 497, 330, 23120, 1341, 11248, 151658, 151645, 198, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 40, 1730, 1378, 18627, 369, 498, 13, 576, 1156, 825, 374, 330, 83863, 323, 17884, 51782, 64192, 52417, 3263, 576, 11221, 525, 25, 19649, 279, 16158, 1119, 2613, 9666, 13, 56476, 279, 28419, 57473, 13, 12514, 279, 19653, 13, 64192, 52546, 279, 16158, 323, 28419, 57473, 13, 52932, 916, 19653, 13, 576, 2086, 825, 374, 330, 83863, 323, 29516, 25442, 261, 1263, 3263, 576, 11221, 525, 25, 12514, 279, 16158, 323, 19653, 25156, 13, 19219, 1105, 3786, 448, 279, 28419, 57473, 304, 264, 272, 33758, 1263, 12000, 13, 52074, 3080, 20748, 13876, 13, 15920, 825, 1035, 498, 1075, 311, 1430, 30, 151645, 198, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 40, 2776, 14589, 11, 714, 438, 458, 15235, 11, 358, 1513, 944, 614, 279, 22302, 311, 2736, 9250, 9079, 1741, 438, 21391, 13966, 13, 4354, 11, 358, 646, 1492, 498, 1477, 803, 18627, 476, 3410, 17233, 10414, 421, 498, 1184, 13, 151645, 198]
```
</details>

### 训练参数配置

**关键训练参数说明：**

| 参数 | 计算公式 | 示例值 |
|------|---------|--------|
| `warmup_steps` | 约为总步数的 20% | 20 |
| 有效批次大小 | `batch_size × gradient_accumulation_steps` | 2 × 8 = 16 |
| 总步数 | `(samples × epochs) ÷ 有效批次大小` | - |
| cutoff_len | 最大输入长度 | 4096 |
| glaive_toolcall_en_demo | 对应在data_info.json文件中的配置训练名称 | 4096 |

### 执行微调训练

```bash
python src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path $MODEL_PATH \
    --dataset glaive_toolcall_en_demo \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir /root/autodl-tmp/tool_call_full \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --cutoff_len 4096 \
    --num_train_epochs 5 \
    --warmup_steps 20 \
    --learning_rate 5e-6 \
    --logging_steps 10 \
    --save_steps 30 \
    --gradient_checkpointing true \
    --bf16
```

### 模型合并

微调完成后，需要将 LoRA 适配器与基础模型合并：

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export \
    --model_name_or_path /root/autodl-tmp/model/Qwen/Qwen2.5-7B-Instruct \
    --adapter_name_or_path /root/autodl-tmp/tool_call_full \
    --template qwen \
    --finetuning_type lora \
    --export_dir /root/autodl-tmp/merge_model_tool_call_20250816_qwen \
    --export_size 2 \
    --export_legacy_format False
```
### 模型测试
```python
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# 假设你的模型支持工具调用
model_path = "/root/autodl-tmp/merge_model_tool_call_20250816_qwen"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


tools = [
    {
        "name": "HandleTongzhiTonggao",
        "description": "该工具专用于新增、修改、删除时讯消息或通告！",
        "parameters": [
        {
            "name": "operation",
            "description": "用户对时讯消息或通告的操作方式，从列表中选择其中一个：['ADD', 'DELETE', 'UPDATE', 'LIST']",
            "required": True
        },
        {
            "name": "type",
            "description": "用户上传消息的类型，从列表中选择其中一个：[\"时讯消息\", \"通告\"]",
            "required": True
        },
        {
            "name": "content",
            "description": "用户上传消息的内容",
            "required": True
        }
        ]
    },
    {
        "name": "WeatherApi",
        "description": "该工具专用于市、区县的实时天气查询",
        "parameters": [
        {
            "name": "query_key",
            "description": "需要查询天气的地区（可以是直辖市、区县）",
            "required": True
        }
        ]
    },
    {
        "name": "WaterMachineApi",
        "description": "物联网饮水机操作接口。",
        "parameters": [
        {
            "name": "operation",
            "description": "从用户关于饮水机操作的问题中提取相关的操作指令，可选择的指令包含：[水壶取水, 水壶停水, 停止加热, 开始保温, 停止保温, 打开语音, 关闭语音]",
            "required": True
        }
        ]
    },
    {
        "name": "HealthReportSimple",
        "description": "专门用于处理用户个人健康监测数据的所有查询。当用户问题明确涉及以下任一方面时，系统必须优先调用此工具：(1)睡眠相关指标（如\"睡眠报告\"、\"睡眠数据\"、\"睡眠质量\"、\"睡眠时长\"、\"睡眠效率\"、\"睡眠评分\"、\"深度睡眠\"、\"浅度睡眠\"）；(2)生理指标（如\"心率\"、\"心率异常\"、\"呼吸率\"、\"呼吸异常\"、\"体动次数\"、\"体动指数\"）；(3)健康状态（如\"健康评分\"、\"健康状况\"、\"健康异常\"）；(4)任何涉及用户个人监测数据的时间性查询（如\"昨晚\"、\"最近几天\"、\"本周\"、\"上周\"等时间段的健康数据）。HealthReport提供专业的健康数据解读和分析。",
        "parameters": [
        {
            "name": "health_report_question",
            "description": "用户关于睡眠健康报告的完整问题，需根据当前问题和历史对话上下文进行综合理解和总结。系统将分析用户的睡眠监测数据并提供专业解读。问题可涉及特定日期或时间段的睡眠质量分析、睡眠趋势比较、睡眠异常解释、健康建议等。系统将根据问题自动检索相关的睡眠数据记录，并给出专业的分析和建议。",
            "required": True
        }
        ]
    },
    {
        "name": "DirectLLMCommunityAiUser",
        "description": "默认回答工具，负责无工具调用的兜底，上述工具无法调用时可调用该工具",
        "parameters": [
        {
            "name": "question",
            "description": "用户任何问题！",
            "required": True
        }
        ]
    }
]


# 构建消息
messages = [
    {"role": "user", "content": "新增时讯消息"},
    {"role": "assistant", "content": "Sure, I can help with that. Could you please provide me with the content for the new announcement?"},
    {"role": "user", "content": "明天中午12点公司放假！"}
]

# 应用聊天模板，这里会自动按照基座模型qwen要求的格式去转换输入参数，转换后输入模型前的最终结果可以查看{decoded_input}，与训练数据的转换结果一致
prompt = tokenizer.apply_chat_template(
    messages, 
    tools=tools,
    add_generation_prompt=True,
    tokenize=False
)

# 生成回复
inputs = tokenizer(prompt, return_tensors="pt")
input_length = inputs.input_ids.shape[1]
decoded_input = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
outputs = model.generate(**inputs, max_new_tokens=256)
new_tokens = outputs[0][input_length:]  # 截取新生成的token
response = tokenizer.decode(new_tokens, skip_special_tokens=True)
print("===============================decode_input================================================")
print(f"{decoded_input}")
print("===============================decode_input================================================")
print("=================================response==============================================")
print(f"{response}")
print("=================================response==============================================")
```

<details>
最终输出结果

===============================decode_input================================================
system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"name": "HandleTongzhiTonggao", "description": "该工具专用于新增、修改、删除时讯消息或通告！", "parameters": [{"name": "operation", "description": "用户对时讯消息或通告的操作方式，从列表中选择其中一个：['ADD', 'DELETE', 'UPDATE', 'LIST']", "required": true}, {"name": "type", "description": "用户上传消息的类型，从列表中选择其中一个：[\"时讯消息\", \"通告\"]", "required": true}, {"name": "content", "description": "用户上传消息的内容", "required": true}]}
{"name": "WeatherApi", "description": "该工具专用于市、区县的实时天气查询", "parameters": [{"name": "query_key", "description": "需要查询天气的地区（可以是直辖市、区县）", "required": true}]}
{"name": "WaterMachineApi", "description": "物联网饮水机操作接口。", "parameters": [{"name": "operation", "description": "从用户关于饮水机操作的问题中提取相关的操作指令，可选择的指令包含：[水壶取水, 水壶停水, 停止加热, 开始保温, 停止保温, 打开语音, 关闭语音]", "required": true}]}
{"name": "HealthReportSimple", "description": "专门用于处理用户个人健康监测数据的所有查询。当用户问题明确涉及以下任一方面时，系统必须优先调用此工具：(1)睡眠相关指标（如\"睡眠报告\"、\"睡眠数据\"、\"睡眠质量\"、\"睡眠时长\"、\"睡眠效率\"、\"睡眠评分\"、\"深度睡眠\"、\"浅度睡眠\"）；(2)生理指标（如\"心率\"、\"心率异常\"、\"呼吸率\"、\"呼吸异常\"、\"体动次数\"、\"体动指数\"）；(3)健康状态（如\"健康评分\"、\"健康状况\"、\"健康异常\"）；(4)任何涉及用户个人监测数据的时间性查询（如\"昨晚\"、\"最近几天\"、\"本周\"、\"上周\"等时间段的健康数据）。HealthReport提供专业的健康数据解读和分析。", "parameters": [{"name": "health_report_question", "description": "用户关于睡眠健康报告的完整问题，需根据当前问题和历史对话上下文进行综合理解和总结。系统将分析用户的睡眠监测数据并提供专业解读。问题可涉及特定日期或时间段的睡眠质量分析、睡眠趋势比较、睡眠异常解释、健康建议等。系统将根据问题自动检索相关的睡眠数据记录，并给出专业的分析和建议。", "required": true}]}
{"name": "DirectLLMCommunityAiUser", "description": "默认回答工具，负责无工具调用的兜底，上述工具无法调用时可调用该工具", "parameters": [{"name": "question", "description": "用户任何问题！", "required": true}]}
</tools>
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
user
新增时讯消息
assistant
Sure, I can help with that. Could you please provide me with the content for the new announcement?
user
明天中午12点公司放假！
assistant
===============================decode_input================================================

=================================response==============================================
<tool_call>
{"name": "HandleTongzhiTonggao", "arguments": {"operation": "ADD", "type": "时讯消息", "content": "明天中午12点公司放假！"}}
</tool_call>
=================================response==============================================
</details>
```

---


## ⚡ 模型量化

### GGUF 格式转换

将 HuggingFace 格式的模型转换为 GGUF 格式，并进行 4 比特量化：

```bash
# 📦 第一步：转换为 F16 格式
python convert_hf_to_gguf.py \
    /root/autodl-tmp/merge_model_tool_call_20250816_qwen \
    --outfile /root/autodl-tmp/tool_call_f16.gguf \
    --outtype f16

# 🗜️ 第二步：量化为 4 比特
./build/llama-quantize \
    /root/autodl-tmp/tool_call_f16.gguf \
    /root/autodl-tmp/tool_call_q4.gguf \
    /root/autodl-tmp/tool_call_q4.gguf \
    Q4_K_M
```

### 量化格式对比

| 格式 | 文件大小 | 质量 | 推荐场景 |
|------|----------|------|----------|
| **F16** | ~13GB | 最高 | 开发测试 |
| **Q8_0** | ~7GB | 很高 | 高质量推理 |
| **Q4_K_M** | ~4GB | 良好 | 生产部署 ⭐ |
| **Q4_0** | ~3.9GB | 一般 | 资源受限 |

---

## 🐋 Ollama 部署

### Modelfile 配置

创建 `Modelfile` 文件来定义模型配置：

```dockerfile
FROM /work/ai/model/tool_call_q4.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}{{ end }}{{ if .Tools }}
# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{ range .Tools }}{{ . }}{{ end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>{{ end }}<|im_end|>
{{ if .Messages }}{{ range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}{{ end }}<|im_start|>assistant
"""

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER stop <|im_end|>
PARAMETER stop <|endoftext|>
```

### API 调用示例

使用标准 OpenAI 格式进行 Function Calling：

```bash
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "toolcall_qwen2.5_q4",
    "messages": [
      {
        "role": "user",
        "content": "新增通告"
      },
      {
        "role": "assistant", 
        "content": "好的，您想新增什么样的通告呢？请告诉我通告的具体内容。"
      },
      {
        "role": "user",
        "content": "今天中午12点公司放假"
      }
    ],
    "stream": false,
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "HandleTongzhiTonggao",
          "description": "该工具专用于新增、修改、删除时讯消息或通告！",
          "parameters": {
            "type": "object",
            "properties": {
              "operation": {
                "type": "string",
                "description": "用户对时讯消息或通告的操作方式",
                "enum": ["ADD", "DELETE", "UPDATE", "LIST"]
              },
              "type": {
                "type": "string", 
                "description": "用户上传消息的类型",
                "enum": ["时讯消息", "通告"]
              },
              "content": {
                "type": "string",
                "description": "用户上传消息的内容"
              }
            },
            "required": ["operation", "type", "content"]
          }
        }
      }
    ]
  }'
```

### 预期返回结果

```json
{
  "model": "toolcall_qwen2.5_q4",
  "created_at": "2025-08-18T07:28:39.03524207Z",
  "message": {
    "role": "assistant",
    "content": "<tool_call>\n{\"name\": \"HandleTongzhiTonggao\", \"arguments\": {\"operation\": \"ADD\", \"type\": \"通告\", \"content\": \"今天中午12点公司放假\"}}\n</tool_call>"
  },
  "done_reason": "stop",
  "done": true,
  "total_duration": 950943370,
  "load_duration": 15252446,
  "prompt_eval_count": 242,
  "prompt_eval_duration": 17372578,
  "eval_count": 45,
  "eval_duration": 908369040
}
```

---

# 🚀 vLLM 量化模型部署

---

## 📋 量化格式支持

### vLLM 支持的量化格式

| 量化方法 | 格式 | vLLM 支持 | 推荐度 |
|---------|------|-----------|--------|
| **AWQ** | `.safetensors` | ✅ 完全支持 | ⭐⭐⭐⭐⭐ |
| **GPTQ** | `.safetensors` | ✅ 完全支持 | ⭐⭐⭐⭐ |
| **SqueezeLLM** | `.safetensors` | ✅ 支持 | ⭐⭐⭐ |
| **FP8** | `.safetensors` | ✅ 支持 | ⭐⭐⭐⭐ |
| **GGUF** | `.gguf` | ❌ 不支持 | ❌ |

> **重要提醒：** vLLM **不支持** GGUF 格式！需要使用 HuggingFace 兼容的量化格式。

---

## 📚 参考资料

- [LlamaFactory 官方文档](https://github.com/hiyouga/LLaMA-Factory)
- [llama.cpp 构建文档](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md)
- [Ollama 模型文件格式](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)

---

*📅 最后更新：2025-08-18*