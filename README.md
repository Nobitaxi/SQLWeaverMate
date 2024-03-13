# SQLWeaverMate
<div align='center'><img src="https://img.shields.io/badge/license-Apach--2.0-green"></div>

## 更新
+ [2024/03/13] 基于InternLM2-chat-7B qlora微调得到我们的SQLWeaverMate V1.0  [InternLM2-chat-7B-SQL](https://modelscope.cn/models/Nobitaxi/InternLM2-chat-7B-SQL/summary)

## 介绍

本项目旨在学习增量训练得到一个Text-to-SQL领域的垂直大模型，并结合LangChain及RAG等技术搭建一个方便使用的Demo。

---

## 训练阶段

### 训练平台
AutoDL平台，RTX 4090(24G)，Ubuntu22.04、Cuda  12.1

### 环境配置
```
# 创建一个python 3.10的环境
conda create --name xtuner python=3.10 -y
# 激活环境
conda activate xtuner

# 拉取xtuner工具源码
mkdir xtuner && cd xtuner
git clone https://github.com/InternLM/xtuner.git
# 进入源码目录（和我起的文件名重复了）
cd xtuner
# 从源码安装 XTuner
pip install -e '.[all]'
```
之后，我们在/root/autodl-tmp/路径下新建一个nl2sql文件夹作为工作路径
### 数据集

使用DB-GPT处理并在Hugging Face开源的数据集，经过筛除掉多轮对话数据以及整理格式后得到19,5297条数据。
DB-GPT-Hub：<https://github.com/eosphoros-ai/DB-GPT-Hub>  

数据集：<https://huggingface.co/datasets/Healthy13/Text2SQL>  

处理后的格式如下：

```
[
  {
    "question": "which states border arizona",
    "context": "CREATE TABLE mountain (mountain_name, mountain_altitude, state_name, country_name); CREATE TABLE city (city_name, state_name, population, country_name); CREATE TABLE road (road_name, state_name); CREATE TABLE border_info (state_name, border); CREATE TABLE river (river_name, length, traverse, country_name); CREATE TABLE state (state_name, capital, population, area, country_name, density); CREATE TABLE highlow (state_name, highest_point, highest_elevation, lowest_point, lowest_elevation); CREATE TABLE lake (lake_name, area, state_name, country_name)",
    "answer": "SELECT border FROM border_info WHERE state_name = 'arizona'"
  },
  ...
  {}
]
```

### 模型下载

python ./model_download.py

```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2-chat-7b', cache_dir='/root/autodl-tmp/nl2sql')
```

### 微调
XTuner提供多个开箱即用的配置文件，可以通过下列命令查看：
```
# 列出所有内置配置
xtuner list-cfg
```
我已经对internlm2_7b_qlora_sql_e3_copy.py文件进行了修改，通过xtuner train命令开始训练，并开启deepspeed 加速
```
xtuner train ./internlm2_7b_qlora_sql_e3_copy.py --deepspeed deepseed_zero2
```

### 将得到的 PTH 模型转换为 HuggingFace 模型，**即：生成 Adapter 文件夹**
```
mkdir hf
export MKL_SERVICE_FORCE_INTEL=1
xtuner convert pth_to_hf ./internlm2_7b_qlora_sql_e3_copy.py ./work_dirs/internlm2_7b_qlora_sql_e3_copy/epoch_3.pth ./hf
```

### 将 HuggingFace adapter 合并到大语言模型：
```
xtuner convert merge ./Shanghai_AI_Laboratory/internlm2-chat-7b ./hf ./merged --max-shard-size 2GB
```
此时，/root/autodl-tmp/nl2sql/路径下文件目录如下：
```
├── Shanghai_AI_Laboratory
│   └── internlm2-chat-7b
│       ├── README.md
│       ├── config.json
│       ├── configuration.json
│       ├── configuration_internlm2.py
│       ├── generation_config.json
│       ├── modeling_internlm2.py
│       ├── pytorch_model-00001-of-00008.bin
│       ├── pytorch_model-00002-of-00008.bin
│       ├── pytorch_model-00003-of-00008.bin
│       ├── pytorch_model-00004-of-00008.bin
│       ├── pytorch_model-00005-of-00008.bin
│       ├── pytorch_model-00006-of-00008.bin
│       ├── pytorch_model-00007-of-00008.bin
│       ├── pytorch_model-00008-of-00008.bin
│       ├── pytorch_model.bin.index.json
│       ├── special_tokens_map.json
│       ├── tokenization_internlm2.py
│       ├── tokenization_internlm2_fast.py
│       ├── tokenizer.model
│       └── tokenizer_config.json
├── dataset
│   └── single_multi_text2sql_xtuner.json
├── hf
│   ├── README.md
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── xtuner_config.py
├── internlm2_7b_qlora_sql_e3_copy.py
├── merged
│   ├── config.json
│   ├── configuration_internlm2.py
│   ├── generation_config.json
│   ├── modeling_internlm2.py
│   ├── pytorch_model-00001-of-00008.bin
│   ├── pytorch_model-00002-of-00008.bin
│   ├── pytorch_model-00003-of-00008.bin
│   ├── pytorch_model-00004-of-00008.bin
│   ├── pytorch_model-00005-of-00008.bin
│   ├── pytorch_model-00006-of-00008.bin
│   ├── pytorch_model-00007-of-00008.bin
│   ├── pytorch_model-00008-of-00008.bin
│   ├── pytorch_model.bin.index.json
│   ├── special_tokens_map.json
│   ├── tokenization_internlm2.py
│   ├── tokenization_internlm2_fast.py
│   ├── tokenizer.json
│   ├── tokenizer.model
│   └── tokenizer_config.json
├── model_download.py
└── work_dirs
    └── internlm2_7b_qlora_sql_e3_copy
        ├── 20240311_092740
        │   ├── 20240311_092740.log
        │   └── vis_data
        │       ├── 20240311_092740.json
        │       ├── config.py
        │       └── scalars.json
        ├── 20240311_093606
        │   ├── 20240311_093606.log
        │   └── vis_data
        │       ├── 20240311_093606.json
        │       ├── config.py
        │       └── scalars.json
        ├── 20240311_093944
        │   ├── 20240311_093944.log
        │   └── vis_data
        │       ├── 20240311_093944.json
        │       ├── config.py
        │       └── scalars.json
        ├── 20240311_094125
        │   ├── 20240311_094125.log
        │   └── vis_data
        │       ├── 20240311_094125.json
        │       ├── config.py
        │       └── scalars.json
        ├── epoch_1.pth
        │   ├── bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
        │   └── mp_rank_00_model_states.pt
        ├── epoch_2.pth
        │   ├── bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
        │   └── mp_rank_00_model_states.pt
        ├── epoch_3.pth
        │   ├── bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
        │   └── mp_rank_00_model_states.pt
        ├── internlm2_7b_qlora_sql_e3_copy.py
        ├── last_checkpoint
        └── zero_to_fp32.py
```
### 使用xtuner chat进行验证
```
# 加载 Adapter 模型对话（Float 16）
xtuner chat ./merged --prompt-template internlm2_chat

# 4 bit 量化加载
xtuner chat ./merged --bits 4 --prompt-template internlm2_chat
```
## 特别鸣谢
[上海人工智能实验室](https://www.shlab.org.cn/) 书生·浦语大模型实战营
