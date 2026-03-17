# Fish Speech S2-Pro — 修改说明

## 概述

对 [fish-speech](https://github.com/fishaudio/fish-speech) 原仓库的 S2-Pro 推理流程做了一系列改动，目标是让模型在 **Windows + 消费级显卡（16-24 GB VRAM）** 上更易使用。

主要改进：
- Meta device 零内存初始化，降低模型加载峰值显存
- 可调 `--max-seq-len`，以 KV cache 大小换显存
- Windows 环境下中日文输出乱码修复
- Notebook 完全重写，一行函数即可生成+播放

## 修改文件清单

| 文件 | 改动摘要 |
|------|----------|
| `fish_speech/models/text2semantic/inference.py` | meta device 加载、`--max-seq-len` CLI 参数、buffer 重建、Windows UTF-8 |
| `inference.ipynb` | 完全重写：调用封装好的 `init_model()`，`tts()` 辅助函数，三种模式示例 |

---

## 详细改动说明

### 1. Meta device 模型加载

**问题**：原 `init_model` 在 GPU 上直接实例化模型再加载权重，Windows WDDM 驱动下显存碎片化严重，1.7B 模型 24 GB 显卡就可能 OOM。

**方案**：`torch.device("meta")` 上下文中创建模型（零内存），然后将权重逐个转为目标 dtype 后用 `load_state_dict(..., assign=True)` 直接替换 meta tensor，峰值显存 ≈ 模型本身大小。

**代码位置**：`inference.py` `init_model()` (L362-451)

```python
# 2. Create model on meta device (zero memory)
with torch.device("meta"):
    model = DualARTransformer(config)

# 4. Convert weights to target dtype and load (replaces meta tensors)
for k in weights:
    if weights[k].is_floating_point():
        weights[k] = weights[k].to(dtype=precision, device=device)
    else:
        weights[k] = weights[k].to(device=device)
model.load_state_dict(weights, strict=False, assign=True)
```

### 2. 可调 `--max-seq-len`

**问题**：模型默认 `max_seq_len=32768`，KV cache 占用大量显存。对于普通长度的文本（几十到几百字），完整 32768 的 cache 并不必要。

**方案**：新增 CLI 参数 `--max-seq-len`（默认 0 = 使用模型原始值 32768）。设小后 KV cache、`freqs_cis`、`causal_mask` 均按实际值分配，显存显著下降。

**代码位置**：`inference.py` `main()` (L898-900, L943-966)

```python
@click.option("--max-seq-len", type=int, default=0,
              help="Override max sequence length (KV cache size). 0 = use model default (32768). "
                   "Smaller values save VRAM and run faster but limit max generation length.")

# 在 main() 中：
max_seq = min(model.config.max_seq_len, max_seq_len) if max_seq_len > 0 else model.config.max_seq_len
model.config.max_seq_len = max_seq
```

### 3. Buffer 延迟重建

**问题**：`freqs_cis`、`causal_mask`、`fast_freqs_cis` 是 non-persistent buffer，meta device 初始化后不在 `state_dict` 中，需要手动重建。

**方案**：`init_model()` 中先注册 placeholder（空 tensor），在 `main()` 中按实际 `max_seq_len` 重建为正确大小。

**代码位置**：
- Placeholder 注册：`inference.py` `init_model()` (L411-423)
- 实际重建：`inference.py` `main()` (L946-956)

```python
# init_model() — placeholder
model.register_buffer("freqs_cis", torch.empty(0, device=device), persistent=False)
model.register_buffer("causal_mask", torch.empty(0, dtype=torch.bool, device=device), persistent=False)

# main() — 按实际 max_seq_len 重建
model.register_buffer("freqs_cis",
    precompute_freqs_cis(max_seq, model.config.head_dim, model.config.rope_base).to(device),
    persistent=False)
model.register_buffer("causal_mask",
    torch.tril(torch.ones(max_seq, max_seq, dtype=torch.bool, device=device)),
    persistent=False)
```

### 4. Windows UTF-8 编码

**问题**：Windows 默认编码为 GBK/CP936，中日文文本在日志输出时乱码或直接报 `UnicodeEncodeError`。

**方案**：在 notebook 环境初始化时设置 `PYTHONUTF8=1` 环境变量并 `reconfigure` stdout/stderr。

**代码位置**：`inference.ipynb` Cell 0（环境初始化）

```python
os.environ["PYTHONUTF8"] = "1"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
```

---

## Notebook 更新 (`inference.ipynb`)

完全重写，替换原有的 notebook。

### 结构

| Cell | 内容 |
|------|------|
| 0 | 环境初始化：import、UTF-8、配置常量 |
| 1 | 加载模型：调用 `init_model()` + buffer 重建 + codec 加载 |
| `tts()` | 辅助函数：一行调用生成+解码+播放+保存 |
| 2 | 随机音色生成示例（无参考音频） |
| 3 | Voice Clone 示例（参考音频 → `encode_audio()` → 传入 `prompt_text/prompt_tokens`） |
| 4 | 多说话人 / 情感控制示例（`<\|speaker:N\|>` 标签 + `[emotion]` 标注） |
| 5 | 显存监控与清理工具 |

### `tts()` 辅助函数

封装了 `generate_long()` → `decode_to_audio()` → `IPython.display.Audio` 的完整流程，支持参数：

```python
tts(text, seed=42, temperature=0.7, top_p=0.9, top_k=30,
    prompt_text=None, prompt_tokens=None, out_path=None)
```

---

## 性能参考

测试环境：RTX 3090 24GB, bf16, S2-Pro 1.7B

| max_seq_len | VRAM（峰值） | 速度 (tokens/s) | 备注 |
|-------------|-------------|-----------------|------|
| 32768（默认）| ~20.8 GB | ~3.5 | 完整上下文 |
| 16384 | ~17.1 GB | ~5.3 | 足够日常使用 |

> 数据来自实际 notebook 运行日志。速度受输入长度、是否有参考音频等因素影响。

---

## 使用方法

### CLI

```bash
# 基本用法（默认 max_seq_len）
python -m fish_speech.models.text2semantic.inference \
    --checkpoint-path checkpoints/s2-pro \
    --text "<|speaker:0|>你好，世界" \
    --output output.wav

# 降低显存：限制 KV cache 到 16384
python -m fish_speech.models.text2semantic.inference \
    --checkpoint-path checkpoints/s2-pro \
    --text "<|speaker:0|>你好，世界" \
    --max-seq-len 16384 \
    --output output.wav

# Voice clone：指定参考音频
python -m fish_speech.models.text2semantic.inference \
    --checkpoint-path checkpoints/s2-pro \
    --text "<|speaker:0|>要合成的文本" \
    --prompt-text "参考音频对应的文本" \
    --prompt-audio reference.wav \
    --output output.wav
```

### Notebook

打开 `inference.ipynb`，按顺序运行 Cell 0-1 加载模型，然后使用 `tts()` 函数：

```python
# 随机音色
tts("<|speaker:0|>你好，世界", out_path="output.wav")

# Voice clone
ref_codes = encode_audio("reference.wav", codec, DEVICE)
tts("要合成的文本", prompt_text=["参考文本"], prompt_tokens=[ref_codes.cpu()])

# 多说话人
tts("""<|speaker:0|>你好啊
<|speaker:1|>[happy]太棒了！""")
```

---

## 注意事项

1. **必须使用 bf16，不要用 fp16**：S2-Pro 模型权重为 bfloat16 训练，fp16 动态范围不足会导致数值溢出和生成质量下降。RTX 30 系及以上均支持 bf16。

2. **Windows 环境设置 `PYTHONUTF8=1`**：CLI 使用时建议在启动前设置环境变量，避免中日文文本处理报错：
   ```powershell
   $env:PYTHONUTF8 = "1"
   ```

3. **`--max-seq-len` 与生成长度的权衡**：减小该值会限制单次生成的最大 token 数。对于长文本（> 500 字），建议保持默认 32768 或至少 16384。短文本（日常对话、几句话）用 8192 也足够。

4. **KV cache 惰性初始化**：`generate()` 函数中 cache 仅在首次调用时创建（`_cache_setup_done` 标记），后续调用复用，避免重复分配。
