"""
Fish Speech S2-Pro — 全流程推理脚本

用法:
    # 随机音色
    python run_tts.py --text "<|speaker:0|>你好，世界"

    # 降低显存
    python run_tts.py --text "<|speaker:0|>你好" --max-seq-len 16384

    # Voice clone（参考音频）
    python run_tts.py --text "<|speaker:0|>要合成的文本" \
        --ref-audio reference.wav --ref-text "参考音频对应文本"

    # 多说话人
    python run_tts.py --text "<|speaker:0|>你好\n<|speaker:1|>[happy]太棒了！"

    # 从文件读取文本
    python run_tts.py --text-file input.txt --ref-audio ref.wav --ref-text "参考文本"
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf
import torch
from loguru import logger


def parse_args():
    p = argparse.ArgumentParser(description="Fish Speech S2-Pro 全流程推理")
    # 输入
    p.add_argument("--text", type=str, default=None, help="待合成文本（支持 <|speaker:N|> 和 [emotion] 标签）")
    p.add_argument("--text-file", type=str, default=None, help="从文件读取待合成文本（与 --text 二选一）")
    # 参考音频（voice clone）
    p.add_argument("--ref-audio", type=str, default=None, help="参考音频路径（用于 voice clone）")
    p.add_argument("--ref-text", type=str, default=None, help="参考音频对应的文本")
    # 输出
    p.add_argument("--output", type=str, default="output.wav", help="输出 WAV 路径 (default: output.wav)")
    # 模型
    p.add_argument("--checkpoint", type=str, default="checkpoints/s2-pro", help="模型路径")
    p.add_argument("--device", type=str, default="cuda", help="设备 (default: cuda)")
    p.add_argument("--max-seq-len", type=int, default=0,
                   help="KV cache 大小，0 = 模型默认 32768。减小可省显存 (default: 0)")
    # 生成参数
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-length", type=int, default=300, help="长文本分 batch 的字节阈值")

    args = p.parse_args()
    if args.text is None and args.text_file is None:
        p.error("必须指定 --text 或 --text-file")
    if args.ref_audio and not args.ref_text:
        p.error("--ref-audio 需要同时指定 --ref-text")
    return args


def main():
    args = parse_args()

    # ── 1. 读取文本 ─────────────────────────────────────────────
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8").strip()
        logger.info(f"从文件读取文本: {args.text_file} ({len(text)} 字符)")
    else:
        text = args.text.replace("\\n", "\n")

    logger.info(f"待合成文本:\n{text}")

    # ── 2. 加载模型 ─────────────────────────────────────────────
    from fish_speech.models.text2semantic.inference import (
        init_model,
        load_codec_model,
        encode_audio,
        decode_to_audio,
        generate_long,
    )
    from fish_speech.models.text2semantic.llama import precompute_freqs_cis

    device = args.device
    precision = torch.bfloat16
    checkpoint = Path(args.checkpoint)

    logger.info("加载 LLM ...")
    t0 = time.time()
    model, decode_one_token = init_model(checkpoint, device, precision, compile=False)

    # 重建 buffer（meta device init 后是 placeholder）
    max_seq = (
        min(model.config.max_seq_len, args.max_seq_len)
        if args.max_seq_len > 0
        else model.config.max_seq_len
    )
    model.config.max_seq_len = max_seq
    model.register_buffer(
        "freqs_cis",
        precompute_freqs_cis(max_seq, model.config.head_dim, model.config.rope_base).to(device),
        persistent=False,
    )
    model.register_buffer(
        "causal_mask",
        torch.tril(torch.ones(max_seq, max_seq, dtype=torch.bool, device=device)),
        persistent=False,
    )
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_len=max_seq, dtype=precision)

    # 加载 Codec
    logger.info("加载 Codec ...")
    codec = load_codec_model(checkpoint / "codec.pth", device, precision)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    logger.info(f"模型加载完成: {time.time() - t0:.1f}s | max_seq_len={max_seq}")
    if torch.cuda.is_available():
        logger.info(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ── 3. 编码参考音频（可选）──────────────────────────────────
    prompt_text = None
    prompt_tokens = None
    if args.ref_audio:
        logger.info(f"编码参考音频: {args.ref_audio}")
        ref_codes = encode_audio(args.ref_audio, codec, device)
        prompt_text = [args.ref_text]
        prompt_tokens = [ref_codes.cpu()]
        logger.info(f"参考音频 VQ codes: {ref_codes.shape}")

    # ── 4. 生成 ────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    logger.info("开始生成 ...")
    t0 = time.time()
    codes_list = []

    for response in generate_long(
        model=model,
        device=device,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=1,
        max_new_tokens=0,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        compile=False,
        iterative_prompt=True,
        chunk_length=args.chunk_length,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
    ):
        if response.action == "sample":
            codes_list.append(response.codes)
            logger.info(f"已生成 batch: {response.text[:50]}...")
        elif response.action == "next":
            break

    if not codes_list:
        logger.error("未生成任何 token，请检查输入文本")
        sys.exit(1)

    gen_time = time.time() - t0

    # ── 5. 解码为音频 ──────────────────────────────────────────
    merged_codes = torch.cat(codes_list, dim=1)
    audio = decode_to_audio(merged_codes.to(device), codec)
    wav = audio.cpu().float().numpy()

    # ── 6. 保存 ────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wav, codec.sample_rate)

    duration = len(wav) / codec.sample_rate
    logger.info(f"已保存: {out_path} ({duration:.2f}s)")
    logger.info(f"生成耗时: {gen_time:.1f}s | RTF: {gen_time / duration:.2f}x")
    if torch.cuda.is_available():
        logger.info(f"VRAM 峰值: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
