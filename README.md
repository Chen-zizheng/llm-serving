# llm-serving

本地起一个 vLLM 服务，然后用 12 条固定 prompt 做并发压测，量首字延迟（TTFT）、端到端延迟和输出吞吐。
数字全部来自 RTX 4060 Laptop 8G 上的真实运行，不是官方 benchmark 抄的。

## 结果

Qwen2.5-1.5B-Instruct，bf16，max_model_len 2048，max_num_seqs 8，eager 模式；
每个并发档位打 24 个请求，temperature=0，max_tokens=128，流式取 TTFT：

| 并发 | TTFT p50 | TTFT p95 | 端到端 p50 | 输出 tok/s | 请求/秒 |
| --- | --- | --- | --- | --- | --- |
| 1 | 33.4 ms | 43.3 ms | 1930.8 ms | 65.0 | 0.51 |
| 4 | 50.6 ms | 59.4 ms | 2215.9 ms | 226.5 | 1.79 |
| 8 | 60.2 ms | 64.5 ms | 2233.5 ms | 450.7 | 3.57 |

读法：单请求时 GPU 大部分时间在等解码，吞吐只有 65 tok/s；
并发上到 8，吞吐涨到 450 tok/s（约 7 倍），代价是 TTFT 从 33 ms 涨到 60 ms。
端到端延迟在 4 并发之后基本不再涨，说明瓶颈从"一次只跑一个请求"变成了"显存带宽"。

原始数据：`docs/bench.json`。

## 结构

```
scripts/
  serve.sh   起服务（参数都写死在里面，改之前先看注释）
  bench.py   压测：并发档位、TTFT、端到端、吞吐，只用标准库
docs/
  bench.json 这一次运行的原始结果
```

## 跑一遍

```bash
# 1. 起服务（前台，日志重定向到文件方便排查）
bash scripts/serve.sh

# 2. 另一个终端压测
python scripts/bench.py --concurrency 1,4,8 --max-tokens 128 --out results.json
```

`bench.py` 的并发是线程池，每个请求独立计时：流式请求记第一个非空 delta 的时刻为 TTFT，
非流式请求只记端到端。prompt 是固定 12 条中文技术问题，跑两轮凑够 24 个请求。

## 踩到的两个坑

**1. 默认配置下起不来：`RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`**

栈顶在 `flashinfer/sampling.py -> top_k_mask_logits -> write_ninja`。
vLLM 0.33 的采样默认走 FlashInfer，而 FlashInfer 的采样 kernel 是运行时用 ninja 现场编译的，需要 nvcc。
机器上只装了 CUDA runtime（torch 自带的），没装 CUDA toolkit，所以这条路走不通。
两条路：装 CUDA toolkit 让它编译，或者关掉 FlashInfer 采样用回 torch 原生采样。
这里选后者：`VLLM_USE_FLASHINFER_SAMPLER=0`。

**2. 第一次按默认配置（`--gpu-memory-utilization 0.85`、不关 compile）跑，把 WSL 跑挂了。**

现象是日志停在 `Initializing model took 2.7s` 之后不再前进，几分钟后 WSL 整个重启，
`/tmp/vllm.log` 跟着消失。8G 显存里 torch.compile 要做 CUDA graph capture，
再叠上 15.7 GB 主机内存，OOM 之后被 kill 掉的是整个发行版而不是一个进程。
后面把显存占用降到 0.55、`--max-model-len` 降到 2048、加 `--enforce-eager` 才稳定。
日志也因此改写到 `/mnt/d/...`，WSL 重启不会再丢。

## 已知限制

- **eager 模式是为了能起来，代价是吞吐。** 关掉 compile 与 CUDA graph，单请求延迟会高一些；
  这张 8G 卡上先把服务跑稳定比多榨 10% 吞吐重要，真要压性能得换显存更大的机器。
- **没有鉴权，也没限流。** 只监听本机 8000 端口，当成本地实验环境用。
- **上下文只有 2048。** 1.5B 模型在 8G 卡上给 KV cache 留的空间有限，长文档场景要降 `--max-num-seqs` 换长度。
- **只测了一个模型、一张卡、一组参数。** 表里的数字是这组配置的结果，换量化或换 batch 策略都要重测。

## 环境

- WSL2 Ubuntu，Python 3.12，vLLM 0.33.0，torch 2.14.0+cu130
- RTX 4060 Laptop 8G（8188 MiB），主机内存 15.7 GB
