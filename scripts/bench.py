# -*- coding: utf-8 -*-
"""对本地 vLLM 服务做并发压测：首字延迟（TTFT）、端到端延迟、输出吞吐。

只用标准库，不额外装压测工具。每个并发档位固定发送 N 个请求（复用同一批 prompt），
temperature=0、max_tokens 固定，尽量让两次运行的差异只来自服务端调度。
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPTS = [
    "用三句话说明 TCP 三次握手为什么不能是两次。",
    "解释什么是数据库的乐观锁，并给一个 SQL 例子。",
    "为什么 Nginx 用 epoll 而不是 select？",
    "写一个 Python 函数，判断字符串是否是回文。",
    "解释一下什么是幂等性，HTTP 方法里哪些是幂等的。",
    "为什么日志采集要用有界队列？队列满了应该怎么办？",
    "说明一下 B+ 树索引为什么适合范围查询。",
    "什么是 CAP 理论？实际系统怎么取舍？",
    "解释 git rebase 和 merge 的区别。",
    "为什么 Docker 镜像要分层？",
    "讲一下 pytest fixture 的作用域。",
    "什么是缓存击穿、穿透和雪崩，各怎么防？",
]


def one_request(base, model, prompt, max_tokens, stream):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": stream}
    if stream:
        body["stream_options"] = {"include_usage": True}
    data = json.dumps(body).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    text = ""
    usage = None
    with urllib.request.urlopen(req, timeout=180) as r:
        if not stream:
            payload = json.loads(r.read())
            t1 = time.perf_counter()
            return {"ttft": None, "e2e": t1 - t0, "completion_tokens": payload["usage"]["completion_tokens"]}
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            obj = json.loads(chunk)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text += piece
    t1 = time.perf_counter()
    return {"ttft": ttft, "e2e": t1 - t0,
            "completion_tokens": (usage or {}).get("completion_tokens", 0)}


def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((len(s) - 1) * q))))
    return s[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--concurrency", default="1,4,8,16")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    levels = [int(x) for x in args.concurrency.split(",")]
    all_results = {"model": args.model, "max_tokens": args.max_tokens, "levels": []}

    # 先跑一次非流式，拿吞吐基准（这时候并发 1，排除首字计时开销）
    warm = one_request(args.base, args.model, "你好", 8, False)

    for c in levels:
        ttfts, e2es, toks = [], [], []
        lock = threading.Lock()

        def work(i):
            r = one_request(args.base, args.model, PROMPTS[i % len(PROMPTS)], args.max_tokens, True)
            with lock:
                if r["ttft"]:
                    ttfts.append(r["ttft"])
                e2es.append(r["e2e"])
                toks.append(r["completion_tokens"])

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=c) as ex:
            list(ex.map(work, range(max(c, 12) * 2)))
        wall = time.perf_counter() - t0
        total_tokens = sum(toks)
        row = {
            "concurrency": c,
            "requests": len(e2es),
            "wall_s": round(wall, 2),
            "ttft_p50_ms": round(pct(ttfts, 0.5) * 1000, 1) if ttfts else None,
            "ttft_p95_ms": round(pct(ttfts, 0.95) * 1000, 1) if ttfts else None,
            "e2e_p50_ms": round(pct(e2es, 0.5) * 1000, 1),
            "e2e_p95_ms": round(pct(e2es, 0.95) * 1000, 1),
            "output_tokens": total_tokens,
            "tokens_per_s": round(total_tokens / wall, 1),
            "req_per_s": round(len(e2es) / wall, 2),
        }
        print(json.dumps(row, ensure_ascii=False))
        all_results["levels"].append(row)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
