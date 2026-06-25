"""Latency and throughput benchmark for the vLLM server.
"""

import time
import asyncio
import aiohttp
from pathlib import Path


def run_benchmark(config: dict) -> None:
    bench_cfg = config["benchmark"]
    server_cfg = config["server"]
    base_url = f"http://{server_cfg.get('host', '0.0.0.0')}:{server_cfg.get('port', 8000)}"

    prompt = "The history of artificial intelligence " * (bench_cfg.get("input_tokens", 512) // 5)

    async def send_request(session: aiohttp.ClientSession) -> float:
        start = time.time()
        async with session.post(
            f"{base_url}/v1/completions",
            json={
                "model": "default",
                "prompt": prompt,
                "max_tokens": bench_cfg.get("output_tokens", 128),
                "temperature": 0,
            },
        ) as resp:
            await resp.json()
        return time.time() - start

    async def run():
        connector = aiohttp.TCPConnector(limit=bench_cfg.get("concurrency", 4))
        async with aiohttp.ClientSession(connector=connector) as session:
            latencies = await asyncio.gather(*[send_request(session) for _ in range(bench_cfg.get("num_requests", 10))])

        latencies.sort()
        n = len(latencies)
        print(f"\nBenchmark Results ({n} requests)")
        print(f"  Mean latency:   {sum(latencies) / n:.2f}s")
        print(f"  P50 latency:    {latencies[n // 2]:.2f}s")
        print(f"  P95 latency:    {latencies[int(n * 0.95)]:.2f}s")
        print(f"  P99 latency:    {latencies[int(n * 0.99)]:.2f}s")
        print(f"  Throughput:     {n / sum(latencies):.2f} req/s")

        output_dir = Path("output/benchmark")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "latencies.txt", "w") as f:
            f.write("\n".join(str(lat) for lat in latencies))
        print(f"  Latencies saved to {output_dir / 'latencies.txt'}")

    asyncio.run(run())
