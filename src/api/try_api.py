"""Start the real server, call every endpoint, print the results.

Run from the repository root:  python -m src.api.try_api
"""

import subprocess
import sys
import time

import httpx

BASE = "http://127.0.0.1:8000"


def main() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main:app", "--port", "8000", "--log-level", "warning"])
    try:
        start = time.time()
        while time.time() - start < 180:
            try:
                if httpx.get(f"{BASE}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(2)
        else:
            raise SystemExit("server did not start within 180 s")
        print(f"server ready after {time.time() - start:.1f} s")

        client = httpx.Client(base_url=BASE, timeout=120)
        print("health:", client.get("/health").json())
        s = client.get("/status").json()
        print("status:", {k: s[k] for k in ("generator", "four_bit", "index_chunks", "gpu_gb_now")})

        r = client.post("/classify", json={"texts": ["i feel so helpless right now", "i am so happy"]})
        print("classify:", r.status_code, [(x["label"], x["score"]) for x in r.json()["results"]],
              r.headers["x-process-time-ms"], "ms")
        r = client.post("/ask", json={"question": "What is the default batch size when using map?"})
        body = r.json()
        print("ask:", r.status_code, repr(body["answer"][:120]), "refused", body["refused"],
              "cited", body["cited"], r.headers["x-process-time-ms"], "ms")

        print("\nInvalid input (expect 422):")
        bad = [
            ("/classify", {"texts": []}),
            ("/classify", {"texts": ["   "]}),
            ("/classify", {"texts": ["x" * 1001]}),
            ("/classify", {"texts": ["ok"] * 33}),
            ("/classify", {}),
            ("/ask", {"question": "hi"}),
            ("/ask", {"question": "q" * 501}),
        ]
        for path, payload in bad:
            code = client.post(path, json=payload).status_code
            print(f"  {path} {str(payload)[:40]:42} -> {code}")
    finally:
        proc.terminate()
        proc.wait(timeout=30)


if __name__ == "__main__":
    main()