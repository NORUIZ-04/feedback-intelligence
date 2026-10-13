"""Download the RAG corpus and the license files of its sources.

Run from the repository root:  python -m src.rag.download
"""

import hashlib
import json
from pathlib import Path

import requests
import yaml

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)
OUT = Path("data/docs")


def fetch(url: str) -> tuple[int, str]:
    resp = requests.get(url, timeout=30)
    return resp.status_code, resp.text


def main() -> None:
    cfg = yaml.safe_load(Path("configs/rag_sources.yaml").read_text(encoding="utf-8"))
    (OUT / "licenses").mkdir(parents=True, exist_ok=True)
    manifest = {"licenses": {}, "documents": {}, "failed": {}}

    for name, url in cfg["licenses"].items():
        status, text = fetch(url)
        if status == 200:
            (OUT / "licenses" / f"{name}_LICENSE.txt").write_text(text, encoding="utf-8")
            first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
            manifest["licenses"][name] = {"url": url, "first_line": first}
        else:
            manifest["failed"][f"license:{name}"] = {"url": url, "status": status}

    for doc in cfg["documents"]:
        status, text = fetch(doc["url"])
        if status == 200:
            path = OUT / f"{doc['name']}.md"
            path.write_text(text, encoding="utf-8")
            manifest["documents"][doc["name"]] = {
                "url": doc["url"],
                "characters": len(text),
                "words": len(text.split()),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        else:
            manifest["failed"][doc["name"]] = {"url": doc["url"], "status": status}

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print()
    print("| Document | Words | Characters |")
    print("|---|---|---|")
    for name, d in manifest["documents"].items():
        print(f"| {name} | {d['words']} | {d['characters']} |")
    print()
    for name, lic in manifest["licenses"].items():
        print(f"License {name}: {lic['first_line']}")
    print()
    print(f"Downloaded {len(manifest['documents'])} of {len(cfg['documents'])} documents")
    for name, info in manifest["failed"].items():
        print(f"FAILED {name}: HTTP {info['status']} {info['url']}")


if __name__ == "__main__":
    main()