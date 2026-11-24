import argparse
import json
import os
from pathlib import Path

from .core import Veritas


def main():
    p = argparse.ArgumentParser(prog="veritas")
    p.add_argument("--config", default=os.getenv("VERITAS_CONFIG"))
    subs = p.add_subparsers(dest="command", required=True)
    idx = subs.add_parser("index")
    idx.add_argument("jsonl", type=Path)
    for name in ["search", "retrieve", "import-searxng"]:
        cmd = subs.add_parser(name)
        cmd.add_argument("query")
        cmd.add_argument("--steps", type=int, default=3)
        cmd.add_argument("--as-of")
    serve = subs.add_parser("serve")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()
    if args.command == "serve":
        import uvicorn

        if args.host not in {"127.0.0.1", "localhost", "::1"} and os.getenv("VERITAS_ENV") != "production":
            p.error("Set VERITAS_ENV=production and VERITAS_TOKEN for a network listener")
        if args.config:
            os.environ["VERITAS_CONFIG"] = args.config
        uvicorn.run("veritas.api:factory", factory=True, host=args.host, port=args.port)
        return
    engine = Veritas.from_config(args.config)
    try:
        if args.command == "index":
            with args.jsonl.open() as f:
                engine.add([json.loads(s) for s in f if s.strip()])
            print(json.dumps({"documents": len(engine.docs)}))
        elif args.command == "retrieve":
            print(json.dumps([h.model_dump() for h in engine.retrieve(args.query)], indent=2))
        elif args.command == "import-searxng":
            from .integrations import searxng

            engine.add(searxng(os.environ["VERITAS_SEARXNG"], args.query))
            print(json.dumps({"documents": len(engine.docs)}))
        else:
            print(engine.search(args.query, args.steps, args.as_of).model_dump_json(indent=2))
    finally:
        engine.close()
