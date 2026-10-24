"""Command line entry point: python -m telecom_agent <command>"""

from __future__ import annotations

import argparse
import json

from .config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="telecom_agent", description="NileTel support agent")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="build the Chroma vector store from data/knowledge_base")
    seed = sub.add_parser("seed-db", help="create the synthetic SQLite customer database")
    seed.add_argument("--reset", action="store_true")

    ask = sub.add_parser("ask", help="ask a single question")
    ask.add_argument("message")
    ask.add_argument("--customer", help="customer ID, e.g. NT-1001")

    chat = sub.add_parser("chat", help="interactive chat in the terminal")
    chat.add_argument("--customer", help="customer ID, e.g. NT-1001")

    serve = sub.add_parser("serve", help="run the FastAPI server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)

    sub.add_parser("logs", help="summarise logs/runs.jsonl")

    args = parser.parse_args()
    settings = get_settings()

    if args.command == "ingest":
        from .ingest import ingest

        print(json.dumps(ingest(settings), indent=2))

    elif args.command == "seed-db":
        from .database import init_db

        print(f"{init_db(settings.db_path, reset=args.reset)} customers in {settings.db_path}")

    elif args.command == "ask":
        from .agent import SupportAgent

        reply = SupportAgent(settings).ask(args.message, customer_id=args.customer)
        print(reply.answer)
        print(f"\n[route={reply.route} tools={reply.tools_called} sources={reply.sources} {reply.latency_ms} ms]")

    elif args.command == "chat":
        from .agent import SupportAgent

        agent = SupportAgent(settings)
        session = None
        print("NileTel support copilot. Type 'exit' to quit.")
        while True:
            try:
                message = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if message.lower() in {"exit", "quit"}:
                break
            if not message:
                continue
            reply = agent.ask(message, customer_id=args.customer, session_id=session)
            session = reply.session_id
            print(f"\nagent> {reply.answer}")
            print(f"[route={reply.route} tools={reply.tools_called} sources={reply.sources} {reply.latency_ms} ms]")

    elif args.command == "serve":
        import uvicorn

        uvicorn.run("telecom_agent.api:app", host=args.host, port=args.port)

    elif args.command == "logs":
        from .monitoring import read_runs, summarize

        print(json.dumps(summarize(read_runs(settings.log_path)), indent=2))


if __name__ == "__main__":
    main()
