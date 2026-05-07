import argparse
import os
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="TinyServe — LLM Inference Server")
    parser.add_argument("--model", default="models", help="Path to model directory")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    args = parser.parse_args()

    os.environ["MODEL_DIR"] = args.model
    print(f"\nTinyServe — Starting server")
    print(f"  Model: {args.model}")
    print(f"  Address: http://{args.host}:{args.port}\n")

    uvicorn.run("server.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()