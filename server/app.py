import json
import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse
from server.engine import InferenceEngine

app = FastAPI(title="TinyServe", version="0.1.0")
engine = InferenceEngine()

# track metrics
metrics = {
    "total_requests": 0,
    "total_tokens": 0,
    "avg_ttft_ms": 0,
    "avg_tokens_per_sec": 0,
}


@app.get("/health")
def health():
    return {"status": "ok", "model": "loaded"}


@app.post("/generate")
async def generate(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    max_tokens = body.get("max_tokens", 50)
    temperature = body.get("temperature", 0.7)
    top_k = body.get("top_k", 50)
    top_p = body.get("top_p", 0.9)

    tokens = []
    start = time.perf_counter()

    for token in engine.generate(prompt, max_tokens, temperature, top_k, top_p):
        tokens.append(token)

    text = "".join(tokens)
    elapsed = (time.perf_counter() - start) * 1000

    metrics["total_requests"] += 1

    return {"text": text, "elapsed_ms": round(elapsed, 1)}


@app.post("/generate/stream")
async def generate_stream(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    max_tokens = body.get("max_tokens", 50)
    temperature = body.get("temperature", 0.7)
    top_k = body.get("top_k", 50)
    top_p = body.get("top_p", 0.9)

    async def token_stream():
        for token in engine.generate(prompt, max_tokens, temperature, top_k, top_p):
            yield {"data": json.dumps({"token": token})}

    return EventSourceResponse(token_stream())


@app.get("/metrics")
def get_metrics():
    return metrics


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
    <html><head><title>TinyServe</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #0f0f0f; color: #e0e0e0; margin: 40px; }
        h1 { color: #00c896; }
        textarea { width: 100%; height: 80px; background: #1a1a1a; color: #e0e0e0; border: 1px solid #333; padding: 10px; font-size: 14px; }
        button { background: #00c896; color: #0f0f0f; border: none; padding: 10px 24px; font-size: 14px; cursor: pointer; margin-top: 8px; }
        #output { margin-top: 16px; padding: 16px; background: #1a1a1a; border: 1px solid #333; min-height: 100px; white-space: pre-wrap; }
        #stats { margin-top: 12px; color: #888; font-size: 13px; }
    </style></head>
    <body>
        <h1>TinyServe</h1>
        <textarea id="prompt" placeholder="Enter your prompt...">Once upon a time</textarea>
        <br><button onclick="generate()">Generate</button>
        <div id="output"></div>
        <div id="stats"></div>
        <script>
        async function generate() {
            const prompt = document.getElementById('prompt').value;
            const output = document.getElementById('output');
            const stats = document.getElementById('stats');
            output.textContent = '';
            stats.textContent = 'Generating...';
            const start = performance.now();

            const res = await fetch('/generate/stream', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt: prompt, max_tokens: 100, temperature: 0.7})
            });

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let tokenCount = 0;

            while (true) {
                const {done, value} = await reader.read();
                if (done) break;
                const text = decoder.decode(value);
                const lines = text.split('\\n');
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            output.textContent += data.token;
                            tokenCount++;
                        } catch(e) {}
                    }
                }
            }

            const elapsed = ((performance.now() - start) / 1000).toFixed(1);
            stats.textContent = tokenCount + ' tokens in ' + elapsed + 's';
        }
        </script>
    </body></html>
    """