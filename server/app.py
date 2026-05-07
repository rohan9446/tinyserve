import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse
from server.engine import InferenceEngine
from server.monitor import Monitor

app = FastAPI(title="TinyServe", version="0.1.0")
engine = InferenceEngine()
monitor = Monitor()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    max_tokens = body.get("max_tokens", 50)
    temperature = body.get("temperature", 0.7)
    top_k = body.get("top_k", 50)
    top_p = body.get("top_p", 0.9)

    tokens = []
    stats = {}
    for chunk in engine.generate(prompt, max_tokens, temperature, top_k, top_p):
        if chunk["type"] == "token":
            tokens.append(chunk["text"])
        elif chunk["type"] == "stats":
            stats = chunk

    monitor.record_request(stats.get("tokens", 0), stats.get("ttft_ms", 0), stats.get("tokens_per_sec", 0))

    return {"text": "".join(tokens), **stats}


@app.post("/generate/stream")
async def generate_stream(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    max_tokens = body.get("max_tokens", 50)
    temperature = body.get("temperature", 0.7)
    top_k = body.get("top_k", 50)
    top_p = body.get("top_p", 0.9)

    async def token_stream():
        for chunk in engine.generate(prompt, max_tokens, temperature, top_k, top_p):
            if chunk["type"] == "token":
                yield {"data": json.dumps({"token": chunk["text"]})}
            elif chunk["type"] == "stats":
                monitor.record_request(chunk["tokens"], chunk["ttft_ms"], chunk["tokens_per_sec"])
                yield {"data": json.dumps({"stats": chunk})}

    return EventSourceResponse(token_stream())


@app.get("/metrics")
def get_metrics():
    return monitor.get_metrics()


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
    <html><head><title>TinyServe</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #0f0f0f; color: #e0e0e0; margin: 40px; }
        h1 { color: #00c896; }
        h2 { color: #aaa; margin-top: 24px; }
        textarea { width: 100%; height: 80px; background: #1a1a1a; color: #e0e0e0; border: 1px solid #333; padding: 10px; font-size: 14px; }
        button { background: #00c896; color: #0f0f0f; border: none; padding: 10px 24px; font-size: 14px; cursor: pointer; margin-top: 8px; }
        #output { margin-top: 16px; padding: 16px; background: #1a1a1a; border: 1px solid #333; min-height: 100px; white-space: pre-wrap; }
        #stats { margin-top: 12px; color: #888; font-size: 13px; }
        .metrics { display: flex; gap: 24px; flex-wrap: wrap; margin-top: 12px; }
        .metric { background: #1a1a1a; border: 1px solid #333; padding: 16px; min-width: 140px; }
        .metric-value { font-size: 24px; color: #00c896; }
        .metric-label { font-size: 12px; color: #888; margin-top: 4px; }
    </style></head>
    <body>
        <h1>TinyServe</h1>
        <textarea id="prompt" placeholder="Enter your prompt...">Once upon a time</textarea>
        <br><button onclick="generate()">Generate</button>
        <div id="output"></div>
        <div id="stats"></div>

        <h2>Server Metrics</h2>
        <div class="metrics" id="metrics"></div>

        <script>
        async function generate() {
            const prompt = document.getElementById('prompt').value;
            const output = document.getElementById('output');
            const stats = document.getElementById('stats');
            output.textContent = '';
            stats.textContent = 'Generating...';

            const res = await fetch('/generate/stream', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt: prompt, max_tokens: 100, temperature: 0.7})
            });

            const reader = res.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const {done, value} = await reader.read();
                if (done) break;
                const text = decoder.decode(value);
                const lines = text.split('\\n');
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.token) output.textContent += data.token;
                            if (data.stats) {
                                stats.textContent = 'TTFT: ' + data.stats.ttft_ms + 'ms | '
                                    + data.stats.tokens + ' tokens | '
                                    + data.stats.tokens_per_sec + ' tok/s | '
                                    + data.stats.total_ms + 'ms total';
                            }
                        } catch(e) {}
                    }
                }
            }
            refreshMetrics();
        }

        async function refreshMetrics() {
            const res = await fetch('/metrics');
            const m = await res.json();
            document.getElementById('metrics').innerHTML =
                metric('Requests', m.total_requests) +
                metric('Tokens', m.total_tokens_generated) +
                metric('Avg TTFT', m.avg_ttft_ms + ' ms') +
                metric('Avg Speed', m.avg_tokens_per_sec + ' tok/s') +
                metric('Memory', m.memory_mb + ' MB') +
                metric('Uptime', m.uptime_seconds + ' s');
        }

        function metric(label, value) {
            return '<div class="metric"><div class="metric-value">' + value + '</div><div class="metric-label">' + label + '</div></div>';
        }

        refreshMetrics();
        setInterval(refreshMetrics, 5000);
        </script>
    </body></html>
    """