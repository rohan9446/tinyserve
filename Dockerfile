FROM python:3.11-slim

WORKDIR /app

# install build tools for C++ sampler
RUN apt-get update && apt-get install -y g++ && rm -rf /var/lib/apt/lists/*

# copy and install dependencies
COPY setup.py .
COPY core/ core/
COPY server/ server/
COPY export/ export/
COPY serve.py .

# compile C++ sampler for Linux
RUN g++ -shared -O2 -fPIC -o core/sampler.so core/sampler.cpp

# install Python deps
RUN pip install --no-cache-dir -e .

# download model at build time
RUN python -c "from export.convert import download_model; download_model('TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'models_tinyllama')"

EXPOSE 8000

CMD ["python", "serve.py", "--model", "models_tinyllama", "--port", "8000"]