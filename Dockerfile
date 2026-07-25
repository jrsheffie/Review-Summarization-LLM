# Dockerfile
#
# Covers the CPU-friendly parts of this project: data preprocessing, tests,
# and inference via src/model_runner.py (both --model bart and --model llm
# run fine on CPU, just slower than on a GPU).
#
# NOTE: BART + LoRA *training* (training/train_bart.py) is GPU-dependent and
# intended to be run in Google Colab (see notebooks/01_train_bart.ipynb) --
# it is not wired up for containerized training here. This image is for
# reproducing the data pipeline and running inference/tests in an isolated
# environment, not for fine-tuning.

FROM python:3.11-slim

WORKDIR /app

# System deps some packages (e.g. langdetect, torch) may need at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run the data pipeline on the committed sample dataset, then the
# model pipeline in BART mode (no API key required, falls back to the base
# pretrained model since the fine-tuned checkpoint isn't included in the image).
CMD ["sh", "-c", "python data/preprocess.py --input data/raw/sample_reviews.csv --output-dir data/processed && python src/model_runner.py --model bart --n-samples 5"]
