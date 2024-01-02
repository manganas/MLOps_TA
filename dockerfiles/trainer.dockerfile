# Base image
FROM python:3.10-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Build specific

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mnist/ mnist/
COPY data/ data/


WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -e .

ENTRYPOINT ["python", "-u", "mnist/train_model.py"]