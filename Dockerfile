FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN /usr/local/bin/python -m pip install --no-cache-dir --disable-pip-version-check -r requirements.txt
COPY . .
ENV PATH=""
ENTRYPOINT ["/usr/local/bin/python", "/app/run.py"]
