FROM python:3.12-slim

WORKDIR /app

#COPY vtide/ ./vtide/
COPY final_tide_constituent.py ./
COPY pytmd_tide_model.py ./
COPY main.py ./
COPY requirements.txt ./

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment, then install Python dependencies
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt \
    && /opt/venv/bin/pip install fastapi uvicorn

ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
