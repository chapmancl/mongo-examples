FROM node:20-slim AS frontend-build

WORKDIR /build/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


FROM python:3.13-slim
RUN apt-get update && apt-get install -y curl
WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt requirements.txt
COPY app.py app.py
COPY mcp_processor.py mcp_processor.py
COPY local_settings.py local_settings.py
COPY --from=rootfolder ./mongomcp/ mongomcp/
COPY --from=rootfolder ./mongoagent/ mongoagent/

RUN pip install --no-cache-dir ./mongomcp ./mongoagent
RUN pip install -r requirements.txt

COPY --from=frontend-build /build/frontend/dist frontend/dist

EXPOSE 8001

CMD ["gunicorn", "-w", "2", "-k", "gthread", "--threads", "4", "-b", "0.0.0.0:8001", "app:app", "--timeout", "300"]
