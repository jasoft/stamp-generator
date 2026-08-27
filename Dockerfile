FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY simsun.ttc .
COPY stamp_webapp.py .

EXPOSE 5000

CMD ["python", "stamp_webapp.py"]
