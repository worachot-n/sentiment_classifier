FROM python:3.10-slim

WORKDIR /code

RUN apt-get update -y && apt-get install -y python3-dev build-essential

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
