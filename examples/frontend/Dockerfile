FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
   openai \
   streamlit \
   boto3 \
   dotenv

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]