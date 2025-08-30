# Simple Dockerfile to run the Streamlit app
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "src/app/ui_streamlit.py", "--server.address=0.0.0.0", "--server.port=8501"]
