FROM python:3.10

WORKDIR /service

COPY . /service/

RUN pip install -r requirements.txt --no-cache-dir

EXPOSE 5000

CMD ["streamlit", "run", "front.py"]
