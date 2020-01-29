FROM python:3.6

RUN pip install flask gunicorn

WORKDIR /app/src
COPY . /app/src

CMD ["gunicorn", "-b", "0.0.0.0:8081", "--access-logfile",  "-", "main:app"]