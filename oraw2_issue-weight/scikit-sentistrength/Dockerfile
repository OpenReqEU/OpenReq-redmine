FROM python:3.6

RUN apt-get update
RUN apt-get install -y default-jre

RUN pip install scikit-learn==0.21.1 pandas flask gunicorn

WORKDIR /app/src
COPY SentiStrengthCom.jar /app/sentiStrengthCom.jar
COPY data /app/data
COPY src /app/src

CMD ["gunicorn", "-b", "0.0.0.0:8080", "--access-logfile",  "-", "main:app"]