FROM python:3

ADD code/URMiner/run.py ./

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt
RUN python -m spacy download fr

COPY ./code/RationalyticsFramework ./RationalyticsFramework
COPY ./code/URMiner ./URMiner

EXPOSE 9704
CMD [ "python", "./run.py"]