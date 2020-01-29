FROM node:9

COPY . /app
WORKDIR /app

RUN yarn
ENTRYPOINT [ "node", "server.js" ]