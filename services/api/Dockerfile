FROM node:20-slim

WORKDIR /app

COPY services/api/package.json services/api/package-lock.json* ./
RUN npm install --omit=dev || npm install

COPY services/api/tsconfig.json web /app/web
COPY services/api/src /app/src
RUN npm run build

ENV PORT=4000
EXPOSE 4000

CMD ["node", "dist/index.js"]