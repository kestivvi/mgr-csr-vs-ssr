FROM node:25.9.0-alpine3.23 AS development-dependencies-env
COPY . /app
WORKDIR /app
RUN npm ci

FROM node:25.9.0-alpine3.23 AS production-dependencies-env
COPY ./package.json package-lock.json /app/
WORKDIR /app
RUN npm ci --omit=dev

FROM node:25.9.0-alpine3.23 AS build-env
COPY . /app/
COPY --from=development-dependencies-env /app/node_modules /app/node_modules
WORKDIR /app
RUN npm run build

FROM node:25.9.0-alpine3.23
COPY ./package.json package-lock.json /app/
COPY --from=production-dependencies-env /app/node_modules /app/node_modules
COPY --from=build-env /app/build /app/build
WORKDIR /app
CMD ["npm", "run", "start"]