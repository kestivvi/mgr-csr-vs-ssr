## Usage

Those templates dependencies are maintained via [pnpm](https://pnpm.io) via `pnpm up -Lri`.

This is the reason you see a `pnpm-lock.yaml`. That being said, any package manager will work. This file can be safely be removed once you clone a template.

```bash
$ npm install # or pnpm install or yarn install
```

### Learn more on the [Solid Website](https://solidjs.com) and come chat with us on our [Discord](https://discord.com/invite/solidjs)

## Available Scripts

In the project directory, you can run:

### `npm run dev` or `npm start`

Runs the app in the development mode.<br>
Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

The page will reload if you make edits.<br>

### `npm run build`

Builds the app for production to the `dist` folder.<br>
It correctly bundles Solid in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.<br>
Your app is ready to be deployed!

## Docker Deployment with Apache

This application is configured to run with Apache HTTP Server in a Docker container.

### Building and Running

```bash
# Build and start the Apache container
docker-compose up --build

# Run in detached mode
docker-compose up -d --build
```

### Access Points

- **Main Application**: [http://localhost:80](http://localhost:80)
- **Apache Status**: [http://localhost:8060/server-status](http://localhost:8060/server-status)
- **Prometheus Metrics**: [http://localhost:9117/metrics](http://localhost:9117/metrics)

### Monitoring

The container includes Apache Exporter for Prometheus monitoring:
- Apache metrics are exposed at `/metrics` endpoint on port 9117
- Apache status page is available at `/server-status` on port 8060
- Extended status is enabled for comprehensive metrics

### Logs

Apache logs are mounted to `/var/log/csr-solidjs-apache` on the host system:
- Access logs: `access.log`
- Error logs: `error.log`
- Status logs: `status.log`

## Deployment

You can deploy the `dist` folder to any static host provider (netlify, surge, now, etc.) or use the provided Docker setup with Apache.
