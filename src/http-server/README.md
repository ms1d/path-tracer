# HTTP-SERVER

## About

A simple HTTP server set up via cpp-httplib to provide basic health
checks + an endpoint to send scene data to be rendered by the path-tracer process.

## Endpoints

See `endpoints`.

- `GET /health`: a basic health check of the http-server process

- `POST /submit-render`: parse + validate scene data, and send to path-tracer

## Middleware

See `middleware` for implementation details.

**Important Note on Execution Flow:**

The middleware uses a "Status-based Short-Circuit" model. If any layer
(or the handler itself) sets the `res.status`, all subsequent logic is skipped.
`cpp-httplib` initialises res.status as **-1**.

- **Incoming:** Forward order.

- **Handler:** Only runs if all incoming layers left `status == -1`.

- **Outgoing:** Reverse order (FILO). Only runs if the status is **still** `-1`.
