# Middleware System

This directory contains a simplified middleware implementation
designed for basic request handling.

## Design Principles

- **Wrap Mandatory:** All routes in `controller.cpp` must be wrapped using `mw.wrap_endpoint`.

- **Status is King:** The `res.status` field (initialized to `-1`)
is used as the short-circuit trigger.
- **Simplicity over Power:** There is no `next()` call. Layers are divided into
`incoming` (before) and `outgoing` (after).

## Order of Operations

1. `incoming_layers` (0 to N)

2. `handler` (if status is -1)

3. `outgoing_layers` (N to 0)

If `status` is modified at any step, the loop terminates immediately.
