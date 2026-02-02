# Troubleshooting

## Relay database is read-only

Use an explicit database path you own:

```bash
python3 -m relay --test --database-url "sqlite:///./bitsota_relay_test.db"
```

## Sidecar port already in use

Set a different port before starting the GUI:

```bash
export BITSOTA_SIDECAR_PORT=8124
```

## Pool auth errors

Pool endpoints require `X-Key`, `X-Timestamp`, and `X-Signature`, and the timestamp must be within 5 minutes of server time.

If you are doing manual tests, use the scripts in `scripts/` that already generate headers instead of hand-crafting requests.
