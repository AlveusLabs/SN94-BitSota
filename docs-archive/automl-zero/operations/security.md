# Security

## Keys

- Never share seed phrases or private keys
- Keep coldkeys offline
- Prefer per-role hotkeys for local testing

## API auth

Relay and Pool authenticate requests using signed headers. Treat your hotkey like an API credential:

- do not log signatures in public logs
- do not reuse hotkeys across roles in production

## Relay admin token

- Treat `ADMIN_AUTH_TOKEN` like a password
- Rotate it if exposed

## Local testing

- Use dedicated test wallets and hotkeys
- Run services on localhost unless you explicitly need LAN access
