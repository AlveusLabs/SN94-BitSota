from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--host", default=os.getenv("BITSOTA_SIDECAR_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("BITSOTA_SIDECAR_PORT", "8123")),
    )
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "sidecar.main:app",
        host=str(args.host),
        port=int(args.port),
        reload=False,
        access_log=False,
        log_level=str(os.getenv("BITSOTA_SIDECAR_LOG_LEVEL", "warning")).lower(),
    )


if __name__ == "__main__":
    main()
