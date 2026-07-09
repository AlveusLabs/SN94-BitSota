#!/usr/bin/env bash
set -euo pipefail

AWS_PROFILE_NAME="${AWS_PROFILE:-moonrocklab-frankfurt}"
AWS_REGION_NAME="${AWS_REGION:-eu-central-1}"
INDEXER_SECRET_ID="${SOTA_INDEXER_ADMIN_SECRET_ID:-base-sota/test/base-sepolia/indexer-admin-token}"
ROOT_PUBLISHER_SECRET_ID="${SOTA_ROOT_PUBLISHER_SECRET_ID:-base-sota/test/base-sepolia/root-publisher}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ADMIN_TOKEN="$(
  aws secretsmanager get-secret-value \
    --profile "$AWS_PROFILE_NAME" \
    --region "$AWS_REGION_NAME" \
    --secret-id "$INDEXER_SECRET_ID" \
    --query SecretString \
    --output text |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["admin_token"])'
)"

ROOT_PUBLISHER_PRIVATE_KEY="$(
  aws secretsmanager get-secret-value \
    --profile "$AWS_PROFILE_NAME" \
    --region "$AWS_REGION_NAME" \
    --secret-id "$ROOT_PUBLISHER_SECRET_ID" \
    --query SecretString \
    --output text |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["root_publisher_private_key"])'
)"

export SOTA_BASE_INDEXER_ADMIN_TOKEN="$ADMIN_TOKEN"
export SOTA_ROOT_PUBLISHER_PRIVATE_KEY="$ROOT_PUBLISHER_PRIVATE_KEY"

exec python3 scripts/sota_base_genesis_batch_publisher.py \
  --once \
  --broadcast \
  --import-artifact \
  --mark-included \
  --report-out /home/mekaneeky/repos/.sota-base-testnet/base-sota-genesis-batch-publisher.json
