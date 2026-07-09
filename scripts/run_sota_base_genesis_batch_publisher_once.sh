#!/usr/bin/env bash
set -euo pipefail

AWS_PROFILE_NAME="${AWS_PROFILE:-moonrocklab-frankfurt}"
AWS_REGION_NAME="${AWS_REGION:-eu-central-1}"
INDEXER_SECRET_ID="${SOTA_INDEXER_ADMIN_SECRET_ID:-base-sota/test/base-sepolia/indexer-admin-token}"
ROOT_PUBLISHER_SECRET_ID="${SOTA_ROOT_PUBLISHER_SECRET_ID:-base-sota/test/base-sepolia/root-publisher}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ADMIN_TOKEN="$(
  python3 scripts/sota_secret_value.py \
    --env SOTA_BASE_INDEXER_ADMIN_TOKEN \
    --secret-id "$INDEXER_SECRET_ID" \
    --field admin_token \
    --field token \
    --aws-profile "$AWS_PROFILE_NAME" \
    --aws-region "$AWS_REGION_NAME"
)"

ROOT_PUBLISHER_PRIVATE_KEY="$(
  python3 scripts/sota_secret_value.py \
    --env SOTA_ROOT_PUBLISHER_PRIVATE_KEY \
    --secret-id "$ROOT_PUBLISHER_SECRET_ID" \
    --field root_publisher_private_key \
    --field private_key \
    --aws-profile "$AWS_PROFILE_NAME" \
    --aws-region "$AWS_REGION_NAME"
)"

export SOTA_BASE_INDEXER_ADMIN_TOKEN="$ADMIN_TOKEN"
export SOTA_ROOT_PUBLISHER_PRIVATE_KEY="$ROOT_PUBLISHER_PRIVATE_KEY"

exec python3 scripts/sota_base_genesis_batch_publisher.py \
  --once \
  --broadcast \
  --import-artifact \
  --mark-included \
  --report-out /home/mekaneeky/repos/.sota-base-testnet/base-sota-genesis-batch-publisher.json
