# Project Log

This log records reader-visible Base SOTA docs milestones. Internal issue IDs,
private deployment details, and fragmented developer test commands do not belong
on this public-facing page.

## 2026-07-04

The Base SOTA docs were reshaped around two audiences:

- nontechnical readers who are new to BitSota or Bittensor;
- Bittensor migrants who need to understand what changes on Base.

The local demo path is now documented as one command:

```bash
cd /home/mekaneeky/repos/SN94-BitSota-live-docs
./scripts/sota_local_demo.py launch
```

The documented demo starts the local EVM, indexer, autoresearch backend,
website, and docs. It seeds demo users, runs miners through self-validation,
and prints the URLs a reviewer should open.

Reader-visible status:

| Area | Status |
| --- | --- |
| Local demo docs | Ready for the launcher command. |
| New-user explanation | Added. |
| Bittensor migration explanation | Added. |
| Support FAQ and triage wording | Drafted for Base Sepolia review. |
| Public deployment instructions | Not published yet. |
| Mainnet claim instructions | Not published yet. |

Remaining launch gaps:

- Base Sepolia or Base mainnet deployment manifest;
- multisig or timelock ownership transfer evidence;
- live Base claims/indexer service;
- monitoring;
- funded browser wallet smoke;
- external audit and fuzz-expansion evidence.
