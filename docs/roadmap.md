<section class="bitsota-hero compact">
  <p class="bitsota-kicker">ROADMAP</p>
  <h1>Future Roadmap</h1>
  <p class="bitsota-lede">Move from operator-run competitions to a cleaner public loop: define, train, verify, pay.</p>
</section>

This roadmap is directional. It does not promise dates, rewards, uptime, or
continued availability of any endpoint or task.

## Current Foundation

The current working foundation is:

- coordinator-backed research tasks;
- signed miner claims and submissions;
- prompt-pack mining through `bitsota-research-agent`;
- validator replay through signed backend worklists;
- backend reward snapshots;
- backend-directed validator weights;
- Pool/Merkle claim publication path.

## Near-Term Product Direction

<div class="bitsota-card-grid">
  <div class="bitsota-card"><strong>Mining GUI</strong><span>Restore a guided GUI path for task discovery, wallet setup, agent launch, submission status, and claims.</span></div>
  <div class="bitsota-card"><strong>Problem-owner intake</strong><span>Turn admin-only task creation into a reviewed posting workflow.</span></div>
  <div class="bitsota-card"><strong>Validator visibility</strong><span>Expose replay backlog, validation outcomes, and weight evidence clearly.</span></div>
  <div class="bitsota-card"><strong>Claim transparency</strong><span>Make reward snapshots, Merkle epochs, recipient coldkeys, and claim status easier to inspect.</span></div>
</div>

## Protocol And Backend Direction

1. Keep the backend as the source of truth for tasks, claims, submissions,
   validation, best results, and reward snapshots.
2. Keep `SN94-BitSota` as the public client layer for miners and validators.
3. Keep task repos public, replayable, and narrow in edit surface.
4. Keep hidden/heldout replay data out of public repos.
5. Keep validator replay, Pool publishing, and chain weight setting as separate
   responsibilities.

## Problem Posting Roadmap

The current code supports admin task creation. The intended product direction is
to add a safer problem-owner flow:

- problem intake form;
- task repo/template validation;
- benchmark smoke test before publication;
- operator review;
- live/paused/closed lifecycle controls;
- public problem page with metric, benchmark, reward, and validation rules.

The policy details for self-serve posting still need product/operator approval.

## Mining Roadmap

The target miner experience is:

1. connect wallet or hotkey;
2. choose a live task;
3. choose manual, local script, or prompt-pack mode;
4. watch claim/submission/verification state;
5. inspect accepted results and claim packages.

Agent Mining should stay CLI-agnostic. The prompt pack and any `INTRO_GUI.md`
workspace contract should stay the same no matter which coding agent executes
the work.

## Validator Roadmap

Validator work should continue toward:

- simpler public validator setup;
- better replay isolation and artifact checks;
- clearer validator allowlist diagnostics;
- visible replay error categories;
- automated evidence that weights point to backend-approved targets.

## Reward And Claim Roadmap

Reward visibility should make these facts easy to answer:

- which accepted submission produced the score;
- which miner hotkey owns the accepted result;
- which recipient coldkey was published;
- which Merkle epoch contains the claim;
- whether the claim is blocked by veto/challenge state;
- which validator weight policy is currently active.
