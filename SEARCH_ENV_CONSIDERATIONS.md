# SEARCH_ENV_CONSIDERATIONS

Design-assumption log for the segmentation-error **search environment** (agent
finds proofreading-worthy sites on rolled-back FlyWire objects). Append-only:
each section is dated; revisit assumptions when evidence arrives. Companion
probes: `scripts/harvest_edit_history.py`, `scripts/probe_historical_roots.py`,
`scripts/probe_snapshot_date.py`; pilot data `edit_sites_pilot.parquet`.

## 2026-09-14 — Verified data-source facts

- Edit histories are tied to **operation_ids** on the **root-id lineage DAG**.
  Root ids are version handles (retired by every edit); **supervoxels and the
  op coordinates are the persistent substrate** — coordinates are absolute in
  the immutable voxel frame and mean the same location forever.
- Change log carries `before_root_ids` / `after_root_ids` / user affiliation
  directly. Historical roots resolve (`get_leaves`) and their **meshes serve**
  via the same graphene source the native renderer uses (verified: 51,585-vert
  mesh for a pre-edit root). Root@timestamp resolution
  (`get_roots(sv, timestamp=T)`) works on the public datastack.
- **Coordinate gotcha**: ChunkedGraph op coords are 16×16×40 nm voxels —
  ×4 in x/y to reach the 4×4×40 skeleton grid (harvester converts at source).
- `flywire_public` has **no L2-cache service** → spawn points for historical
  roots come from **mesh vertices** (fetched anyway; subsample for spread).
  Lab-production l2cache is *inferred* available, unverified — ask the lab.
- Pilot (200 eval-pool neurons): 16,391 op coordinates, 194/200 roots, median
  9 ops/neuron (q25 4 / q75 20 / max 142), 70% splits / 30% merges. Edit-site
  z-fraction ~uniform (median 0.58) — not the old z-max task.
- Frozen-date sweep: edits-after-T per neuron median 9/8/6/1/0 and ≥3-flag
  coverage 86/81/73/44/6 % for T = 2020-10 / 2021-06 / 2022-01 / 2022-09 /
  2023-06. **Early states are mostly OVER-segmented (small fragments later
  merged in)**; no merge-monster in the size@T sample (max ~2,200 L2 nodes).

## 2026-09-14 — Task design decisions (v1)

- **World**: one global frozen timestamp T per bank (candidate band
  2020-10 … 2022-01; early T maximizes flags). Objects = roots@T, size-banded
  by L2 count (min ~50: excludes degenerate stubs whose whole extent is inside
  ε of their own flags; max ~3,000: merge-monster/VRAM insurance).
- **Flags**: the flattened global annotation table (all op coordinates, no
  time-gating — USER DECISION), spatially joined per object at bank-build
  time (flag must lie on the object's geometry). The geometric join
  incidentally drops most temporally-invalid flags (revert chains reference
  geometry absent at T). Static files throughout; zero runtime API calls.
- **Detection mechanism**: **declare verb** (new head; hierarchical policy
  already supports variable verb counts). **Terminate on successful
  declaration or TimeLimit; wrong declares NEVER terminate** (they cost a
  small fixed penalty and the episode continues — terminating wrong declares
  would let a random policy self-terminate at step 1 and kill search
  learning). Success = declare within ε of an on-object flag; ε in **nm**
  (anisotropy-scaled ×(4,4,40)), ~3 µm primary with 1.5/3/6 µm reported.
  Rationale vs alternatives: dwell-K was the fallback (no spam equilibrium,
  zero new heads) — terminate-on-success declare keeps the existing
  success@budget eval apparatus verbatim while adding precision-from-
  wrong-declares for free. The fixed-N multi-flag variant (non-terminating
  correct declares, once-per-flag redemption, P/R metrics) is specified as
  v1.1 if recall-style training is wanted.
- **Penalty schedule**: small fixed FP penalty (~-0.02 vs +1.0 success) —
  stationary objective, readable return curves. Verb-collapse risk (v7 zoom
  lesson) is low because wrong declares don't terminate and per-head entropy
  normalization keeps the verb explored. **Penalty ANNEALING is reserved for
  the phase-2 precision fine-tune** (short horizon, on a search-competent
  checkpoint), not used from scratch.
- **Shaping**: Δ(distance to nearest flag), success bonus on terminal
  declare. TimeLimit with the standard dual-budget (300/600) reporting.

## 2026-09-14 — Found-state tracking: DEFERRED (USER DECISION)

**v1 trains on a fully static candidate set — no filtering of completed/found
states.** Rationale: with batch ≈ 4k steps (~15–40 episodes/iter) and a bank
of ≥ ~2–5k objects, a full run (~20k episodes) visits each object ~4× under
different spawns/orientations/zooms — per-flag memorization is out of scale,
and same-batch "conflicts" (two episodes on one object per update window)
are ~6 %/iter at B=5k and harmless (one iteration of reward staleness).

- **REQUIRED EVAL (build with v1)**: track whether successful identifications
  are **biased toward a specific subset** — find-rate concentration per flag,
  per object, and per op-type (split vs merge; size/era strata). This is the
  tripwire that decides whether any dynamic mechanism is ever needed.
- **If easy-error bias emerges**: preferred remedy is **curriculum over
  successively NEWER frozen states** (train on later T banks where presumably
  the easy errors are already fixed and remaining flags are harder) — cheap,
  since every state pre-exists as a real root and bank-building is a re-join.
- Dynamic alternatives analyzed and shelved:
  - *Flag retirement (neutral zones)*: found flags pay 0 / don't terminate
    (never penalize — that trains the detector to ignore real errors). Exact
    mastery correspondence, no geometry change; world never visibly cleans up.
  - *Chronological pointer-advance*: advance an object one lineage state per
    refresh; aggregate curriculum, mastery↔removal correspondence broken;
    each advance requires a spatial re-join (geometry changes). Implement as
    periodic offline bank refresh (driver-side found events → rewrite bank →
    providers reload via mtime), never live cross-worker state.
  - *Advance-to-found-op* (jump to the youngest state incorporating the found
    edit): drains fast — each find consumes ~half the remaining flags →
    object exhausted in ~3–4 successful finds (median-9-op objects); fine for
    one run at B≥5k, a drained-object sampling problem for small banks or
    multi-run reuse (drained objects must rotate out of sampling).

## 2026-09-14 — Out-of-order / specific-edit states

Stored roots exist only along the true edit sequence; "state@T + only op_k"
is synthesizable per op type:

- **Merge ops: essentially free** — load BOTH parties' roots at T in the
  segment list; the union renders as the freshly-merged object (seam and all).
- **Split ops: possible with real work** — recover the cut's partition from
  the after-roots' leaf sets ∩ A@T's leaves, then assemble a custom object
  from per-L2-chunk mesh fragments (graphene meshing is L2-granular
  underneath). A bank-builder capability, not a config flag. Deferred until
  something needs it.

## 2026-09-17 — Rendering pre-edit states: Chrome vs Simulator backend

Worked out (the hard way, via ngl.flywire.ai) how to *display* a historical
pre-edit object. It splits the two backends sharply.

**NG-link facts** (ngl.flywire.ai = the datastack's `viewer_site`; reference
generator `scripts/gen_search_error_links.py`, START/EVAL pairs):
- Source `graphene://https://prodv1.flywire-daf.com/segmentation/1.0/flywire_public`.
  CAVE's `segmentation_source()` returns the alias `prod.flywire-daf.com`,
  which is **server-only**; the browser host is **prodv1**. **No `middleauth+`
  prefix** in a logged-in browser — that path returns empty ("HTTP error 0");
  `middleauth+` is only for token/CloudVolume contexts.
- Layer type must be **`segmentation_with_graph`** (vanilla NG's
  `segmentation` is rejected).
- **Historical root IDs load directly** in `segments` — a root id pins its
  agglomeration version forever; **no `timestamp` needed** (the timestamp
  field actively interfered). Loading `before_root_ids` IS the un-fixed
  geometry.
- ngl.flywire.ai runs the **OLD NG state format**: `navigation.pose.position.
  voxelCoordinates` + `voxelSize`; the modern top-level `dimensions`/`position`
  are silently ignored.
- Graphene is **AUTH-GATED** (FlyWire login). The public precomputed snapshot
  (`gs://flywire_v141_m783`) **cannot** show pre-edit roots — only graphene can.

**Simulator backend (native renderer) — the natural fit.** It renders from
MESHES fetched via CloudVolume from the graphene source, and CloudVolume
fetches **historical-root meshes with the server-side CAVE token**
(`~/.cloudvolume/secrets/cave-secret.json`) — verified. No browser, no
middleauth handshake. The env just needs `before_root_ids` and fetches their
meshes exactly as it fetches current roots today. ⇒ **STRONGLY PREFERRED for
the edit-search training env**; the pre-edit world is a static bank of
`(before_root_ids, spawn vertices, flag coords)` and loads with zero auth
ceremony.

**Chrome backend (ChromeRenderer) — needs auth work.** Its default config
uses the public precomputed source ⇒ cannot show pre-edit roots. Rendering
them requires headless Chrome to load the graphene (`segmentation_with_graph`)
source, which is middleauth-gated. Headless has no interactive login ⇒ the
CAVE token must be injected into the browser session (cookie/localStorage) or
a token-bearing source used. ngllib's `default_middle_auth_start_url` hints at
this path, but headless token injection is **UNBUILT/UNVERIFIED** — real
fragility. (EM stays public precomputed, no auth; only segmentation needs the
token.)

**Takeaway:** build the edit-search TRAINING env on the **simulator backend**
(mesh-based, token-authed server-side). Reserve the **Chrome backend** for the
CUA-baseline comparison, where a real logged-in browser session handles
middleauth interactively — which is exactly what the CUA-baseline branch does.

## Open items

- Lab asks: full operation-log export (avoids the ~5-day full-corpus API
  crawl; a 2–5k-object bank is a ~5 h polite crawl meanwhile); production
  datastack access / l2cache availability.
- Choose T (probe favors 2020-10…2022-01) and the size band; build the bank
  builder (roots@T + mesh prefetch to local store + spawn-vertex subsample +
  flag spatial join) and the reward/termination factory pair + declare verb.
- Eval pool: frozen (object@T, spawn, flags) set built from held-out neurons
  (training bank must exclude the 200-neuron eval pool).
- Phase 2 (later): declare-precision fine-tune with annealed FP penalty and
  terminate-on-declare switched in late.
