# Codebase Recommendations

## Overall Assessment

The project is in a good place for a serious prototype. It has:

- a real config-driven architecture
- packaging and test coverage
- config validation
- useful observability
- a clearer separation between generator, runtime, tools, and evaluation layers

The biggest remaining improvements are not emergency fixes. They are mostly about reducing future complexity and pushing the runtime closer to a truly spec-driven system.

## Highest-Impact Recommendations

### 1. Refactor the Runtime Into Smaller Internal Components

`src/engine/cognitive_engine.py` is becoming the main concentration point for complexity.

It currently owns:

- operator selection
- slot mutation
- interruption handling
- affirmation mapping
- impasse handling
- cycle execution
- metric recording

Recommended next step:

- split engine logic into smaller helpers or modules such as:
  - state mutation / slot update helpers
  - routing interruption handling
  - operator selection / cycle execution
  - engine metrics recording

Why:

- lowers maintenance risk
- makes spec-driven refactors easier
- improves testability

### 2. Add Structured Rule Traceability

The system currently exposes recommendation rationale text, but not full structured rule traces.

Recommended next step:

- surface explicit rule ids in tool output
- track which rule matched
- track when fallback happened
- optionally expose rejected candidate rules for debugging

Why:

- improves explainability
- supports auditability
- makes the system more genuinely neural-symbolic
- helps move behavior from “prompt explanation” to “spec-backed explanation”

### 3. Keep Removing Hardcoded Orchestration Assumptions

The repo is more spec-driven than before, but Python still defines key runtime semantics.

Still hardcoded in practice:

- what `nlu`, `action`, and `orchestration` mean
- how phase progression works
- how confirmation and interruption behavior is applied
- how operator execution semantics are interpreted

Recommended next step:

- continue moving runtime decision behavior into config boundaries where practical
- prefer config-defined behavior switches over Python-side special cases

Why:

- gets closer to the original “change the CSVs, retarget the system” goal

### 4. Consider a Longer-Lived Runtime Model

The current CLI flow works, but a short-lived process is awkward for:

- live observability
- multi-session concurrency
- cleaner session lifecycle handling
- realistic deployment behavior

Recommended next step:

- eventually move from a pure CLI session model toward a longer-lived service or session manager

Why:

- simplifies metrics
- improves multi-user behavior
- makes observability more reliable

## Medium-Impact Improvements

### 5. Split Decoder Responsibilities

`src/conversation/decoder.py` still combines:

- prompt construction
- schema scoping
- DSPy invocation
- JSON extraction/parsing
- validation
- telemetry

Recommended next step:

- separate prompt building, output parsing, and validation/telemetry into smaller units

Why:

- easier maintenance
- easier debugging
- cleaner future extension

### 6. Keep Strengthening Spec Validation

Validation is already useful, but it can grow with the spec-driven architecture.

Recommended next step:

- validate more agent config semantics
- validate rule traceability fields if added
- validate orchestration-specific config if more runtime behavior moves into spec

Why:

- prevents broken specs from failing late at runtime

### 7. Keep Dashboard Layers Separate

The current split between a full diagnostic dashboard and a cleaner demo dashboard is the right pattern.

Recommended next step:

- keep demo dashboards audience-focused
- keep diagnostic dashboards deeper and noisier
- avoid merging both use cases into one board

Why:

- supports both standups and debugging without clutter

## Current Main Risks

These are the most notable gaps still visible today:

- interruption recovery is improved but still not perfect in messy mid-flow conversations
- runtime behavior is still only partially spec-driven
- engine complexity is growing faster than the other layers
- observability is strong, but still somewhat handcrafted

## Suggested Priority Order

1. Refactor the engine into smaller pieces
2. Add structured rule traceability with explicit rule ids
3. Externalize more orchestration behavior into spec/config
4. Eventually move to a longer-lived runtime model
