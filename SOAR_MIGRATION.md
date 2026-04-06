# Path To A Real Soar-Style System

This repo is now meaningfully Soar-style, but it is not a full Soar system yet.

The major steps still required are:

1. Make execution control part of symbolic state.
   Status: In progress.
   Pending and interrupted operators are now projected into working memory as `control.*` WMEs, and current goal / subgoal state is projected there as well.

2. Move dialogue progression into productions and preferences.
   Status: Partially complete.
   Operator selection is production-based, but some conversation progression still lives in engine code.

3. Expand preference semantics beyond weighted acceptable/require/reject.
   Status: Partial.
   The runtime supports stronger `require` and `reject`, but not the fuller Soar-style preference vocabulary.

4. Make substates own impasse resolution behavior.
   Status: Partial.
   Impasses and substates exist, but recovery is still a compatibility-oriented layer rather than a full subgoal system.

5. Introduce semantic and episodic memory layers.
   Status: Not started.
   The runtime currently has working memory plus CSV-backed rules, not SMEM/EpMem-style memory subsystems.

6. Add chunking / learning.
   Status: Not started.
   The system does not yet learn new productions from resolved impasses.

7. Make explanations first-class symbolic artifacts.
   Status: In progress.
   Production, policy, and causal traces exist, but they are not yet a full proof chain for every decision.

8. Separate perception, cognition, and action as independent services or durable subsystems.
   Status: Partial.
   The codebase has clearer module boundaries now, but the runtime is still a single-process CLI.

## What Was Implemented In This Migration Batch

- Added first-class control state projection into working memory via `src/engine/control.py`
- Added config-derived runtime modeling via `src/engine/runtime_model.py`
- Added a generic operator execution layer via `src/engine/operator_handler.py`
- Added a Soar-like control boundary via `src/soar/controller.py`, `src/soar/io_manager.py`, and `src/ui/agent_runner.py`
- Moved interrupted-operator resume decisions behind declarative policy rules
- Kept the current CLI and config compatibility intact

## What To Build Next

- Compile follow-up prompt sequencing into control productions
- Represent current goal / subgoal explicitly in working memory
- Add richer preference comparison and justification
- Start a real semantic-memory retrieval layer
- Add chunk creation from resolved impasses
