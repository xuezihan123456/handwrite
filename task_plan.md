# Task Plan: 2026-04-11 Classroom Note Realism Foundation

## Goal
Push HandWrite toward a user-facing classroom-note product by adding a prototype-backed realism foundation, a precheck/report loop, and a more honest note-generation workflow without blocking on shipping the full 2000+ asset body now.

## Current Phase
Complete

## Phases

### Phase 1: Product Direction Lock
- [x] Read docs, source, tests, and current README files
- [x] Run the iterative product interview and lock the classroom-note direction
- [x] Save the main decisions to OMX memory and planning files
- **Status:** complete

### Phase 2: Design & Execution Setup
- [x] Write the design doc to `docs/superpowers/specs/2026-04-11-classroom-note-realism-design.md`
- [x] Write the implementation plan to `docs/superpowers/plans/2026-04-11-classroom-note-realism-foundation.md`
- [x] Split the work into independent lanes for parallel execution
- [x] Spawn multi-agent workers for core/runtime and API/demo lanes
- [ ] Update planning files to reflect the new execution plan
- **Status:** in_progress

### Phase 3: Prototype Runtime Foundation
- [ ] Add a prototype-library runtime module
- [ ] Add a starter built-in prototype pack
- [ ] Add a local builder script for future larger packs
- [ ] Extend engine routing to prefer real model -> prototype glyph -> improved fallback
- **Status:** pending

### Phase 4: Public API & Demo Loop
- [ ] Expose note-realism inspection from the public API
- [ ] Change defaults toward the flowing classroom-note style
- [ ] Add demo precheck/report output while preserving multi-page preview/downloads
- **Status:** pending

### Phase 5: Docs, Packaging, and Verification
- [ ] Update package data configuration if assets are shipped
- [ ] Rewrite README messaging around classroom-note generation and realism precheck
- [ ] Run focused pytest commands for changed areas
- [ ] Run a combined regression command and record results in `progress.md`
- [ ] If needed, run a gstack-style review/health pass before final summary
- **Status:** pending

## Key Decisions
1. Product wedge: continuous paragraph classroom notes, not generic handwriting or training UX.
2. Default style should feel natural and flowing, readable, and slightly imperfect.
3. The first realism problem to fix is print-looking single characters.
4. This iteration ships the product system, not the full 2000+ built-in asset body.
5. Use a hybrid implementation: prototype-library foundation + minimal precheck/report loop.
6. Do not silently degrade or auto-rewrite user content.

## Risks
| Risk | Mitigation |
|------|------------|
| Small starter pack may undersell the vision | Improve fallback rendering and expose precheck coverage/reporting |
| Parallel edits may conflict | Keep disjoint file ownership and integrate carefully |
| Current environment blocks some temp dirs | Continue using repository-controlled temp locations for pytest |

## Notes
- User explicitly asked for autonomous execution with no more blocking questions.
- User explicitly asked for multi-agent parallel development using GPT-5.4 with xhigh reasoning.
- Sandbox cannot be globally changed from inside the session; request escalation inline only when needed.


### Phase 6: Custom Prototype Pack Product Loop
- [x] Connect the local prototype library builder to the public runtime API
- [x] Let the demo accept a custom prototype pack directory or manifest path
- [x] Show the active prototype pack/source in the precheck report
- [x] Re-run focused verification and full regression after the integration
- **Status:** complete


### Phase 7: Product Workflow Polish
- [x] ? demo ?????? preset / ??????
- [x] ??? `prototype_pack` ?????????
- [x] ?? README ??????? preset ???????
- [x] ??????????????
- **Status:** complete


### Phase 8: Note Session Productization
- [x] ????????? session ???/????
- [x] ? demo ??????? session report / status / artifacts
- [x] ?????????? -> ?? -> ????????????
- [x] ?????????????
- **Status:** complete


### Phase 9: Note Session CLI Product Entry
- [x] ??????????????? CLI
- [x] ?? inline text / text file / preset / prototype_pack / output_dir
- [ ] ????? PNG / PDF / session report
- [x] ??????????????????
- **Status:** in_progress
