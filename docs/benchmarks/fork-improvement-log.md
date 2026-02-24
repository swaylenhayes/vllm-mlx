# Fork Improvement Log (Measured, Append-Only)

This log is the ongoing record of measured backend improvements in this fork.

Rules:
- Append new entries at the top.
- Only include entries with reproducible measurement notes.
- Keep claims bounded to what was measured.

## Entries

### 2026-02-24 - `d890ef6`

- Change:
  - Added LiquidAI/LFM tool parser (`liquidai`, `liquid`, `lfm`)
  - Fixed reasoning/tool parser ordering in streaming and non-streaming paths
  - Added thinking budget controls (`--max-thinking-tokens`, request `max_thinking_tokens`)
- Measurement status: functional validation complete, throughput delta not yet benchmarked in phase table
- Validation:
  - Tool parser/ordering/tests passed in targeted suite
  - Reasoning budget controls validated at request/CLI contract level
- Notes:
  - Throughput impact should be recorded after next benchmark round

### 2026-02-24 - `140377c`

- Change:
  - Added `frequency_penalty` request support (mapped to backend repetition penalty)
- Measurement status: validated behavior
- Validation:
  - Request model, server mapping, and tests updated and passing
  - Confirmed usable as repetition tuning control from API clients

## Entry Template

Copy this block for each new measured improvement:

```md
### YYYY-MM-DD - `<commit>`

- Change:
  - <short summary of backend change>
  - <optional second bullet>
- Measurement status: <benchmarked|validated behavior|pending benchmark>
- Measurement setup:
  - Model(s): `<model id(s)>`
  - Command/profile: `<serve/bench command>`
  - Workload: <prompt/tool/request profile>
- Baseline:
  - Commit: `<baseline commit>`
  - Key metric(s): <value(s)>
- Result:
  - Commit: `<new commit>`
  - Key metric(s): <value(s)>
  - Delta: <percent/value delta>
- Caveats:
  - <what this does not prove>
- Links:
  - <docs/tests/benchmark artifact paths>
```
