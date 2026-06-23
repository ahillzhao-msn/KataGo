# KataGo — Trunk Feature Extraction Fork — Customization Guide

This document describes the customizations applied on top of vanilla [lightvector/KataGo](https://github.com/lightvector/KataGo) upstream `master` branch.

## What Was Added

### 1. `batch_analysis` Command (`command/batch_analysis.cpp`, new)

A self-contained module (~890 lines) that provides batch SGF analysis with trunk feature extraction:

- Accepts `-list games.csv` or multiple CSV paths
- For each SGF: loads game, extracts main line, skips < 10 moves
- For each position: evaluates with NNEvaluator, collects head(12)+trunk(256)+pick(256) features
- Writes per-player KAB2-formatted NPZ binary output (zlib-compressed)
- Handles errors per-file and continues

### 2. Stream Mode (`-stream`, `-no-trunk`)

Pipes KAB2 frames to stdout (zero disk I/O):

- Protocol: `[1B 'B'/'W'][4B idLen][game id][4B size][KAB2 payload]`
- Terminated by `0x01` (task complete) and `0x00` (process end)
- Lite mode (`-no-trunk`): scalars only (10 floats/move), 100x smaller

### 3. Daemon Mode (`-daemon`)

Model-resident persistent process:

- stdin interface: one games.csv path per line
- `reset` clears batch cache; `quit` exits
- Task latency drops from ~10s+ cold start to analysis-only time

### 4. HumanSL Rank Assessment (`-human-model <model>`)

Optional second pass with a human-style model:

- Annotates player rank (20k–9d) with confidence
- Each rank scored: confidence = softmax(log-probs / temperature)
- Stored in PlayerSummary header of KAB2 format

### 5. Trunk/Pick Output in NNOutput

```cpp
bool includeTrunk = false;    // request trunk features from backend
bool includePick = false;     // request pick features from backend
float* trunk = NULL;          // trunk: trunkNumChannels × nnXLen × nnYLen
float* pick = NULL;           // pick: trunkNumChannels (at move position)
```

Zero performance impact when disabled (default).

### 6. Command Registration

- `int batch_analysis(const std::vector<std::string>& subArgs);` in `MainCmds` namespace
- Registered in `main.cpp` with help text and command handler
- `-batch-size` CLI parameter for NNEvaluator max batch control
- `-profile` hidden flag for per-game timing breakdown

## KAB2 Format

Combined+zlib NPZ: `[4B B payload][B KAB2][4B W payload][W KAB2]`

96-byte header (PlayerSummary) + per-move: scalars(10) + pick(C) + avgTrunk(C) float32.

## Files Changed

| File | Change |
|------|--------|
| `command/batch_analysis.cpp` | New file (~890 lines) |
| `command/batch_analysis.h` | New header |
| `main.cpp` | +9 lines (command registration) |
| `main.h` | +3 lines (function declaration) |
| `neuralnet/nninputs.cpp` | +14 lines (trunk/pick fields) |
| `neuralnet/nninputs.h` | +4 lines (flags and pointers) |
| `neuralnet/nneval.cpp` | +3 lines (trunk channel count) |
| `neuralnet/nneval.h` | +1 line (trunk channels constant) |
| `neuralnet/openclbackend.cpp` | +37 lines (GPU trunk buffer read) |
| `search/search.cpp` | +13 lines / -8 lines (minor) |
| `strmodel/*` | Strength model evaluation modules |

## Upgrade Notes

This fork is rebased directly on upstream `master`. To upgrade to a future upstream release:

1. Fetch upstream: `git fetch upstream`
2. Rebase onto new upstream/master: `git rebase upstream/master`
3. Resolve any conflicts in the files listed above

The customizations are intentionally minimal and isolated to make this process smooth.
