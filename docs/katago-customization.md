# KataGo fork — Customization Guide

This fork adds one command, `batch_analysis`, to [lightvector/KataGo](https://github.com/lightvector/KataGo).
It is the analysis backend of [KataRank](https://github.com/ahillzhao-msn/KataRank): it turns SGF
games into per-player KAB2 feature files (per-move scalars + trunk features + a per-player
summary with an objective HumanSL rank anchor).

Last upstream merge: `92e62759` (2026-06-22). Updated 2026-10-15.

## Footprint in upstream files

Everything else lives in the new `cpp/command/batch_analysis.{h,cpp}`.

| File | Lines | Change |
|---|---|---|
| `neuralnet/nninputs.h` | +4 | `NNOutput` gains `includeTrunk` / `includePick` flags and `trunk` / `pick` buffers (off by default) |
| `neuralnet/nninputs.cpp` | +14 | Free `trunk` / `pick` in the destructor and in `operator=` (they are never copied) |
| `neuralnet/nneval.h` | +2 | `NNResultBuf::includeTrunk`; `NNEvaluator::getModelTrunkNumChannels()` |
| `neuralnet/nneval.cpp` | +5 | Carry `includeTrunk` from the request into the output; trunk channel getter |
| `neuralnet/openclbackend.cpp` | +46 | Read the trunk buffer back from the GPU when any row asks for it; FP16 → FP32; strip the `paddedNNXYLen` padding |
| `main.h` / `main.cpp` | +3 / +9 | Register the `batch_analysis` command |
| `CMakeLists.txt` | +1 | Add `command/batch_analysis.cpp` |

Only the OpenCL backend reads trunk features back. CUDA / Eigen builds still run `batch_analysis`,
but only with `-no-trunk`.

## `batch_analysis`

```
katago batch_analysis -model <model.bin.gz> [-config <cfg>]
  [-list <games.csv> | -sgf-dir <dir/>] [-output-dir <out/>]
  [-visits N] [-min-moves N] [-max-games N]
  [-stream] [-no-trunk] [-daemon]
  [-batch-size N (default 64)] [-threads N (default 16)]
  [-human-model <human.bin.gz>]
```

- **File mode** (default): one `<sgf-stem>.npz` per game,
  `[4B B_size][B KAB2][4B W_size][W KAB2]`, zlib-compressed, plus `_meta.csv`.
- **Stream mode** (`-stream`): uncompressed frames on stdout,
  `['B'/'W'][uint32 idLen][game id][uint32 size][KAB2]` per player, ending with `0x00`.
- **Daemon mode** (`-daemon`, implies `-stream`): models stay loaded; each stdin line is the path
  of a `games.csv` job; each job's frames end with `0x01`; `reset` clears caches and counters;
  `quit` exits. This is how KataRank runs it.
- `-no-trunk`: scalars only (10 floats per move), much smaller and faster.
- `-threads`: positions are evaluated concurrently and folded back in move order, so the output
  is identical to a serial run.

The NN cache is disabled (`nnCacheSizePowerOfTwo = -1`): every position is evaluated exactly
once, and with trunk outputs the cache used to grow without bound.

## KAB2 payload

```
NPZHeader (96 bytes):
  magic "KAB2" | numMoves | scalarDim (=10) | trunkDim | pickDim | nnXLen | nnYLen | flags
  PlayerSummary (16 floats, 64 bytes)
per move: scalars[10] + pick[trunkDim] + avgTrunk[trunkDim]      (no vectors with -no-trunk)
```

Scalars (white's perspective): `whiteWin`, `whiteLoss`, `whiteNoResult`, `whiteScoreMean/50`,
`shorttermScoreError/10`, `policyPrior`, `policyRank/361`, `isWhite`, `winDelta`, `scoreDelta/50`.

`PlayerSummary`: accuracy top-1 / top-3, mean log prior, mean win rate, mean score lead, mean
complexity, score variance, approximate score drop, mean win / score delta, then the HumanSL
fields below.

## HumanSL rank anchor (v2, `-human-model`)

A second pass scores each game with the HumanSL model:

1. **Gate**: the main model's point loss is computed for every move; for each player only the moves
   inside that player's 10–90% loss quantile band vote.
2. **Profiles**: the voting moves are scored at 5 fixed rank profiles: 20k, 13k, 6k, 2d, 9d.
3. **Posterior**: a natural cubic spline through the 5 log-likelihood totals gives a posterior over
   all 29 ranks (20k … 9d).

Written to `PlayerSummary`: `humanRankIdx` (rounded posterior mean, `-1` if not run),
`humanRankExp` (posterior mean), `humanRankSd`, `humanProfiles` (5), `humanAnchorVer` (2; 0 = absent),
and `humanLogPrior`. `_meta.csv` has the matching columns.

Measured on GOPredict games: split-half reliability r = 0.775 (full 29-profile scan: 0.78) at about
14% of the full scan's compute.

## Reproducibility

OpenCL results depend on the tuning file under `KataGoData/opencltuning/`; with the same tuning file
the output is bit-identical across rebuilds.

## Upgrading to a newer KataGo

1. Merge upstream; conflicts can only come from the files in the table above.
2. Rebuild (`cpp/build_katago.cmd` on Windows), then compare a small batch's output against the
   previous binary using the same tuning file (it must be bit-identical unless the network changed).
