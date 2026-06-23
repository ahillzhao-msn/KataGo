# KataGo — Trunk Feature Extraction & Batch Analysis Fork

This fork extends **[lightvector/KataGo](https://github.com/lightvector/KataGo)** with trunk feature extraction, batch SGF analysis, daemon/stream modes, and HumanSL rank assessment for Go strength evaluation models.

**Based on:** upstream `master` (v1.16.5+) | **Backends:** OpenCL + CUDA + TensorRT | **Format:** KAB2 (zlib-compressed)

> **Upstream README** → See [lightvector/KataGo](https://github.com/lightvector/KataGo) for the original project documentation, including installation, GUI setup, GTP usage, training, and full feature list.

---

## Custom Features

| Component | Description |
|-----------|-------------|
| **`batch_analysis` command** | Batch SGF analysis: extract head(12-dim) + trunk(256-dim) + pick(256-dim) features from neural net. KAB2 binary output with zlib compression. |
| **Stream mode** | Pipe KAB2 frames to stdout — zero disk I/O, suitable for pipeline processing. |
| **Daemon mode** | Model-resident persistent process: task latency drops from ~10s+ to analysis-only time. stdin interface: one games.csv path per line; `reset` clears cache; `quit` exits. |
| **Lite mode** | Scalars-only (10 floats/move), 100x smaller output — ideal for rank inference pipelines. |
| **HumanSL rank assessment** | Optional second pass with a human-style model to annotate player rank (20k–9d) and confidence. |
| **Trunk/Pick output** | `includeTrunk`/`includePick` flags in NNOutput; OpenCL backend reads trunk buffer from GPU. Zero performance impact when disabled. |
| **SGF naming** | Output files named after original SGF filename (not hex IDs). Meta CSV includes `sgf_path` column. |

## Quick Start

```bash
# First-time GPU tuning (required once)
katago tuner -config analysis_config.cfg -model model.bin.gz

# Batch analysis (default: combined+zlib KAB2 output)
katago batch_analysis \
  -config analysis_config.cfg \
  -model model.bin.gz \
  -list games.csv \
  -output-dir ./features/

# Stream mode (pipe to stdout, zero disk I/O)
katago batch_analysis \
  -model model.bin.gz \
  -list games.csv \
  -stream

# Lite mode (scalars only, no trunk/pick)
katago batch_analysis \
  -model model.bin.gz \
  -list games.csv \
  -stream -no-trunk

# With HumanSL rank annotation
katago batch_analysis \
  -model model.bin.gz \
  -list games.csv \
  -stream -no-trunk \
  -human-model humansl.bin.gz

# Daemon mode (persistent process)
katago batch_analysis -model model.bin.gz -daemon
```

## KAB2 Output Format

Each `.npz` file contains combined+compressed per-player data:

```
[4B Black payload size][Black KAB2 data][4B White payload size][White KAB2 data]
```

KAB2 payload: 96-byte header (with PlayerSummary) + per-move scalars(10) + pick(C) + avgTrunk(C) float32.

Output naming: `<sgf_stem>_B.npz` / `<sgf_stem>_W.npz`

See [docs/katago-customization.md](docs/katago-customization.md) for in-depth upgrade strategy and internals.

## Design & Research

- **[STRENGTH_MODEL_RESEARCH.md](./STRENGTH_MODEL_RESEARCH.md)** — comprehensive technical analysis: trunk features, HumanSL integration, representation theory, scoring system design, and KAB2 format specification.
- **[docs/katago-customization.md](./docs/katago-customization.md)** — detailed code modifications and upgrade strategy.

## Credits

- Original KataGo: [lightvector/KataGo](https://github.com/lightvector/KataGo)
- Strength model concept: [Animiral/go-strength-model](https://github.com/Animiral/go-strength-model)
- Preprocessing patterns: [ahillzhao-msn/go-analyzer](https://github.com/ahillzhao-msn/go-analyzer)
- This fork: [ahillzhao-msn/KataGo](https://github.com/ahillzhao-msn/KataGo)

## For Original KataGo Usage

For basic KataGo usage (GTP, analysis engine, GUIs, tuning, compiling from source, backend comparison, troubleshooting), please refer to:

- **Full upstream README**: [github.com/lightvector/KataGo](https://github.com/lightvector/KataGo)
- **Pre-compiled binaries**: [Releases page](https://github.com/lightvector/KataGo/releases)
- **Neural nets**: [katagotraining.org](https://katagotraining.org/)
- **Docs**: [Compiling](Compiling.md) | [GTP Extensions](docs/GTP_Extensions.md) | [Analysis Engine](docs/Analysis_Engine.md) | [Selfplay Training](SelfplayTraining.md)
- **Research**: [KataGoMethods.md](docs/KataGoMethods.md) | [GraphSearch.md](docs/GraphSearch.md) | [TrainingHistory.md](TrainingHistory.md)

## License

Except for included external libraries under `cpp/external/` and `cpp/core/sha2.cpp` (each with their own licenses), all code in this repo is released under the [LICENSE](LICENSE) file (Apache 2.0 / MIT).
