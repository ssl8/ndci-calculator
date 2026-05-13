# NDCI Calculator — Normalised Diagnostic Contribution Index

A reference implementation of the Normalised Diagnostic Contribution Index
(NDCI), formalised in Suslu, Ali, Jennions (2025) and applied across
multiple subsystems in Suslu, Ali, Jennions (2026).

NDCI scores how much *diagnostic information* a given sensor contributes
to a fault-detection dictionary, normalised against the contribution of
the most informative sensor in the candidate pool. It's the inner
fitness-evaluation primitive that MOSOF uses to weight diagnostic
performance.

## What NDCI does

Given:

- a binary **sensor × fault detectability matrix** `D` where `D[s, f] = 1`
  if sensor `s` can detect fault `f`, else `0`,
- optional **fault prior probabilities** `p_f` (uniform if omitted),

NDCI computes, for each sensor:

```
NDCI(s) = I(s) / max_t I(t)
```

where `I(s)` is the information-theoretic contribution of sensor `s` to
disambiguating the fault set — operationalised here as the entropy
reduction in the fault posterior when `s` is observed. The max-normalisation
ensures NDCI lives in `[0, 1]` and is directly comparable across
sensor pools of different sizes.

## Run the demo

```bash
pip install -r requirements.txt
python ndci_demo.py
```

The demo:

1. Loads the published ECS detectability matrix from
   `../../data/ndci-matrix.json` (the dataset that drives the website's
   thesis figures).
2. Computes per-sensor NDCI.
3. Re-ranks sensors by NDCI under three scenarios — uniform priors,
   actuator-fault-weighted priors, and leak-fault-weighted priors —
   to illustrate the *stakeholder-aware* dimension of the method.
4. Prints a ranked table and writes `ndci_per_sensor.csv`.

Expected runtime: under one second.

## File layout

```
.
├── README.md
├── requirements.txt   # numpy
└── ndci_demo.py       # ~120 lines, single file
```

## Citation

```bibtex
@article{Suslu2025NDCI,
  author  = {Suslu, Burak and Ali, Fakhre and Jennions, Ian K.},
  title   = {NDCI Integration to Multi-Objective Sensor Optimisation Framework — An ECS Case},
  journal = {Sensors},
  volume  = {25},
  number  = {9},
  pages   = {2661},
  year    = {2025},
  doi     = {10.3390/s25092661}
}
```

## License

MIT.
