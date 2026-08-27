# ECO 629 — code

Runnable code for ECO 629 *Studies in Quantitative Methods*, Stony Brook University,
Fall 2026. Taught by Fedor Iskhakov.

**Lecture notes: <https://dse.iskh.me>** — the notes are the place to start. They
carry the theory, the derivations and the worked examples, with each chapter also
downloadable as PDF. This repository holds the same models as standalone code you
can clone, run and modify.

Solution methods and structural estimation of dynamic models: single-agent discrete
and continuous choice, micro-founded equilibrium models, and dynamic games —
NFXP, MPEC, CCP, NPL, EPL and MSM.

## Running it

Plain Python — `numpy`, `scipy`, `matplotlib`, `sympy`, and JupyterLab for the
notebooks. Nothing else.

```bash
git clone https://github.com/fediskhakov/sb-dse-code.git
cd sb-dse-code
```

**uv** (the course default):

```bash
uv venv                                 # creates .venv with a suitable Python
uv pip install -r requirements.txt
uv run jupyter lab                      # no activation needed
```

**conda** ([Miniforge](https://conda-forge.org/download/)):

```bash
conda env create -f environment.yml
conda activate eco629
jupyter lab
```

These are the two supported ways, and they match the setup described in the lecture
notes. If you already built the `eco629` environment for the notes, it covers
everything here — just activate it.

Each model lives in its own directory with a short README saying what it does and
how to run it.

## Using it

Take it, run it, break it, rewrite it. The code is written to be read: it favors
clarity over speed, and follows the notation of the lecture notes rather than of any
particular paper. Where a model can be solved several ways — value function
iteration, policy iteration, Newton–Kantorovich, EGM — the implementations are kept
side by side so they can be compared on the same problem.

If something does not run, or a result looks wrong, please open an issue or a pull
request.

## License

MIT — see [LICENSE](LICENSE). The lecture notes themselves are licensed separately
under CC BY-NC-SA 4.0.
