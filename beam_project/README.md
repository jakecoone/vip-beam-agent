# Beam Design and AI Model Comparison

Prepared for Jacob Coone. The original beam_model.py is included unchanged. Its starting design and the SciPy optimization were run successfully; numerical_baseline.json contains those measured results. AI runs are still pending your local API key and Ollama setup. Do not submit until you run the notebook and inspect its outputs. Your username is jcoone and your repository URL is already configured as https://github.com/jakecoone/vip-beam-agent. No changes have been pushed to GitHub.

## What this runs

- SciPy SLSQP numerical optimization, using the supplied black box.
- GPT-4o mini through the OpenAI API: `gpt-4o-mini-2024-07-18` (pinned GPT-4o mini snapshot).
- Llama 3.2 1B through Ollama: `llama3.2:1b`.
- Two independent runs per AI model; each starts at 50 x 80 mm and has exactly 11 proposal opportunities.
- One common prompt structure and strict JSON validation rule. No repaired responses or replacement proposals.
- CSV logging, best-feasible-mass plot, comparison table, PDF report and software/model metadata.

The numerical optimizer must stay separate from the AI agents to meet the worksheet. Each method calls the same unmodified evaluator. The worksheet says "meters" once but its example and parameter names use millimeters. The code uses mm and verifies the example's expected outputs before making AI requests.

## Setup in VS Code on your Mac

1. Extract the ZIP into a project folder. The original `beam_model.py` is already included. Do not replace it with your own equations.
2. Open this folder in VS Code. In its terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

3. Create a file named `.env` in this folder. Add your own key locally:

```text
OPENAI_API_KEY=your_key_here
```

Do not paste your API key into ChatGPT, the notebook, or the script. API requests use your OpenAI API account.

4. Install and start Ollama from https://ollama.com. Then run:

```bash
ollama pull llama3.2:1b
ollama list
ollama --version
```

Keep Ollama running. The default server is `http://localhost:11434`. If needed, set `OLLAMA_HOST` to your server's base URL. Record the Ollama version and the model digest from `ollama list` in your README before submission; the model tag can change over time.

## Run the notebook (recommended for the submission)

Open `jcoone_beam_ai.ipynb`, choose the `.venv` Python kernel, and keep the prefilled USERNAME and REPO_URL values. Run all cells and save the executed notebook as `<your_Purdue_username>_beam_ai.ipynb`. The notebook includes the full experiment implementation and displays its results, logs, plot, and report link. It does not depend on importing the companion script.

The Python script is already named `jcoone_beam_ai.py`. The output CSV and PDF are generated using the username entered in the notebook.

## Or run the script

Your username and repository address are already set:

```bash
python jcoone_beam_ai.py
```

This creates the CSV, PDF, history PNG and summary JSON in `results/`. Running the notebook afterwards runs the experiment again and makes additional API calls. Prefer running only the notebook for the final experiment if you need saved notebook outputs.

## Before submitting

Review the generated PDF and logs. The report derives comparisons from recorded results; it does not invent missing measurements. Add your own observations about specific proposals if useful. Confirm the repository URL is correct and that both AI models actually ran. If a service fails, the CSV records `service_error` with the exception type and the report counts those failures. Inspect your local API/server setup, then rerun if appropriate. Missing evaluator or inconsistent baseline outputs stop the experiment.

Submit exactly these four files to Brightspace:

- `<username>_beam_ai.ipynb` (executed and saved)
- `<username>_beam_ai.py`
- `results/<username>_beam_ai.csv`
- `results/<username>_beam_ai.pdf`

Push the notebook, Python script, CSV, README and requirements to your GitHub repository. Include your results PDF if desired. Do not commit `.env`, API keys or `.venv`. No repository was created or pushed by this package.

## Measurement details

The CSV includes every actual numerical design evaluation and every AI opportunity including the start and invalid replies. For SciPy, identical coordinates requested by the objective and constraints are cached, so they share one black-box call and one row. For AI, repeats are evaluated again, logged and counted. Iteration means black-box call index for SciPy and proposal index for AI.

An AI reply must be a JSON object with exactly `width_mm` and `height_mm`, both finite numbers within bounds. Markdown, extra keys, duplicate keys, numeric strings, booleans and out-of-bounds designs are invalid. Original replies are retained in the CSV. Invalid replies and service failures use a proposal slot without calling the beam evaluator. Evaluator failures are counted as attempted black-box calls, but cannot become feasible results.

Strict feasibility means both constraint margins are nonnegative and dimensions satisfy bounds. The optimizer's separate normalized `1e-6` tolerance is reported, but the best-feasible table and plot use strict feasibility. The starting design remains the incumbent if no proposal improves it. A flat line with service failures is not evidence of successful AI optimization.

Both models receive the same messages and their own evaluation history. Neither receives the SciPy results. Temperature is 0.7, output limit is 160 tokens, and there is no fixed seed. Ollama uses `top_p=1.0` and `num_ctx=8192`. Other sampling defaults are backend-specific. No automatic SDK retry is enabled. Times include requests, evaluation and logging, and may include Ollama loading; figures and report generation are excluded. The summary JSON records installed package versions, platform and evaluator hash. Two runs are too few for strong statistical conclusions.

## References

- Assignment: Individual Project Part 2, Beam Design and AI Model Comparison, Purdue, Fall 2026.
- https://developers.openai.com/api/docs/models/gpt-4o-mini
- https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create
- https://docs.ollama.com/api/chat

## Creating the API file in Mac VS Code

Open this project folder with File > Open Folder. In Explorer, select New File and name it exactly `.env` (including the leading dot, with no `.txt` extension). Place it beside `jcoone_beam_ai.py`. Add `OPENAI_API_KEY=your_actual_key` and save with Command-S. Get your key at https://platform.openai.com/api-keys. The program calls load_dotenv() to load the file; you do not need a VS Code environment-file setting.

After installing requirements, check that the key loads without displaying it:

```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Key loaded:', bool(os.getenv('OPENAI_API_KEY')))"
```

Expect `Key loaded: True`. This checks loading only, not whether the key has API access. Keep `.env` out of GitHub; the included `.gitignore` excludes it.
