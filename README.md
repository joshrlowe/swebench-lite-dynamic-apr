# Dynamic APR Benchmark on SWE-Bench Lite

This repository runs the benchmark-agnostic framework for dynamic cross-model repair of human and LLM-induced bugs described in the paper submitted, and measures pass@k scores and self-bias scores for

You must meet the following requirements to run this on your own machine:

- Python 3.11+
- Docker
- A .env file with an Anthropic API key, OpenAI API key, and an OpenRouter API key

To run this experiment, run the following commands:

```bash
pip install -e .
python3 run.py
```

This runs all three parts of the experiment across five models and prints results to the terminal.

## Files and their Purpose

`run.py` is the entry point for our experiment and runs all three sub-experiments across all models and gives us our results.

`config.py` Reads our environment variables and downloads the SWE-Bench lite cases from Hugging Face.

`models.py` contains all of our LLM API clients from Anthropic, OpenAI, and OpenRouter, and also contains retry logic using a `get_client` factory for API calls that fail.

`pipeline.py` runs our repair loop including bug injections, prompt templates, and patch extraction from LLM responses in the format that's required for SWE-Bench's test harness.

`config.yaml` contains oru 5 models and the parameters we use when we call them.

`swebench_selected_instances.json` contains the IDs for the 20 SWE-bench Lite cases we use in this study.

## Note

A lot of the code for loading the SWE-Bench Lite pipeline and creating our experiments was pulled from the following codebases:

- [SWE-Bench](https://github.com/SWE-bench/SWE-bench)
- [SWE-Bench Lite](https://github.com/swe-bench/SWE-bench/tree/main/swebench/collect/make_lite)
