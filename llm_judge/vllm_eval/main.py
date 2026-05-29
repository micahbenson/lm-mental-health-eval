from collections import defaultdict
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

import load_dataset
import llm_interface
import prompts
import utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _bib_stem(bibfile: Path) -> str:
    bib_name = bibfile.name
    for suffix in (".bib.gz", ".bib.txt", ".bib", ".csv"):
        if bib_name.endswith(suffix):
            return bib_name[: -len(suffix)]
    return bibfile.stem


if __name__ == "__main__":
    cli_config = OmegaConf.from_cli()
    if "config_file" not in cli_config:
        cli_config.config_file = str(Path(__file__).parent / "config.yml")

    file_config = OmegaConf.load(cli_config.config_file)
    config = OmegaConf.merge(file_config, cli_config)

    log.info("Config file: %s", cli_config.config_file)
    log.info("Model: %s", config.LLM.model)
    bibfile = Path(config.Dataset.data_file)
    
    if ".bib" in config.Dataset.data_file:
        venues = [v.strip() for v in config.Dataset.venues.split(",") if v.strip()]
        matcher = load_dataset.build_venue_matcher(venues, config.Dataset.include_findings)
        log.info("Venue filter: %s (findings=%s)", venues, config.Dataset.include_findings)
        papers = load_dataset.parse_bib(bibfile, matcher)
    
    elif ".csv" in config.Dataset.data_file:
        papers = load_dataset.load_csv(config.Dataset.data_file)

    if not papers:
        log.error("No papers with abstracts after filtering. Aborting.")
        raise SystemExit(1)

    # sample_per_year = config.Dataset.get("sample_per_year")
    # if sample_per_year:
    #     rng = np.random.default_rng(0)
    #     by_year = defaultdict(list)
    #     for p in papers:
    #         by_year[p["year"]].append(p)
    #     sampled = []
    #     for ps in by_year.values():
    #         if len(ps) > sample_per_year:
    #             idx = rng.choice(len(ps), sample_per_year, replace=False)
    #             ps = [ps[i] for i in idx]
    #         sampled.extend(ps)
    #     log.info(
    #         "Sampled %d -> %d papers (%d/year max)",
    #         len(papers), len(sampled), sample_per_year,
    #     )
    #     papers = sampled

    print(f"{len(papers)} found")
    # print(papers[0])
    bib_stem = _bib_stem(bibfile)
    outdir = Path(config.output.result_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    judge_cfg = config.get("judge", {})
    if judge_cfg.get("skip", False):
        log.info("Skipping judge (-- judge.skip=true)")
        raise SystemExit(0)

    print("Loading judge")
    llm = llm_interface.LLMWrapper(config)

    per_paper_path = outdir / f"{bib_stem}_per_paper_judge{config.output.suffix}.csv"
    print(f"Using system prompt: {judge_cfg.get("use_system_prompt")}")
    judge_df = utils.run_judge(
        llm,
        papers,
        out_path=per_paper_path,
        binary_prompt=judge_cfg.get("binary_prompt", False),
        enable_thinking=judge_cfg.get("enable_thinking", False),
        enable_thinking_argument=judge_cfg.get("enable_thinking_argument", "enable_thinking"),
        max_papers=judge_cfg.get("max_papers"),
        use_system_prompt=judge_cfg.get("use_system_prompt", False),
        use_dspy_prompts=judge_cfg.get("use_dspy_prompts", False)
    )

    # yearly = (
    #     judge_df.groupby("year")
    #     .agg(
    #         **{f"mean_{k}": (k, "mean") for k, _ in prompts.JUDGE_AXES},
    #         n_papers=("key", "size"),
    #     )
    #     .reset_index()
    # )
    # yearly_path = outdir / f"{bib_stem}_yearly_judge_prompt3.csv"
    # yearly.to_csv(yearly_path, index=False)

    log.info("Wrote %d per-paper rows to %s", len(judge_df), per_paper_path)
    # log.info("Wrote yearly aggregates to %s", yearly_path)
    # print("\n=== Yearly judge axis prevalences ===")
    # print(yearly.to_string(index=False))
    log.info("Done. Outputs in %s", outdir)

