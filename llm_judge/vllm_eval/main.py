from pathlib import Path
import logging

import pandas as pd
from omegaconf import OmegaConf

import load_dataset
import llm_interface
import prompts
import utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


if __name__ == "__main__":
    cli_config = OmegaConf.from_cli()
    if "config_file" not in cli_config:
        cli_config.config_file = str(Path(__file__).parent / "config.yml")

    file_config = OmegaConf.load(cli_config.config_file)
    config = OmegaConf.merge(file_config, cli_config)

    log.info("Config file: %s", cli_config.config_file)
    log.info("Model: %s", config.LLM.model)

    limit = config.Dataset.get("limit")
    items = load_dataset.load_responses_csv(config.Dataset.data_file, limit=limit)

    if not items:
        log.error("No responses found in input file. Aborting.")
        raise SystemExit(1)

    log.info("%d responses loaded", len(items))

    input_stem = Path(config.Dataset.data_file).stem
    outdir = Path(config.output.result_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    judge_cfg = config.get("judge", {})
    if judge_cfg.get("skip", False):
        log.info("Skipping judge (judge.skip=true)")
        raise SystemExit(0)

    log.info("Loading judge model")
    llm = llm_interface.LLMWrapper(config)

    out_path = outdir / f"{input_stem}_judge{config.output.suffix}.csv"
    judge_df = utils.run_judge(
        llm,
        items,
        out_path=out_path,
        enable_thinking=judge_cfg.get("enable_thinking", False),
        enable_thinking_argument=judge_cfg.get("enable_thinking_argument", "enable_thinking"),
        max_items=judge_cfg.get("max_items"),
    )

    log.info("Wrote %d rows to %s", len(judge_df), out_path)
    log.info("Done. Outputs in %s", outdir)
