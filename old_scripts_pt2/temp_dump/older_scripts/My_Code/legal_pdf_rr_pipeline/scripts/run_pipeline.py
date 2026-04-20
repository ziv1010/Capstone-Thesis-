#!/usr/bin/env python3
"""
run_pipeline.py
Orchestrator: runs all five pipeline stages sequentially.
"""

import argparse
import importlib.util
import logging
import os
import shutil
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger("pipeline")


def _load_script(name: str, filename: str):
    """Load a sibling script as a module."""
    path = os.path.join(SCRIPT_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def setup_logging(log_dir: str) -> str:
    """Set up file + console logging; return the log-file path."""
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"pipeline_{ts}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_path


def reset_dir_contents(path: str) -> None:
    """Remove previous stage outputs so runs do not mix documents."""
    os.makedirs(path, exist_ok=True)
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        else:
            os.unlink(full_path)


def run_inference_stage(
    model_name: str,
    inference_input_path: str,
    predictions_dir: str,
    output_root: str,
    device: str | None,
    embedder: str,
    checkpoint_path: str | None,
    hier_checkpoint_path: str | None,
    hier_assets_dir: str | None,
) -> None:
    if model_name == "toinlegalbert":
        s04 = _load_script("s04_toinlegalbert", "04_infer_toinlegalbert.py")
        models_dir = os.path.join(output_root, "models", "LegalSeg_ToInLegalBERT")
        s04.run(
            inference_input_path,
            predictions_dir,
            models_dir,
            device=device,
            embedder=embedder,
            checkpoint_path=checkpoint_path,
        )
        return

    if model_name == "hier_bilstm_crf":
        s04 = _load_script("s04_hier_bilstm_crf", "04_infer_hier_bilstm_crf.py")
        models_dir = os.path.join(output_root, "models", "LegalSeg_Hier_BiLSTM_CRF")
        s04.run(
            inference_input_path,
            predictions_dir,
            models_dir,
            device=device,
            checkpoint_path=hier_checkpoint_path,
            assets_dir=hier_assets_dir,
        )
        return

    raise ValueError(f"Unsupported inference model: {model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete Legal PDF → Rhetorical Role pipeline."
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Folder containing input PDF files.",
    )
    parser.add_argument(
        "--output_root", required=True,
        help="Root output directory (should be the legal_pdf_rr_pipeline dir).",
    )
    parser.add_argument(
        "--device", default=None,
        help="PyTorch device (cuda:0 / cpu). Auto-detect if omitted.",
    )
    parser.add_argument(
        "--embedder", default="absolute", choices=("absolute", "relative"),
        help="ToInLegalBERT positional embedder. The released checkpoint does not serialize this choice.",
    )
    parser.add_argument(
        "--checkpoint_path", default=None,
        help="Optional path to a complete ToInLegalBERT checkpoint directory or zip.",
    )
    parser.add_argument(
        "--hier_checkpoint_path", default=None,
        help="Optional path to a Hier_BiLSTM-CRF checkpoint file. If omitted, the pipeline uses the local OpenNyAI-fine-tuned checkpoint when present, otherwise the released LegalSeg checkpoint.",
    )
    parser.add_argument(
        "--hier_assets_dir", default=None,
        help="Optional directory containing word2idx.json and tag2idx.json for the selected Hier_BiLSTM-CRF checkpoint.",
    )
    parser.add_argument(
        "--model", default="toinlegalbert",
        choices=("toinlegalbert", "hier_bilstm_crf", "auto"),
        help="Inference backend. 'auto' tries ToInLegalBERT first and falls back to Hier_BiLSTM-CRF.",
    )
    args = parser.parse_args()

    # Resolve paths
    output_root = os.path.abspath(args.output_root)
    input_dir = os.path.abspath(args.input_dir)

    extracted_text_dir = os.path.join(output_root, "extracted_text")
    sentence_json_dir = os.path.join(output_root, "sentence_json")
    predictions_dir = os.path.join(output_root, "predictions")
    structured_dir = os.path.join(output_root, "structured_outputs")
    log_dir = os.path.join(output_root, "logs")

    log_path = setup_logging(log_dir)
    logger.info("Pipeline started. Log file: %s", log_path)
    logger.info("Input dir : %s", input_dir)
    logger.info("Output root: %s", output_root)

    reset_dir_contents(extracted_text_dir)
    reset_dir_contents(sentence_json_dir)
    reset_dir_contents(predictions_dir)
    reset_dir_contents(structured_dir)

    t0 = time.time()

    try:
        # ------------------------------------------------------------------
        # Stage 1: Extract text from PDFs
        # ------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("STAGE 1: PDF Text Extraction")
        logger.info("=" * 60)
        s01 = _load_script("s01", "01_extract_pdf_text.py")
        s01.run(input_dir, extracted_text_dir)

        # ------------------------------------------------------------------
        # Stage 2: Sentence segmentation
        # ------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("STAGE 2: Sentence Segmentation")
        logger.info("=" * 60)
        s02 = _load_script("s02", "02_sentence_segment.py")
        s02.run(extracted_text_dir, sentence_json_dir)

        # ------------------------------------------------------------------
        # Stage 3: Prepare inference input
        # ------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("STAGE 3: Prepare Inference Input")
        logger.info("=" * 60)
        s03 = _load_script("s03", "03_prepare_toinlegalbert_input.py")
        inference_input_path = os.path.join(predictions_dir, "inference_input.json")
        s03.run(sentence_json_dir, inference_input_path)

        # ------------------------------------------------------------------
        # Stage 4: Model inference
        # ------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("STAGE 4: Model Inference")
        logger.info("=" * 60)
        if args.model == "auto":
            logger.info("Attempting ToInLegalBERT first; fallback is Hier_BiLSTM-CRF")
            try:
                run_inference_stage(
                    model_name="toinlegalbert",
                    inference_input_path=inference_input_path,
                    predictions_dir=predictions_dir,
                    output_root=output_root,
                    device=args.device,
                    embedder=args.embedder,
                    checkpoint_path=args.checkpoint_path,
                    hier_checkpoint_path=args.hier_checkpoint_path,
                    hier_assets_dir=args.hier_assets_dir,
                )
            except Exception as exc:
                logger.warning("ToInLegalBERT failed: %s", exc)
                logger.info("Falling back to Hier_BiLSTM-CRF")
                run_inference_stage(
                    model_name="hier_bilstm_crf",
                    inference_input_path=inference_input_path,
                    predictions_dir=predictions_dir,
                    output_root=output_root,
                    device=args.device,
                    embedder=args.embedder,
                    checkpoint_path=args.checkpoint_path,
                    hier_checkpoint_path=args.hier_checkpoint_path,
                    hier_assets_dir=args.hier_assets_dir,
                )
        else:
            display_name = "ToInLegalBERT" if args.model == "toinlegalbert" else "Hier_BiLSTM-CRF"
            logger.info("Selected inference backend: %s", display_name)
            run_inference_stage(
                model_name=args.model,
                inference_input_path=inference_input_path,
                predictions_dir=predictions_dir,
                output_root=output_root,
                device=args.device,
                embedder=args.embedder,
                checkpoint_path=args.checkpoint_path,
                hier_checkpoint_path=args.hier_checkpoint_path,
                hier_assets_dir=args.hier_assets_dir,
            )

        # ------------------------------------------------------------------
        # Stage 5: Group labels into structured output
        # ------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("STAGE 5: Group Labels → Structured Output")
        logger.info("=" * 60)
        s05 = _load_script("s05", "05_group_labels.py")
        s05.run(predictions_dir, structured_dir)
    except Exception as exc:
        elapsed = time.time() - t0
        logger.error("Pipeline failed after %.1f seconds: %s", elapsed, exc)
        raise SystemExit(1)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Pipeline finished in %.1f seconds.", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
