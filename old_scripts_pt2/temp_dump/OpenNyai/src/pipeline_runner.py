"""Core OpenNyAI pipeline orchestration."""

from __future__ import annotations

import email
import inspect
import logging
import multiprocessing
import os
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from tqdm import tqdm

from .config import PipelineConfig, OutputLayout, document_output_paths
from .io_utils import DocumentInput, compact_tree_report, configure_logging, discover_text_files, read_text_file, write_json, write_text
from .output_formatter import (
    build_combined_output,
    build_ner_output,
    build_rhetorical_roles_output,
    build_sentence_annotations_output,
    build_summary_output,
)
from .validators import normalize_summary_length, validate_gpu_request, validate_pipeline_config, warn_if_short_text


MODEL_WHEEL_FALLBACK_URLS = {
    "en_legal_ner_trf": "https://huggingface.co/opennyaiorg/en_legal_ner_trf/resolve/main/en_legal_ner_trf-any-py3-none-any.whl",
}


def patch_opennyai_summarizer_device_mismatch(logger: logging.Logger) -> None:
    """Patch an upstream OpenNyAI summarizer bug triggered by newer PyTorch builds."""
    import torch
    from opennyai.summarizer.models.model_builder import ExtSummarizer

    if getattr(ExtSummarizer.forward, "_opennyai_device_patch", False):
        return

    def _patched_forward(self, src, segs, clss, mask_src, mask_cls, sentence_rhetorical_roles):
        top_vec = self.bert(src, segs, mask_src)
        batch_indices = torch.arange(top_vec.size(0), device=top_vec.device).unsqueeze(1)
        clss = clss.to(top_vec.device)
        mask_cls = mask_cls.to(top_vec.device)
        sentence_rhetorical_roles = sentence_rhetorical_roles.to(top_vec.device)
        sents_vec = top_vec[batch_indices, clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        if self.args.use_rhetorical_roles:
            sent_one_hot = torch.nn.functional.one_hot(sentence_rhetorical_roles, num_classes=16).float()
            sent_scores_concatenated = torch.cat((sents_vec, sent_one_hot), dim=2)
            sent_scores = self.ext_layer(sent_scores_concatenated, mask_cls).squeeze(-1)
        else:
            sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls

    _patched_forward._opennyai_device_patch = True
    ExtSummarizer.forward = _patched_forward
    logger.info("Patched OpenNyAI summarizer device mismatch for current PyTorch runtime.")


def import_opennyai_api() -> Tuple[Any, Any]:
    """Import Data and Pipeline classes defensively across versions."""
    import opennyai

    try:
        from opennyai.utils.preprocess import Data
    except Exception:
        Data = getattr(opennyai, "Data", None)

    try:
        from opennyai import Pipeline
    except Exception:
        from opennyai.pipeline import Pipeline

    if Data is None or Pipeline is None:
        raise ImportError("Unable to import OpenNyAI Data/Pipeline classes from the installed package.")
    return Data, Pipeline


def filter_supported_kwargs(target: Any, kwargs: Dict[str, Any], logger: logging.Logger, label: str) -> Dict[str, Any]:
    """Drop unsupported kwargs instead of crashing on minor API changes."""
    signature = inspect.signature(target)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return kwargs

    supported_names = set(signature.parameters)
    supported_kwargs = {key: value for key, value in kwargs.items() if key in supported_names}
    unsupported_names = sorted(set(kwargs) - supported_names)
    if unsupported_names:
        logger.warning("Ignoring unsupported %s kwargs for this OpenNyAI version: %s", label, unsupported_names)
    return supported_kwargs


def _download_file(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response:
        destination.write_bytes(response.read())


def _repair_wheel_filename(downloaded_wheel: Path) -> Path:
    with zipfile.ZipFile(downloaded_wheel) as wheel_file:
        metadata_path = next(name for name in wheel_file.namelist() if name.endswith(".dist-info/METADATA"))
        wheel_path = next(name for name in wheel_file.namelist() if name.endswith(".dist-info/WHEEL"))

        metadata_message = email.message_from_bytes(wheel_file.read(metadata_path))
        wheel_message = email.message_from_bytes(wheel_file.read(wheel_path))

    distribution_name = metadata_message.get("Name", downloaded_wheel.stem).replace("-", "_")
    version = metadata_message.get("Version", "0.0.0")
    tag = wheel_message.get_all("Tag", ["py3-none-any"])[0]

    repaired_wheel = downloaded_wheel.with_name(f"{distribution_name}-{version}-{tag}.whl")
    if repaired_wheel != downloaded_wheel:
        downloaded_wheel.replace(repaired_wheel)
    return repaired_wheel


def _install_model_wheel_with_filename_repair(model_name: str, model_url: str, logger: logging.Logger) -> None:
    logger.info("Downloading %s model package from %s", model_name, model_url)
    with tempfile.TemporaryDirectory() as temporary_directory:
        parsed_url = urllib.parse.urlparse(model_url)
        download_name = Path(parsed_url.path).name or f"{model_name}.whl"
        download_path = Path(temporary_directory) / download_name
        _download_file(model_url, download_path)
        repaired_wheel = _repair_wheel_filename(download_path)
        logger.info("Installing repaired model wheel %s", repaired_wheel.name)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", str(repaired_wheel)],
            check=True,
        )


def ensure_ner_model_available(model_name: str, logger: logging.Logger) -> None:
    """Install the requested NER model package when it is not already available."""
    import spacy

    if model_name in spacy.util.get_installed_models():
        return

    logger.info("Missing spaCy model package %s; attempting installation.", model_name)

    model_url = MODEL_WHEEL_FALLBACK_URLS.get(model_name)
    try:
        from opennyai.utils.download import PIP_INSTALLER_URLS, install

        model_url = PIP_INSTALLER_URLS.get(model_name, model_url)
        if model_url is None:
            raise RuntimeError(f"No download URL found for NER model {model_name}.")
        try:
            install(model_url)
        except Exception as exc:
            logger.warning(
                "OpenNyAI's default installer failed for %s (%s). Falling back to wheel filename repair.",
                model_name,
                exc,
            )
            _install_model_wheel_with_filename_repair(model_name, model_url, logger)
    except Exception:
        if model_url is None:
            raise
        _install_model_wheel_with_filename_repair(model_name, model_url, logger)

    if model_name not in spacy.util.get_installed_models():
        raise RuntimeError(f"Model package {model_name} is still unavailable after installation attempts.")


def _build_data_object(Data: Any, documents: List[DocumentInput], config: PipelineConfig, logger: logging.Logger) -> Any:
    data_kwargs = {
        "input_text": [document.text for document in documents],
        "preprocessing_nlp_model": config.preprocessing_model,
        "use_cache": True,
        "verbose": config.verbose,
        "use_gpu": config.use_gpu,
        "file_ids": [document.internal_id for document in documents],
    }
    filtered_kwargs = filter_supported_kwargs(Data, data_kwargs, logger, "Data")
    return Data(**filtered_kwargs)


def _build_pipeline_object(Pipeline: Any, config: PipelineConfig, logger: logging.Logger) -> Any:
    pipeline_kwargs = {
        "components": ["NER", "Rhetorical_Role", "Summarizer"],
        "use_gpu": config.use_gpu,
        "verbose": config.verbose,
        "ner_model_name": config.ner_model_name,
        "ner_do_sentence_level": config.ner_do_sentence_level,
        "ner_do_postprocess": config.ner_do_postprocess,
        "ner_mini_batch_size": config.batch_size,
        "summarizer_summary_length": config.summary_length,
    }
    filtered_kwargs = filter_supported_kwargs(Pipeline, pipeline_kwargs, logger, "Pipeline")
    return Pipeline(**filtered_kwargs)


def discover_input_files(config: PipelineConfig, logger: logging.Logger) -> List[Path]:
    """Discover input text files once in the parent process."""
    discovered_files = discover_text_files(config.input_dir, config.glob_pattern)
    logger.info(
        "Discovered %s text file(s) in %s with pattern %r",
        len(discovered_files),
        config.input_dir,
        config.glob_pattern,
    )
    return discovered_files


def validate_unique_file_ids(paths: Sequence[Path]) -> None:
    """Fail early when multiple input paths would map to the same output file_id."""
    seen_file_ids = set()
    for path in paths:
        file_id = path.stem
        if file_id in seen_file_ids:
            raise RuntimeError(f"Duplicate file_id collision detected for {file_id!r}.")
        seen_file_ids.add(file_id)


def prepare_documents(
    paths: Sequence[Path],
    config: PipelineConfig,
    layout: OutputLayout,
    logger: logging.Logger,
) -> Tuple[List[DocumentInput], List[Dict[str, str]]]:
    """Read input text files and assign worker-local internal identifiers."""
    documents: List[DocumentInput] = []
    skipped_records: List[Dict[str, str]] = []

    for path in paths:
        file_id = path.stem
        output_paths = document_output_paths(layout, file_id)
        if not config.overwrite and output_paths["combined"].exists():
            logger.info("Skipping %s because outputs already exist and --overwrite was not set.", file_id)
            skipped_records.append({"file_id": file_id, "reason": "existing_output", "source_path": str(path)})
            continue

        text, encoding_name = read_text_file(path)
        if encoding_name != "utf-8":
            logger.warning("Read %s using fallback decoder %s", path.name, encoding_name)
        if not text.strip():
            logger.warning("Skipping empty file %s", path)
            skipped_records.append({"file_id": file_id, "reason": "empty_file", "source_path": str(path)})
            continue

        warn_if_short_text(file_id, text, logger)
        documents.append(
            DocumentInput(
                source_path=path,
                file_id=file_id,
                internal_id=f"doc{len(documents) + 1:06d}",
                text=text,
            )
        )

    return documents, skipped_records


def _normalize_pipeline_results(pipeline_result: Any) -> Dict[str, Dict[str, Any]]:
    raw_docs = pipeline_result if isinstance(pipeline_result, list) else [pipeline_result]
    result_by_internal_id: Dict[str, Dict[str, Any]] = {}
    for raw_doc in raw_docs:
        if not isinstance(raw_doc, dict):
            raise TypeError(f"Unexpected OpenNyAI pipeline result type: {type(raw_doc)!r}")
        internal_id = str(raw_doc.get("id", "")).strip()
        if not internal_id:
            raise RuntimeError("OpenNyAI pipeline result is missing the internal document id.")
        if internal_id in result_by_internal_id:
            raise RuntimeError(f"Duplicate internal result id {internal_id!r} returned from OpenNyAI.")
        result_by_internal_id[internal_id] = raw_doc
    return result_by_internal_id


def _persist_document_outputs(document: DocumentInput, raw_doc: Dict[str, Any], layout: OutputLayout, logger: logging.Logger) -> None:
    output_paths = document_output_paths(layout, document.file_id)
    combined_output = build_combined_output(
        file_id=document.file_id,
        internal_id=document.internal_id,
        source_path=str(document.source_path),
        raw_doc=raw_doc,
    )
    annotations_output = build_sentence_annotations_output(document.file_id, raw_doc)
    ner_output = build_ner_output(document.file_id, raw_doc)
    rhetorical_roles_output = build_rhetorical_roles_output(document.file_id, raw_doc)
    summary_output, summary_text = build_summary_output(document.file_id, raw_doc)

    write_json(output_paths["combined"], combined_output)
    write_json(output_paths["annotations"], annotations_output)
    write_json(output_paths["ner"], ner_output)
    write_json(output_paths["rhetorical_roles"], rhetorical_roles_output)
    write_json(output_paths["summary_json"], summary_output)
    write_text(output_paths["summary_text"], summary_text)

    logger.info(
        "Saved outputs for %s to %s, %s, %s, %s, %s and %s",
        document.file_id,
        output_paths["combined"],
        output_paths["annotations"],
        output_paths["ner"],
        output_paths["rhetorical_roles"],
        output_paths["summary_json"],
        output_paths["summary_text"],
    )


def _persist_pipeline_results(
    documents: Sequence[DocumentInput],
    pipeline_result: Any,
    layout: OutputLayout,
    logger: logging.Logger,
) -> List[str]:
    result_by_internal_id = _normalize_pipeline_results(pipeline_result)
    expected_ids = {document.internal_id for document in documents}
    received_ids = set(result_by_internal_id)
    if expected_ids != received_ids:
        raise RuntimeError(
            "OpenNyAI batch result ids do not match the requested documents. "
            f"expected={sorted(expected_ids)!r}, received={sorted(received_ids)!r}"
        )

    succeeded_file_ids = []
    for document in documents:
        _persist_document_outputs(document, result_by_internal_id[document.internal_id], layout, logger)
        succeeded_file_ids.append(document.file_id)
    return succeeded_file_ids


def _run_pipeline_for_documents(
    *,
    documents: Sequence[DocumentInput],
    config: PipelineConfig,
    layout: OutputLayout,
    logger: logging.Logger,
    progress_desc: str,
) -> Dict[str, Any]:
    failed_records: List[Dict[str, str]] = []
    succeeded_file_ids: List[str] = []

    if not documents:
        logger.warning("No documents are available for inference after filtering and skips.")
        return {"failed_records": failed_records, "succeeded_file_ids": succeeded_file_ids}

    Data, Pipeline = import_opennyai_api()
    patch_opennyai_summarizer_device_mismatch(logger)

    logger.info("Initialising OpenNyAI Data and Pipeline objects.")
    data = _build_data_object(Data, list(documents), config, logger)
    pipeline = _build_pipeline_object(Pipeline, config, logger)
    logger.info(
        "Inference started for %s document(s) with pipeline_batch_size=%s.",
        len(documents),
        config.pipeline_batch_size,
    )

    progress = tqdm(total=len(documents), desc=progress_desc, disable=not config.verbose)
    try:
        for batch_start in range(0, len(documents), config.pipeline_batch_size):
            batch_documents = list(documents[batch_start : batch_start + config.pipeline_batch_size])
            logger.info(
                "Starting inference for batch %s-%s (%s document(s)).",
                batch_start + 1,
                batch_start + len(batch_documents),
                len(batch_documents),
            )
            try:
                processed_batch = [data[index] for index in range(batch_start, batch_start + len(batch_documents))]
                pipeline_result = pipeline(processed_batch)
                succeeded_file_ids.extend(_persist_pipeline_results(batch_documents, pipeline_result, layout, logger))
                progress.update(len(batch_documents))
                continue
            except Exception as exc:  # pragma: no cover - exercised during runtime
                logger.exception(
                    "Batch %s-%s failed (%s). Retrying documents individually.",
                    batch_start + 1,
                    batch_start + len(batch_documents),
                    exc,
                )

            for document_offset, document in enumerate(batch_documents):
                document_index = batch_start + document_offset
                logger.info("Starting inference for %s", document.file_id)
                try:
                    processed_document = data[document_index]
                    pipeline_result = pipeline([processed_document])
                    succeeded_file_ids.extend(_persist_pipeline_results([document], pipeline_result, layout, logger))
                except Exception as exc:  # pragma: no cover - exercised during runtime
                    failed_records.append(
                        {
                            "file_id": document.file_id,
                            "source_path": str(document.source_path),
                            "error": str(exc),
                        }
                    )
                    logger.exception("Failed processing %s", document.file_id)
                finally:
                    progress.update(1)
    finally:
        progress.close()

    logger.info("Inference finished.")
    return {"failed_records": failed_records, "succeeded_file_ids": succeeded_file_ids}


def _parse_gpu_devices(raw_gpu_devices: str) -> Tuple[int, ...]:
    if not raw_gpu_devices.strip():
        return ()

    devices = []
    for token in raw_gpu_devices.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            devices.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid GPU device id {token!r} in --gpu_devices.") from exc

    if len(set(devices)) != len(devices):
        raise ValueError(f"Duplicate GPU device ids are not allowed: {devices!r}")
    return tuple(devices)


def _resolve_worker_gpu_devices(config: PipelineConfig, logger: logging.Logger) -> Tuple[int, ...]:
    requested_devices = _parse_gpu_devices(config.gpu_devices)
    if not config.use_gpu:
        if requested_devices:
            logger.warning("Ignoring --gpu_devices because --use_gpu was not requested.")
        return ()

    if requested_devices:
        if len(requested_devices) < config.worker_processes:
            raise ValueError(
                f"--gpu_devices supplied {len(requested_devices)} device(s), but --worker_processes="
                f"{config.worker_processes} requires at least that many."
            )
        return requested_devices[: config.worker_processes]

    if config.worker_processes == 1:
        return ()

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to auto-select GPU devices for worker_processes > 1.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable, so worker_processes > 1 cannot auto-select GPU devices.")

    device_count = torch.cuda.device_count()
    if device_count < config.worker_processes:
        raise RuntimeError(
            f"Requested {config.worker_processes} worker process(es), but only {device_count} GPU(s) are available."
        )

    selected_devices = tuple(range(config.worker_processes))
    logger.info("Auto-selected GPU devices for workers: %s", ", ".join(str(device) for device in selected_devices))
    return selected_devices


def _pin_visible_gpu_device(device_id: int, logger: logging.Logger) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    logger.info("Pinned this process to GPU %s via CUDA_VISIBLE_DEVICES.", device_id)


def _partition_paths(paths: Sequence[Path], worker_processes: int) -> List[List[Path]]:
    partitions = [[] for _ in range(worker_processes)]
    for index, path in enumerate(paths):
        partitions[index % worker_processes].append(path)
    return [partition for partition in partitions if partition]


def _clone_config(config: PipelineConfig, *, verbose: bool | None = None) -> PipelineConfig:
    return PipelineConfig(
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        use_gpu=config.use_gpu,
        glob_pattern=config.glob_pattern,
        overwrite=config.overwrite,
        batch_size=config.batch_size,
        pipeline_batch_size=config.pipeline_batch_size,
        worker_processes=config.worker_processes,
        gpu_devices=config.gpu_devices,
        summary_length=config.summary_length,
        preprocessing_model=config.preprocessing_model,
        ner_model_name=config.ner_model_name,
        ner_do_sentence_level=config.ner_do_sentence_level,
        ner_do_postprocess=config.ner_do_postprocess,
        verbose=config.verbose if verbose is None else verbose,
    )


def _run_worker_process(
    worker_index: int,
    input_paths: Sequence[Path],
    config: PipelineConfig,
    layout: OutputLayout,
    assigned_gpu: int | None,
) -> Dict[str, Any]:
    logger, log_path = configure_logging(
        layout.logs,
        logger_name=f"opennyai_pipeline.worker{worker_index}",
        filename_suffix=f"_worker{worker_index}",
        include_stdout=False,
    )
    logger.info("Worker %s assigned %s discovered file(s).", worker_index, len(input_paths))
    if assigned_gpu is not None:
        _pin_visible_gpu_device(assigned_gpu, logger)

    worker_config = _clone_config(config, verbose=False)
    documents, skipped_records = prepare_documents(input_paths, worker_config, layout, logger)
    logger.info(
        "Worker %s prepared %s document(s) for inference and skipped %s.",
        worker_index,
        len(documents),
        len(skipped_records),
    )

    run_result = _run_pipeline_for_documents(
        documents=documents,
        config=worker_config,
        layout=layout,
        logger=logger,
        progress_desc=f"OpenNyAI worker{worker_index}",
    )
    return {
        "worker_index": worker_index,
        "log_path": str(log_path),
        "succeeded_file_ids": run_result["succeeded_file_ids"],
        "failed_records": run_result["failed_records"],
        "skipped_records": skipped_records,
    }


def run_pipeline(config: PipelineConfig, layout: OutputLayout, cli_command: str) -> Dict[str, Any]:
    """Run the local OpenNyAI pipeline and persist all requested outputs."""
    validate_pipeline_config(config)
    logger, log_path = configure_logging(layout.logs)
    logger.info("CLI command: %s", cli_command)
    logger.info("Input directory: %s", config.input_dir)
    logger.info("Output directory: %s", config.output_dir)

    config = PipelineConfig(
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        use_gpu=config.use_gpu,
        glob_pattern=config.glob_pattern,
        overwrite=config.overwrite,
        batch_size=config.batch_size,
        pipeline_batch_size=config.pipeline_batch_size,
        worker_processes=config.worker_processes,
        gpu_devices=config.gpu_devices,
        summary_length=normalize_summary_length(config.summary_length, logger),
        preprocessing_model=config.preprocessing_model,
        ner_model_name=config.ner_model_name,
        ner_do_sentence_level=config.ner_do_sentence_level,
        ner_do_postprocess=config.ner_do_postprocess,
        verbose=config.verbose,
    )

    discovered_files = discover_input_files(config, logger)
    validate_unique_file_ids(discovered_files)
    ensure_ner_model_available(config.ner_model_name, logger)

    effective_worker_processes = min(config.worker_processes, max(len(discovered_files), 1))
    if effective_worker_processes != config.worker_processes:
        logger.info(
            "Reducing worker_processes from %s to %s because only %s file(s) were discovered.",
            config.worker_processes,
            effective_worker_processes,
            len(discovered_files),
        )
        config = PipelineConfig(
            input_dir=config.input_dir,
            output_dir=config.output_dir,
            use_gpu=config.use_gpu,
            glob_pattern=config.glob_pattern,
            overwrite=config.overwrite,
            batch_size=config.batch_size,
            pipeline_batch_size=config.pipeline_batch_size,
            worker_processes=effective_worker_processes,
            gpu_devices=config.gpu_devices,
            summary_length=config.summary_length,
            preprocessing_model=config.preprocessing_model,
            ner_model_name=config.ner_model_name,
            ner_do_sentence_level=config.ner_do_sentence_level,
            ner_do_postprocess=config.ner_do_postprocess,
            verbose=config.verbose,
        )

    failed_records: List[Dict[str, str]] = []
    skipped_records: List[Dict[str, str]] = []
    succeeded_file_ids: List[str] = []
    worker_log_paths = [str(log_path)]

    if config.worker_processes > 1:
        assigned_gpus = _resolve_worker_gpu_devices(config, logger)
        path_partitions = _partition_paths(discovered_files, config.worker_processes)
        logger.info(
            "Launching %s worker process(es) with pipeline_batch_size=%s.",
            len(path_partitions),
            config.pipeline_batch_size,
        )
        if config.verbose:
            logger.info("Worker mode disables verbose OpenNyAI progress bars inside child processes to avoid mixed output.")

        mp_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(path_partitions), mp_context=mp_context) as executor:
            future_to_worker = {}
            for worker_index, input_paths in enumerate(path_partitions):
                assigned_gpu = assigned_gpus[worker_index] if assigned_gpus else None
                future = executor.submit(
                    _run_worker_process,
                    worker_index,
                    input_paths,
                    config,
                    layout,
                    assigned_gpu,
                )
                future_to_worker[future] = worker_index

            for future in as_completed(future_to_worker):
                worker_result = future.result()
                worker_log_paths.append(worker_result["log_path"])
                succeeded_file_ids.extend(worker_result["succeeded_file_ids"])
                failed_records.extend(worker_result["failed_records"])
                skipped_records.extend(worker_result["skipped_records"])
                logger.info(
                    "Worker %s finished: %s succeeded, %s failed, %s skipped.",
                    worker_result["worker_index"],
                    len(worker_result["succeeded_file_ids"]),
                    len(worker_result["failed_records"]),
                    len(worker_result["skipped_records"]),
                )
    else:
        assigned_gpus = _resolve_worker_gpu_devices(config, logger)
        if assigned_gpus:
            _pin_visible_gpu_device(assigned_gpus[0], logger)

        validate_gpu_request(config.use_gpu, logger)
        documents, skipped_records = prepare_documents(discovered_files, config, layout, logger)
        logger.info(
            "Prepared %s document(s) for inference, skipped %s.",
            len(documents),
            len(skipped_records),
        )
        run_result = _run_pipeline_for_documents(
            documents=documents,
            config=config,
            layout=layout,
            logger=logger,
            progress_desc="OpenNyAI",
        )
        succeeded_file_ids.extend(run_result["succeeded_file_ids"])
        failed_records.extend(run_result["failed_records"])

    run_report = {
        "total_files": len(discovered_files),
        "succeeded": len(succeeded_file_ids),
        "failed": len(failed_records),
        "skipped": len(skipped_records),
        "failed_files": [item["file_id"] for item in failed_records],
        "failed_details": failed_records,
        "skipped_files": [item["file_id"] for item in skipped_records],
        "skipped_details": skipped_records,
        "log_file": str(log_path),
        "worker_log_files": worker_log_paths,
        "pipeline_batch_size": config.pipeline_batch_size,
        "worker_processes": config.worker_processes,
        "gpu_devices": config.gpu_devices,
    }
    write_json(layout.logs / "run_report.json", run_report)

    tree_roots = [config.output_dir]
    if config.input_dir.exists():
        tree_roots.insert(0, config.input_dir)

    tree_report = compact_tree_report(tree_roots)
    return {
        "run_report": run_report,
        "tree_report": tree_report,
        "log_path": log_path,
        "succeeded_file_ids": succeeded_file_ids,
    }
