import json
import logging
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from parxy_core.models import Document

from parxyval.evaluation.factory import get_metric, get_metrics_name

from typing import Optional, List

import typer

app = typer.Typer()

@app.command()
def evaluate(
    driver: Optional[str] = typer.Argument(
        default='pymupdf',
        help='The Parxy driver to evaluate. If omitted defaults to pymupdf.',
    ),
    metrics: Optional[List[str]] = typer.Option(
        ["sequence_matcher"],
        '--metric',
        '-m',
        help='The metric to evaluate.',
    ),
    all_metrics: Optional[bool] = typer.Option(
        False,
        '--all-metrics',
        '-a',
        help='Evaluate using all defined metrics.',
    ),
    golden_folder: Optional[str] = typer.Option(
        'data/doclaynet/json',
        '--golden',
        '-g',
        help='Folder with the ground truth for the dataset.',
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    input_folder: Optional[str] = typer.Option(
        'data/doclaynet/processed/pymupdf',
        '--input',
        '-i',
        help='Folder with the parsed documents to use for the evaluation.',
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_folder: Optional[str] = typer.Option(
        'data/doclaynet/results/',
        '--output',
        '-o',
        help='Folder to store the evaluation results.',
        exists=False,
        file_okay=False,
        dir_okay=True,
    ),
):

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
   
    metrics_name = [metric.lower().strip().replace("-", "_").replace(" ", "_")
                    for metric in metrics if get_metric(metric)]
    
    if all_metrics is True:
        metrics_name = get_metrics_name()

    if not os.path.exists(input_folder):
        logging.debug(f"The specified input folder [{input_folder}] does not exist!")
        raise typer.Exit(code=422)
    if not os.path.exists(golden_folder):
        logging.debug(f"The specified golden folder [{golden_folder}] does not exist!")
        raise typer.Exit(code=422)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if len(metrics_name) == 0:
        logging.debug(f"The specified metrics are not implemented!")
        raise typer.Exit(code=422)
    
    metrics_fn = list([get_metric(metric) for metric in metrics_name])

    logging.debug(f"Input folder: {input_folder}")
    logging.debug(f"Output folder: {output_folder}")
    logging.debug(f"Metric: {metrics_name}")

    res_list = []
    for filename in os.listdir(input_folder):
        logging.debug(f"Processing {filename}")

        # Read the parsing result
        with open(os.path.join(input_folder, filename), "r") as f:
            doc = Document(**json.loads(f.read()))

        # Read the ground truth
        try:
            with open(os.path.join(golden_folder, filename), "r") as f:
                golden_doc = Document(**json.loads(f.read()))
        except FileNotFoundError:
            logging.error(f"File [{filename}] does not exist!")
            continue

        base_data = {
            "filename": filename,
            "collection": golden_doc.source_data["collection"],
            "doc_category": golden_doc.source_data["doc_category"],
            "original_filename": golden_doc.source_data["original_filename"],
            "page_no": golden_doc.source_data["page_no"],
            "processing_time_seconds": doc.source_data["processing_time_seconds"],
        }

        # merge all metrics dicts into one
        metrics_dict = {}
        for metric_fn in metrics_fn:
            metrics_dict.update(metric_fn(golden_doc, doc))

        # merge base data + metrics
        row = {**base_data, **metrics_dict}
        res_list.append(row)

    logging.debug(f"Processed {len(res_list)} documents.")
    timestamp_str = str(time.time()).replace(".", "")
    res_df = pd.DataFrame(res_list)
    input_folder_name = input_folder.replace(os.sep, "/").replace("\\", "/")
    input_folder_name = input_folder_name.split("/")[-1].replace(" ", "_").lower()
    res_df.to_csv(os.path.join(output_folder, f"eval_{input_folder_name}_{timestamp_str}.csv"), index=False)
    logging.debug(f"Results written to {output_folder}/eval_{input_folder_name}_{timestamp_str}.csv")


    # TODO: print out some basic data from res_df, like average score for each metric and average processing_time_seconds
