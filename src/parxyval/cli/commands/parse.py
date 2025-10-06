import json
import logging
import os
import sys
import time

from datasets import load_dataset, IterableDatasetDict

from parxy_core.facade import Parxy
from parxy_core.exceptions import ParsingException

from parxyval.models.dataset import Dataset

from typing import Optional, List

from rich import print
from rich.progress import track, Progress, SpinnerColumn, TextColumn

import typer

import pymupdf

app = typer.Typer()

@app.command()
def parse(
    driver: Optional[str] = typer.Option(
        "pymupdf",
        '--driver',
        '-d',
        help='The Parxy driver to use.',
    ),
    limit: Optional[int] = typer.Option(
        100,
        '--limit',
        '-l',
        help='Limit the number of documents to process.',
    ),
    skip: Optional[int] = typer.Option(
        None,
        '--skip',
        '-s',
        help='Skip the specified amount of documents.',
    ),
    input_path: Optional[str] = typer.Option(
        'data/doclaynet/pdf',
        '--input',
        '-i',
        help='Folder with the documents to process.',
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_path: Optional[str] = typer.Option(
        'data/doclaynet/processed/',
        '--output',
        '-o',
        help='Folder to store the processed documents.',
        exists=False,
        file_okay=False,
        dir_okay=True,
    ),
    ignore_local_files: Optional[bool] = typer.Option(
        False,
        '--ignore-local',
        help='Ignore existing downloaded documents and download them from the dataset. Use in case you have a different amount of files locally than the ones you would like to process.',
    ),
    level: Optional[str] = typer.Option(
        "block",
        '--level',
        help='The level at which perform the processing.',
    ),
):
    """Parse documents using Parxy."""

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )

    driver = driver.lower().strip()
    output_folder = output_path + driver

    # Check if the input and output dir exist
    if input_path and not os.path.isdir(input_path):
        sys.exit(f"The specified input folder [{input_path}] does not exist!")
        
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    print(f"Parsing documents from {Dataset.DOCLAYNETV2.value} using {driver}...")
    print()

    # Initialize the object to iterate over
    if ignore_local_files:
        # Hugging Face dataset
        logging.debug(f"Input dataset: ds4sd/DocLayNet-v1.2")
        iterator: IterableDatasetDict = load_dataset(Dataset.DOCLAYNETV2.value, split="train",
                                                     streaming=True, columns=["metadata", "pdf"])
    else:
        # List of filenames
        iterator: list[str] = os.listdir(input_path)
        logging.debug(f"Input folder: {input_path}")

    logging.debug(f"Output folder: {output_folder}")

    # Init parxy
    parxy = Parxy.driver(driver)

    # TODO: llmwhisperer is configured to use High Quality with Form Elements by default, allow to change it

    count_processed = 0
    count_skipped = 0

    min_processing_time = 999
    max_processing_time = -1

    total = limit if skip is None else limit+skip

    pymupdf.TOOLS.mupdf_display_errors(False)
    pymupdf.TOOLS.mupdf_display_warnings(False)

    errors = []

    for entry in track(iterator, total=total, description="Processing...", transient=True):

        # Skip the first `skip` entry
        if skip is not None and count_skipped < skip:
            count_skipped += 1
            continue

        # Select the right input type
        if ignore_local_files:
            output_filename = entry["metadata"]["page_hash"] + ".json"
            doc_to_process: bytes = entry["pdf"]
        else:
            output_filename = entry.replace(".pdf", ".json")
            doc_to_process: str = os.path.join(input_path, entry)

        logging.debug(f"Processing {output_filename}")

        # Track time taken for the next action
        start_time = time.perf_counter()


        try:
            
            parsed_document = parxy.parse(doc_to_process, level=level)

            # Print time taken and number of results
            time_taken = time.perf_counter() - start_time

            parsed_document.source_data = {
                # TODO: add also metadata coming from the dataset like filename, document name, category, ...
                'processing_time_seconds': time_taken
            }

            res_json = parsed_document.model_dump()

            # Store the JSON result
            with open(os.path.join(output_folder, output_filename), "w") as file:
                json.dump(res_json, file)

            count_processed += 1

            min_processing_time = min(min_processing_time, time_taken)
            max_processing_time = max(max_processing_time, time_taken)

        except ParsingException as pex:
            errors.append(f"{output_filename}: {str(pex)}")
            # print(pex)
            continue


        # Terminate after `limit` processed entries
        if limit is not None and count_processed >= limit:
            break

    logging.debug(f"Skipped {count_skipped} documents")
    logging.debug(f"Processed {count_processed} documents")
    logging.debug(f"Errors {len(errors)} documents")

    print(f"[green]Documents parsed in {output_folder}[/green]")

    print()

    print(f"Processed {count_processed} documents")
    print(f"Processing time between {min_processing_time} and {max_processing_time} seconds per document")

    if len(errors) > 0:
        
        print(f"{len(errors)} Errors")
        print(errors)
