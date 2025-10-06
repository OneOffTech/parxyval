import json
import logging
import os
from typing import Optional

import typer

import datasets

from datasets import load_dataset
from datasets import load_dataset_builder
from datasets import disable_progress_bars

from rich import print

from parxyval.models.dataset import Dataset
from parxyval.data_handler.doclaynet_to_parxy import doclaynet_v12_to_parxy


app = typer.Typer()


@app.command()
def download(
    limit: Optional[int] = typer.Option(
        100,
        '--limit',
        '-l',
        help='Limit the number of entries to download from the dataset.',
    ),
    skip: Optional[int] = typer.Option(
        None,
        '--skip',
        '-s',
        help='Skip the specified entries from the beginning of the dataset.',
    ),
    output_path: Optional[str] = typer.Option(
        'data/doclaynet',
        '--output',
        '-o',
        help='Folder to store the dataset.',
        exists=False,
        file_okay=False,
        dir_okay=True,
    ),
    include_pdf: Optional[bool] = typer.Option(
        False,
        '--include-pdf',
        help='Download the PDF files for each entry from the dataset.',
    ),
    debug: Optional[bool] = typer.Option(
        False,
        '--debug',
        '-v',
        help='Print debug information.',
    ),
):
    """Create a ground truth by downloading the dataset and converting to to Parxy format."""

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(asctime)s : %(levelname)s : %(name)s : %(message)s',
    )

    # Set logging level to WARNING or ERROR
    datasets.logging.set_verbosity_warning()
    disable_progress_bars()

    print(f'Creating ground truth from {Dataset.DOCLAYNETV2.value}...')

    json_folder = f'{output_path}/json'
    pdf_folder = f'{output_path}/pdf' if include_pdf else None

    dataset_columns = ['metadata']  # Columns to extract from the HF dataset

    # Create directories if they don't exist
    if json_folder:
        dataset_columns.extend(
            ['pdf_cells', 'category_id']
        )  # Columns needed to download JSON representation
        logging.debug(f'Output folder (json): {json_folder}')
        if not os.path.isdir(json_folder):
            os.makedirs(json_folder)

    if pdf_folder:
        dataset_columns.extend(['pdf'])  # Column needed to download PDF files
        logging.debug(f'Output folder (pdf): {pdf_folder}')
        if not os.path.isdir(pdf_folder):
            os.makedirs(pdf_folder)

    # Initialize the generator to load the dataset
    # TODO: handle when user is not authenticated to HF as the download require authentication

    ds_builder = load_dataset_builder(Dataset.DOCLAYNETV2.value)

    print(f'Dataset: {Dataset.DOCLAYNETV2.value}')
    print(f'Dataset: {ds_builder.info.dataset_name} ({ds_builder.info.builder_name})')
    print(f'Split: Train ({ds_builder.info.splits["train"].num_examples} examples)')

    print(f'Creating ground truth (limit={limit}, skip={skip})...')

    data = load_dataset(
        Dataset.DOCLAYNETV2.value,
        split='train',
        streaming=True,
        columns=dataset_columns,
    )

    count_processed = 0
    count_skipped = 0
    for row in data:
        # Skip the first `skip` entries
        if skip is not None and count_skipped < skip:
            count_skipped += 1
            continue

        logging.debug(f'Processing {row["metadata"]["page_hash"]}...')

        # Convert json to Parxy document
        if json_folder:
            res = doclaynet_v12_to_parxy(
                row['pdf_cells'], row['metadata'], row['category_id']
            )
            with open(
                os.path.join(json_folder, row['metadata']['page_hash'] + '.json'), 'w'
            ) as json_file:
                json.dump(res.model_dump(), json_file)

        # Store PDF file
        if pdf_folder:
            with open(
                os.path.join(pdf_folder, row['metadata']['page_hash'] + '.pdf'), 'wb'
            ) as pdf_file:
                pdf_file.write(row['pdf'])

        count_processed += 1

        # Terminate after `n_limit` processed entries
        if limit is not None and count_processed >= limit:
            break

    logging.debug(f'Skipped {count_skipped} records')
    logging.debug(f'Processed {count_processed} records')

    print(f'Ground truth created in [green]{json_folder}[/green].')
    print(f'Entries: {count_processed}')
