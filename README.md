![pypi](https://img.shields.io/pypi/v/parxyval.svg)
![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

# Parxyval


Parxyval – The Developer's Knight of the Parsing Table.

An evaluation framework for document parsing, inspired by the quest for the Holy Grail. 🏰⚔️

In a world of imperfect parsers, Parxyval helps you measure, compare, and discover which tool truly preserves the meaning of your documents. Benchmark precision, recall, structure, and reliability across multiple parsing services — and let your pipeline find its Grail.


**Requirements**

- Python 3.12 or above.
- A Hugging Face account for downloading datasets that requires a login


**Next steps**

- [Getting started](#getting-started)
  - [Available commands](#command-reference)
- [Benchmarks](#benchmarks)
- [Supported evaluations](#evaluations)



## Getting Started

Parxyval is an unstructured document processing evaluation framework offering a CLI interface to benchmark PDF parsing solutions. Follow these steps to get started:

Parxyval is available on Pypi. You can try it using `uv`.

```bash
uvx parxyval --help
```



1. **Download the Dataset**
```bash
# Download sample documents from DocLayNet dataset
parxyval download --limit 100 --include-pdf
```

The ground truth is stored in `./data/doclaynet/json` while pdf files are stored in `./data/doclaynet/pdf`


2. **Parse Documents**
```bash
# Parse PDFs using your chosen driver (default: pymupdf)
parxyval parse --driver pymupdf 

# you can personalize input and output locations
# --input data/doclaynet/pdf --output data/doclaynet/processed
```

Parxyval supports all drivers available in [Parxy](https://github.com/OneOffTech/parxy).

Pdf files are read from `./data/doclaynet/pdf` and the parser outputs is written in `./data/doclaynet/processed/{driver}`, e.g. `./data/doclaynet/processed/pymupdf`

3. **Evaluate Results**

```bash
# Run evaluation with selected metrics
parxyval evaluate --metric sequence_matcher --metric bleu_score --input ./data/doclaynet/processed/pymupdf
```

### Command Reference

#### `parxyval download`
Download documents from the DocLayNet dataset.

Options:
- `--limit, -l`: Number of entries to download (default: 100)
- `--skip, -s`: Skip specified number of entries
- `--output, -o`: Output folder path (default: data/doclaynet)
- `--include-pdf`: Download PDF files (default: False)

#### `parxyval parse`
Parse PDF documents using specified driver.

Options:
- `--driver, -d`: Parser driver to use (default: pymupdf)
- `--limit, -l`: Maximum documents to process (default: 100)
- `--skip, -s`: Skip specified number of documents
- `--input, -i`: Input folder with PDFs (default: data/doclaynet/pdf)
- `--output, -o`: Output folder for results (default: data/doclaynet/processed)

#### `parxyval evaluate`
Evaluate parsing results against ground truth.

Arguments:
- `driver`: Parser driver to evaluate (default: pymupdf)

Options:
- `--metric, -m`: Metrics to use (can be specified multiple times)
- `--golden, -g`: Ground truth folder (default: data/doclaynet/json)
- `--input, -i`: Parsed documents folder (default: data/doclaynet/processed/pymupdf)
- `--output, -o`: Results output folder (default: data/doclaynet/results)


## Benchmarks

Parxyval supports various benchmarks for the evaluation of document processing services.

- [DocLayNet](https://huggingface.co/datasets/ds4sd/DocLayNet-v1.2): Evaluate text and layout using the DocLayNet v1.2 dataset.


_Datasets we are evaluating to support:_

- [DP-Bench: Document Parsing Benchmark](https://huggingface.co/datasets/upstage/dp-bench)
- [OmniDocBench](https://huggingface.co/datasets/opendatalab/OmniDocBench)

## Evaluations

Parxyval provides a comprehensive suite of text evaluation metrics to assess the quality of PDF parsing results. Each metric focuses on different aspects of text similarity and accuracy:

### Text Similarity Metrics

- **Sequence Matcher**: Measures the similarity between two texts using Python's difflib sequence matcher. Ideal for detecting overall textual similarities and differences.

- **Jaccard Similarity**: Computes the similarity between page contents by measuring the intersection over union of their token sets. Perfect for assessing vocabulary overlap between parsed and reference texts.

- **Edit Distance**: Calculates the normalized Levenshtein distance between texts, measuring the minimum number of single-character edits required to change one text into another. Useful for identifying character-level parsing accuracy.

### Natural Language Processing Metrics

- **BLEU Score**: A precision-based metric that compares n-grams between the parsed and reference texts. Particularly effective for evaluating the preservation of word sequences and phrases.

- **METEOR Score**: Advanced metric that considers stemming, synonymy, and paraphrasing. Provides a more nuanced evaluation of semantic similarity between parsed and reference texts.

### Information Retrieval Metrics

- **Precision**: Measures the accuracy of the parsed text by calculating the proportion of correctly parsed tokens relative to all tokens in the parsed text.

- **Recall**: Evaluates completeness by calculating the proportion of reference tokens that were correctly captured in the parsed text.

- **F1 Score**: The harmonic mean of precision and recall, providing a balanced measure of parsing accuracy.

All metrics are computed page-wise and then averaged across the entire document, ensuring a comprehensive evaluation of parsing quality at both local and global levels.


## Security Vulnerabilities

Please review our [security policy](./.github/SECURITY.md) on how to report security vulnerabilities.


## Supporters

The project is provided and supported by OneOff-Tech (UG) and Alessio Vertemati.

<p align="left"><a href="https://oneofftech.de" target="_blank"><img src="https://raw.githubusercontent.com/OneOffTech/.github/main/art/oneofftech-logo.svg" width="200"></a></p>


## Licence and Copyright

Parxy is licensed under the [MIT licence](./LICENCE).

- Copyright (c) 2025-present Alessio Vertemati, @avvertix
- Copyright (c) 2025-present Oneoff-tech UG, www.oneofftech.de
- All contributors

