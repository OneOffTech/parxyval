import glob
import json
import logging
import os
import sys
from pathlib import Path

from typing import Annotated, Dict, List, Optional, Tuple

import typer

from .commands import app as sub_commands

app = typer.Typer(
    name='parxyval',
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

app.add_typer(sub_commands)


@app.callback()
def main():
    """Parxyval, The Developer's Knight of the Parsing Table at your service for evaluating document parsers."""
    pass


if __name__ == '__main__':
    app()
