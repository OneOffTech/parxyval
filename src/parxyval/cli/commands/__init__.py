
import typer

from .download import app as download_dataset_app
from .parse import app as parse_app
from .evaluate import app as evaluation_app

app = typer.Typer()

app.add_typer(download_dataset_app)
app.add_typer(parse_app)
app.add_typer(evaluation_app)