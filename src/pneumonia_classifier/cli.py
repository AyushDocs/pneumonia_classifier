"""Console script for pneumonia_classifier."""
import pneumonia_classifier

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for pneumonia_classifier."""
    console.print("Replace this message by putting your code into "
               "pneumonia_classifier.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
