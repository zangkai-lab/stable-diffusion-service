import click
import uvicorn


@click.group()
def main() -> None:
    pass


@main.command()
@click.option(
    "-p",
    "--port",
    default=5050,
    show_default=True,
    type=int,
    help="The port to listen on.",
)
def run(port: int) -> None:
    """Run the service."""
    uvicorn.run(
        "services.apis:app",
        host="0.0.0.0",
        port=port,
    )
