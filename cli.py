"""Small Rich terminal interface for the academic demo."""
import asyncio

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from app.config import get_settings
from app.llm import ProviderError, check_connection
from app.orchestration.engine import ResearchEngine
from app.rag.vector_store import VectorStore
from app.store import list_sessions


async def main():
    console = Console()
    settings = get_settings()
    while True:
        console.print("\n[bold]Academic Research Assistant[/bold]")
        console.print("1. Check system status\n2. Ingest research papers\n3. View indexed papers\n"
                      "4. Start new research\n5. View previous research sessions\n6. Exit")
        choice = Prompt.ask("Select", choices=["1", "2", "3", "4", "5", "6"])
        if choice == "6":
            return
        if choice == "1":
            try:
                console.print(await check_connection(settings))
            except ProviderError as exc:
                console.print(f"[red]Provider: {exc}[/red]")
            console.print(f"Indexed papers: {len(VectorStore(settings).papers())}")
        elif choice == "2":
            reindex = Prompt.ask("Re-index all PDFs?", choices=["y", "n"], default="n") == "y"
            for filename, result in VectorStore(settings).ingest(reindex=reindex).items():
                console.print(f"{filename}: {result}")
        elif choice == "3":
            table = Table("Title", "Filename", "Document ID")
            for paper in VectorStore(settings).papers():
                table.add_row(paper["paper_title"], paper["filename"], paper["document_id"])
            console.print(table)
        elif choice == "4":
            question = Prompt.ask("Research question")
            state = await ResearchEngine(settings).run(question, console.print)
            console.print(f"Status: {state.status.value}")
            if state.final_report:
                console.print(f"Final report: {state.final_report.path}")
                console.print(state.final_report.markdown)
            elif state.events and state.events[-1].error:
                console.print(f"[red]{state.events[-1].error}[/red]")
        else:
            table = Table("Session ID", "Question", "Status", "Report")
            for state in list_sessions(settings.sessions_directory):
                table.add_row(state.session_id, state.question, state.status.value,
                              state.final_report.path if state.final_report else "")
            console.print(table)


if __name__ == "__main__":
    asyncio.run(main())
