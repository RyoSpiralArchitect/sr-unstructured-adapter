"""Local ontology CLI; remote extraction requires an explicit --execute."""
import json
from pathlib import Path


def add_parser(commands):
    parser = commands.add_parser("ontology", help="Ingest raw sources into one reusable local SQLite ontology")
    sub = parser.add_subparsers(dest="ontology_command", required=True)
    for name in ("ingest", "import", "list", "search", "inspect", "status", "tools", "call"):
        command = sub.add_parser(name)
        command.add_argument("--database", required=True, type=Path)
        if name in {"ingest", "import"}:
            command.add_argument("source", type=Path)
            command.add_argument("--name", help="Document entry name; defaults to its source filename")
            command.add_argument("--expected-snapshot", help="Current version required to replace a document")
        if name not in {"status", "tools", "call"}:
            command.add_argument("--collection", default="default")
        if name == "ingest":
            command.add_argument("--model", required=True)
            command.add_argument("--execute", action="store_true", help="Send source content to the selected OpenAI model")
        if name == "search":
            command.add_argument("query", nargs="?", default="")
            command.add_argument("--name")
            command.add_argument("--snapshot-id")
            command.add_argument("--kind", choices=("all", "claim", "term"), default="all")
            command.add_argument("--status")
            command.add_argument("--modality")
            command.add_argument("--limit", type=int, default=20)
            command.add_argument("--offset", type=int, default=0)
        if name == "inspect":
            command.add_argument("--name", required=True)
            command.add_argument("--snapshot-id", required=True)
            command.add_argument("--kind", required=True, choices=("claim", "term"))
            command.add_argument("--id", required=True)
        if name == "status":
            command.add_argument("run_id")
        if name in {"tools", "call"}:
            command.add_argument("--collections", nargs="+", required=True)
        if name == "call":
            command.add_argument("tool")
            command.add_argument("--arguments", type=Path, required=True, help="JSON object of tool arguments")


def handle(args):
    from .local_ontology import import_ontology, inspect_ontology, list_ontologies, search_ontology
    from .ontology_tools import OntologyTools
    command = args.ontology_command
    if command == "ingest":
        from .ontology_ingest import ingest_ontology_source
        return ingest_ontology_source(args.source, args.database, model=args.model, collection=args.collection,
            name=args.name, expected_snapshot=args.expected_snapshot, execute=args.execute)
    if command == "import":
        return import_ontology(args.source, args.database, collection=args.collection,
            name=args.name, expected_snapshot=args.expected_snapshot)
    if command == "list":
        return list_ontologies(args.database, collection=args.collection)
    if command == "search":
        return search_ontology(args.database, **{key: getattr(args, key) for key in
            ("collection", "query", "name", "snapshot_id", "kind", "status", "modality", "limit", "offset")})
    if command == "inspect":
        return inspect_ontology(args.database, **{key: getattr(args, key) for key in
            ("collection", "name", "snapshot_id", "kind", "id")})
    if command == "status":
        from .ontology_ingest import ontology_ingestion_status
        return ontology_ingestion_status(args.database, args.run_id)
    tools = OntologyTools(args.database, tuple(args.collections))
    if command == "tools":
        return {"status": "complete", "tools": tools.definitions()}
    with args.arguments.open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("Tool arguments exceed 64 KiB")
    return tools.call(args.tool, json.loads(raw))
