"""CLI for the opt-in table → knowledge → storage path."""

import json
import os
from pathlib import Path

from .knowledge import OntologyMapping, extract_knowledge
from .knowledge_store import databricks_plan, databricks_status, export_bundle, upload_adls, write_databricks, write_sqlite
from .tables import extract_pdf_tables


def add_parser(subparsers):
    parser = subparsers.add_parser("knowledge", help="Extract grounded table knowledge and persist it")
    commands = parser.add_subparsers(dest="knowledge_command", required=True)
    extract = commands.add_parser("extract", help="Extract ruled PDF tables using an explicit ontology mapping")
    extract.add_argument("source", type=Path)
    extract.add_argument("--mapping", required=True, type=Path)
    extract.add_argument("--out", required=True, type=Path)
    extract.add_argument("--parquet", action="store_true")
    sqlite = commands.add_parser("sqlite", help="Atomically insert a verified bundle into SQLite")
    sqlite.add_argument("bundle", type=Path)
    sqlite.add_argument("--database", type=Path, required=True)
    adls = commands.add_parser("adls", help="Upload a verified bundle to an existing ADLS filesystem")
    adls.add_argument("bundle", type=Path)
    adls.add_argument("--account-url", default=os.getenv("AZURE_STORAGE_ACCOUNT_URL"))
    adls.add_argument("--filesystem", required=True)
    adls.add_argument("--prefix", required=True)
    adls.add_argument("--receipt", type=Path, required=True)
    adls.add_argument("--execute", action="store_true")
    db = commands.add_parser("databricks", help="Plan or execute one MERGE into an existing Delta table")
    db.add_argument("bundle", type=Path)
    db.add_argument("--host", default=os.getenv("DATABRICKS_HOST"))
    db.add_argument("--warehouse-id", default=os.getenv("DATABRICKS_WAREHOUSE_ID"))
    db.add_argument("--table", required=True)
    db.add_argument("--receipt", type=Path)
    db.add_argument("--execute", action="store_true")
    status = commands.add_parser("databricks-status", help="Look up a saved statement without resubmitting")
    status.add_argument("--receipt", type=Path, required=True)
    status.add_argument("--host", default=os.getenv("DATABRICKS_HOST"))


def handle(args):
    try:
        if args.knowledge_command == "extract":
            mapping = OntologyMapping.model_validate_json(args.mapping.read_bytes())
            package = extract_knowledge(extract_pdf_tables(args.source), mapping)
            result = export_bundle(package, args.out, parquet=args.parquet)
            result["entities"] = len(package.entities)
            result["relations"] = len(package.relations)
            result["warnings"] = package.warnings
        elif args.knowledge_command == "sqlite":
            result = write_sqlite(args.bundle, args.database)
        elif args.knowledge_command == "adls":
            if not args.execute or not args.account_url:
                raise ValueError("ADLS upload requires --execute and an explicit account URL")
            result = upload_adls(args.bundle, account_url=args.account_url, filesystem=args.filesystem,
                                 prefix=args.prefix, receipt=args.receipt)
        elif args.knowledge_command == "databricks":
            if not args.warehouse_id:
                raise ValueError("A SQL warehouse ID is required")
            if not args.execute:
                result = databricks_plan(args.bundle, table=args.table, warehouse_id=args.warehouse_id)
            else:
                if not args.receipt or not args.host:
                    raise ValueError("Execution requires --receipt and --host")
                result = write_databricks(args.bundle, host=args.host, token=os.getenv("DATABRICKS_TOKEN", ""),
                    table=args.table, warehouse_id=args.warehouse_id, receipt=args.receipt)
        else:
            if not args.host:
                raise ValueError("A workspace host is required")
            result = databricks_status(args.receipt, host=args.host, token=os.getenv("DATABRICKS_TOKEN", ""))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") in {None, "complete", "committed"} else 1
    except Exception as exc:
        # Provider messages/URLs can contain credentials or source text.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                          "message": "Operation failed; verify inputs, optional dependencies, and any reserved receipt."}))
        return 1
