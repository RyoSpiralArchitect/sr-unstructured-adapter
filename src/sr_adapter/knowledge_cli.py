"""CLI for opt-in document relations, table knowledge and storage."""

import json
import os
from pathlib import Path

from .knowledge import OntologyMapping, extract_knowledge
from .knowledge_store import databricks_plan, databricks_status, export_bundle, upload_adls, write_databricks, write_sqlite
from .tables import extract_pdf_tables


def add_parser(subparsers):
    parser = subparsers.add_parser("knowledge", help="Extract source-linked relations and persist them")
    commands = parser.add_subparsers(dest="knowledge_command", required=True)
    extract = commands.add_parser("extract", help="Extract ruled PDF tables using an explicit ontology mapping")
    extract.add_argument("source", type=Path)
    extract.add_argument("--mapping", required=True, type=Path)
    extract.add_argument("--out", required=True, type=Path)
    extract.add_argument("--parquet", action="store_true")
    narrative = commands.add_parser("narrative", help="Extract model-proposed relation tables with source evidence")
    narrative.add_argument("source", type=Path)
    narrative.add_argument("--ontology", required=True, type=Path)
    narrative.add_argument("--project-endpoint", required=True)
    narrative.add_argument("--model", required=True)
    narrative.add_argument("--out", required=True, type=Path)
    narrative.add_argument("--execute", action="store_true")
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
    foundry = commands.add_parser("foundry", help="Export relation contexts and Foundry IQ / prompt-agent configuration")
    foundry.add_argument("bundles", nargs="+", type=Path)
    foundry.add_argument("--out", required=True, type=Path)
    foundry.add_argument("--prefix", default="sr-relations")
    foundry.add_argument("--target", type=Path, help="Optional JSON with existing Search/Foundry endpoints and deployment names")
    publish = commands.add_parser("foundry-publish", help="Publish a verified Foundry export to explicitly configured existing services")
    publish.add_argument("bundle", type=Path)
    publish.add_argument("--target", type=Path, required=True)
    publish.add_argument("--receipt", type=Path, required=True)
    publish.add_argument("--execute", action="store_true")
    fabric = commands.add_parser("fabric", help="Export verified normalized Parquet and a Fabric load plan (no cloud writes)")
    fabric.add_argument("bundles", nargs="+", type=Path)
    fabric.add_argument("--out", required=True, type=Path)
    fabric.add_argument("--target", required=True, type=Path)
    fabric.add_argument("--mode", choices=("knowledge", "tabular"), default="knowledge",
                        help="tabular: scalar columns from selected native PPTX slides, without model inference")
    fabric.add_argument("--slides", nargs="+", type=int, help="One-based physical PPTX slide numbers (tabular mode)")
    fabric.add_argument("--page-labels", nargs="+", help="Optional printed page labels, aligned with --slides")
    for command in ("fabric-publish", "fabric-status"):
        fabric_action = commands.add_parser(command, help="Publish or inspect an explicit Fabric snapshot receipt")
        fabric_action.add_argument("bundle", type=Path)
        fabric_action.add_argument("--target", required=True, type=Path)
        fabric_action.add_argument("--receipt", required=True, type=Path)
        if command == "fabric-publish":
            fabric_action.add_argument("--execute", action="store_true")


def handle(args):
    try:
        if args.knowledge_command == "extract":
            mapping = OntologyMapping.model_validate_json(args.mapping.read_bytes())
            package = extract_knowledge(extract_pdf_tables(args.source), mapping)
            result = export_bundle(package, args.out, parquet=args.parquet)
            result["entities"] = len(package.entities)
            result["relations"] = len(package.relations)
            result["warnings"] = package.warnings
        elif args.knowledge_command == "narrative":
            from .narrative import NarrativeOntology
            from .narrative_runtime import extract_narrative
            result = extract_narrative(args.source, ontology=NarrativeOntology.model_validate_json(args.ontology.read_bytes()),
                endpoint=args.project_endpoint, model=args.model, destination=args.out, execute=args.execute)
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
        elif args.knowledge_command == "foundry":
            from .foundry import FoundryTarget, export_foundry
            target = FoundryTarget.model_validate_json(args.target.read_bytes()) if args.target else None
            result = export_foundry(args.bundles, args.out, prefix=args.prefix, target=target)
        elif args.knowledge_command == "foundry-publish":
            from .foundry import FoundryTarget
            from .foundry_runtime import publish_foundry
            if not args.execute:
                raise ValueError('Publication requires --execute; inspect the export deployment-plan.json first')
            result = publish_foundry(args.bundle, target=FoundryTarget.model_validate_json(args.target.read_bytes()), receipt=args.receipt)
        elif args.knowledge_command == "fabric":
            from .fabric import FabricTarget, export_fabric
            target = FabricTarget.model_validate_json(args.target.read_bytes())
            if args.mode == "tabular":
                from .fabric_tabular import export_tabular
                if len(args.bundles) != 1 or args.bundles[0].suffix.lower() != ".pptx":
                    raise ValueError("Tabular mode requires one PPTX and explicit --slides")
                result = export_tabular(args.bundles[0], args.out, target=target,
                                        slides=args.slides, page_labels=args.page_labels)
            else:
                if args.slides is not None or args.page_labels is not None:
                    raise ValueError("Slide selection requires --mode tabular")
                result = export_fabric(args.bundles, args.out, target=target)
        elif args.knowledge_command in {"fabric-publish", "fabric-status"}:
            from .fabric import FabricTarget
            from .fabric_runtime import fabric_status, publish_fabric
            target = FabricTarget.model_validate_json(args.target.read_bytes())
            if args.knowledge_command == "fabric-publish":
                if not args.execute:
                    raise ValueError("Publication requires --execute; inspect deployment-plan.json first")
                result = publish_fabric(args.bundle, target=target, receipt=args.receipt)
            else:
                result = fabric_status(args.bundle, target=target, receipt=args.receipt)
        else:
            if not args.host:
                raise ValueError("A workspace host is required")
            result = databricks_status(args.receipt, host=args.host, token=os.getenv("DATABRICKS_TOKEN", ""))
        # JSON escapes preserve Unicode data even on legacy Windows consoles.
        # Bundle files remain UTF-8; reporting must not fail after a successful write.
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0 if result.get("status") in {None, "prepared", "complete", "committed", "configured"} else 1
    except Exception as exc:
        # Provider messages/URLs can contain credentials or source text.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                          "message": "Operation failed; verify inputs, optional dependencies, and any reserved receipt."}))
        return 1
