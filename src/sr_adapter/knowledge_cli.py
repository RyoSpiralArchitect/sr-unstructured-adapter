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
    narrative.add_argument("--label-policy", choices=("literal", "preserve"), default="literal",
                           help="preserve: retain nonliteral model labels with explicit match columns")
    visual = commands.add_parser("visual", help="Extract unreviewed image relation candidates with visual regions")
    visual.add_argument("source", type=Path)
    visual.add_argument("--project-endpoint", required=True)
    visual.add_argument("--model", required=True)
    visual.add_argument("--page-label", required=True)
    visual.add_argument("--strategy", choices=("grounded", "direct"), default="grounded")
    visual.add_argument("--out", required=True, type=Path)
    visual.add_argument("--execute", action="store_true")
    dataset = commands.add_parser("dataset", help="Export destination-independent relation tables and replay sources (local only)")
    dataset.add_argument("bundles", nargs="+", type=Path)
    dataset.add_argument("--profile", choices=("narrative", "visual"), required=True)
    dataset.add_argument("--out", required=True, type=Path)
    dataset.add_argument("--review", type=Path, help="Optional visual review findings JSON")
    view = commands.add_parser("dataset-view", help="Project a saved relation dataset into search cards or a graph (local only)")
    view.add_argument("bundle", type=Path)
    view.add_argument("--view", choices=("search", "graph"), required=True)
    view.add_argument("--out", type=Path, required=True)
    dataset_sqlite = commands.add_parser("dataset-sqlite", help="Commit every table of a verified relation dataset into SQLite")
    dataset_sqlite.add_argument("bundle", type=Path)
    dataset_sqlite.add_argument("--database", required=True, type=Path)
    receive = commands.add_parser("dataset-receive", help="Atomically receive paired search/graph views and activate a local collection")
    receive.add_argument("search_bundle", type=Path)
    receive.add_argument("graph_bundle", type=Path)
    receive.add_argument("--database", required=True, type=Path)
    receive.add_argument("--collection", required=True)
    receive.add_argument("--expected-snapshot", help="Required current source snapshot when replacing an active collection")
    query = commands.add_parser("dataset-query", help="Search received relations with explicit source/status context")
    query.add_argument("--database", required=True, type=Path)
    query.add_argument("--collection", required=True)
    query.add_argument("--snapshot", help="Previously received source snapshot; defaults to active")
    query.add_argument("--query", default="")
    query.add_argument("--scope", choices=("relation", "context"), default="relation")
    query.add_argument("--assertion-status", choices=("asserted", "denied", "unknown", "ambiguous"))
    query.add_argument("--support", choices=("explicit", "uncertain"))
    query.add_argument("--limit", type=int, default=20)
    query.add_argument("--offset", type=int, default=0)
    trace = commands.add_parser("dataset-trace", help="Trace a received search card to its exact source evidence")
    trace.add_argument("card_id")
    trace.add_argument("--database", required=True, type=Path)
    trace.add_argument("--collection", required=True)
    trace.add_argument("--snapshot", help="Source snapshot returned by the original query")
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
    fabric.add_argument("--mode", choices=("knowledge", "tabular", "visual", "narrative"), default="knowledge",
                        help="tabular: native PPTX; visual: image candidates; narrative: document relations, all scalar columns")
    fabric.add_argument("--slides", nargs="+", type=int, help="One-based physical PPTX slide numbers (tabular mode)")
    fabric.add_argument("--page-labels", nargs="+", help="Optional printed page labels, aligned with --slides")
    fabric.add_argument("--review", type=Path, help="Optional review findings JSON (visual mode only; exports scalar review rows)")
    iq = commands.add_parser("fabric-iq", help="Prepare a snapshot-bound Fabric Data Agent / Foundry IQ blueprint (local only)")
    iq.add_argument("bundle", type=Path, help="Verified visual Fabric export")
    iq.add_argument("--fabric-target", required=True, type=Path)
    iq.add_argument("--foundry-target", required=True, type=Path)
    iq.add_argument("--retrieval-model", type=Path, help="Explicit KB planning-model JSON; separate from prompt model")
    iq.add_argument("--data-agent-id", help="Existing published Fabric Data Agent GUID; omit for an unconfigured blueprint")
    iq.add_argument("--out", required=True, type=Path)
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
                endpoint=args.project_endpoint, model=args.model, destination=args.out, execute=args.execute,
                label_policy=args.label_policy)
        elif args.knowledge_command == "visual":
            from .visual_runtime import extract_visual
            result = extract_visual(args.source, args.out, endpoint=args.project_endpoint, model=args.model,
                page_label=args.page_label, strategy=args.strategy, execute=args.execute)
        elif args.knowledge_command == "dataset":
            from .relation_dataset_store import export_relation_dataset
            result = export_relation_dataset(args.bundles, args.out, profile=args.profile,
                review=json.loads(args.review.read_bytes()) if args.review else None)
        elif args.knowledge_command == "dataset-sqlite":
            from .relation_dataset_store import write_dataset_sqlite
            result = write_dataset_sqlite(args.bundle, args.database)
        elif args.knowledge_command == "dataset-view":
            from .relation_dataset_store import export_relation_view
            result = export_relation_view(args.bundle, args.out, view=args.view)
        elif args.knowledge_command == "dataset-receive":
            from .relation_receiver import receive_relation_views
            result = receive_relation_views(args.search_bundle, args.graph_bundle, args.database,
                collection=args.collection, expected_snapshot=args.expected_snapshot)
        elif args.knowledge_command == "dataset-query":
            from .relation_receiver import search_received_relations
            result = search_received_relations(args.database, collection=args.collection, query=args.query,
                scope=args.scope, assertion_status=args.assertion_status, support=args.support,
                limit=args.limit, offset=args.offset, snapshot_id=args.snapshot)
        elif args.knowledge_command == "dataset-trace":
            from .relation_receiver import trace_received_relation
            result = trace_received_relation(args.database, collection=args.collection,
                card_id=args.card_id, snapshot_id=args.snapshot)
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
        elif args.knowledge_command == "fabric-iq":
            from .fabric import FabricTarget
            from .fabric_iq import FabricIQRetrievalModel, export_fabric_iq
            from .foundry import FoundryTarget
            result = export_fabric_iq(args.bundle, args.out,
                fabric_target=FabricTarget.model_validate_json(args.fabric_target.read_bytes()),
                foundry_target=FoundryTarget.model_validate_json(args.foundry_target.read_bytes()),
                data_agent_id=args.data_agent_id,
                retrieval_model=FabricIQRetrievalModel.model_validate_json(args.retrieval_model.read_bytes())
                if args.retrieval_model else None)
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
            if args.review is not None and args.mode != "visual":
                raise ValueError("Review findings require --mode visual")
            if args.mode == "tabular":
                from .fabric_tabular import export_tabular
                if len(args.bundles) != 1 or args.bundles[0].suffix.lower() != ".pptx":
                    raise ValueError("Tabular mode requires one PPTX and explicit --slides")
                result = export_tabular(args.bundles[0], args.out, target=target,
                                        slides=args.slides, page_labels=args.page_labels)
            else:
                if args.slides is not None or args.page_labels is not None:
                    raise ValueError("Slide selection requires --mode tabular")
                if args.mode == "visual":
                    from .fabric_visual import export_visual
                    result = export_visual(args.bundles, args.out, target=target,
                                           review=json.loads(args.review.read_bytes()) if args.review else None)
                elif args.mode == "narrative":
                    from .fabric_narrative import export_narrative_fabric
                    result = export_narrative_fabric(args.bundles, args.out, target=target)
                else:
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
        from .relation_receiver import ReceiverConflict
        if isinstance(exc, ReceiverConflict):
            print(json.dumps({"status": "failed", "error_type": "ReceiverConflict",
                "message": "Collection changed or source profile differs. Query the current snapshot and use "
                           "--expected-snapshot for an update; use a separate collection for another profile."}))
            return 1
        # Provider messages/URLs can contain credentials or source text.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                          "message": "Operation failed; verify inputs, optional dependencies, and any reserved receipt."}))
        return 1
