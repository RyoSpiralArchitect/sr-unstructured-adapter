"""One bounded fixed-Luna question run; no network without --execute."""

import argparse
import json
import os
from pathlib import Path

import httpx

from sr_adapter.ontology_question import answer_ontology_question, question_response
from sr_adapter.tables import canonical_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--collection", action="append", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "model": "gpt-6-luna",
                    "max_model_calls": args.max_rounds,
                    "model_calls": 0,
                }
            )
        )
        return
    count = 0

    def complete(request):
        nonlocal count
        current = count
        count += 1
        wire = {
            "model": "gpt-6-luna",
            "store": False,
            "reasoning": {"effort": "medium"},
            "max_output_tokens": 6000,
            "text": {"format": {"type": "json_object"}},
            "input": [
                {
                    "role": "system",
                    "content": request["guidance"]
                    + " Respond with a JSON object containing actions and answer. arguments is a JSON object. Final citations use collection,name,snapshot_id,kind,id.",
                },
                {"role": "user", "content": canonical_json(request).decode()},
            ],
        }
        (args.out / f"wire-request-{current:02d}.json").write_bytes(
            canonical_json(wire)
        )
        with (args.out / f"submitted-{current:02d}").open("x") as marker:
            marker.write("gpt-6-luna\n")
        with httpx.Client(timeout=300, follow_redirects=False) as client:
            with client.stream(
                "POST",
                "https://api.openai.com/v1/responses",
                content=canonical_json(wire),
                headers={
                    "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                    "Content-Type": "application/json",
                },
            ) as response:
                body = bytearray()
                for chunk in response.iter_bytes(65536):
                    body.extend(chunk)
                    if len(body) > 8 * 1024 * 1024:
                        (args.out / f"wire-response-{current:02d}.partial").write_bytes(
                            body
                        )
                        raise ValueError("Response limit")
                (args.out / f"wire-response-{current:02d}.json").write_bytes(body)
                response.raise_for_status()
        return question_response(json.loads(body))

    result = answer_ontology_question(
        args.database,
        collections=tuple(args.collection),
        question=args.question,
        complete=complete,
        destination=args.out,
        max_rounds=args.max_rounds,
        max_inspections=12,
        max_tool_calls=24,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
