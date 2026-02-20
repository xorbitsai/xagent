#!/usr/bin/env python
"""Export OpenAPI schema to file"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xagent.web.app import app


def export_openapi(output_path: str = "openapi.json"):
    """Export OpenAPI schema to JSON file"""
    schema = app.openapi()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    print(f"OpenAPI schema exported to {output_path}")
    print(f"API title: {schema['info']['title']}")
    print(f"API version: {schema['info'].get('version', '1.0.0')}")
    print(f"Total endpoints: {len(schema['paths'])}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export OpenAPI schema")
    parser.add_argument(
        "-o",
        "--output",
        default="openapi.json",
        help="Output file path (default: openapi.json)",
    )
    args = parser.parse_args()

    export_openapi(args.output)
