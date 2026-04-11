"""Minimal jaxnn CLI."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="jaxnn",
        description="JaxNN command-line interface",
    )
    sub = parser.add_subparsers(dest="cmd")

    # jaxnn list
    p_list = sub.add_parser("list", help="List available models")
    p_list.add_argument(
        "--pretrained", action="store_true", help="Only pretrained models"
    )
    p_list.add_argument(
        "filter", nargs="?", default="*", help="Glob filter (e.g. resnet*)"
    )

    # jaxnn info
    p_info = sub.add_parser("info", help="Show pretrained config for a model")
    p_info.add_argument("model", help="Model name (e.g. resnet34.a1_in1k)")

    args = parser.parse_args()

    if args.cmd == "list":
        import jaxnn

        models = (
            jaxnn.list_pretrained(args.filter)
            if args.pretrained
            else jaxnn.list_models(args.filter)
        )
        for m in models:
            print(m)

    elif args.cmd == "info":
        import jaxnn

        cfg = jaxnn.get_pretrained_cfg(args.model)
        if cfg is None:
            print(f"No config found for '{args.model}'", file=sys.stderr)
            sys.exit(1)
        import json

        print(json.dumps(cfg.to_dict(), indent=2))

    else:
        parser.print_help()
