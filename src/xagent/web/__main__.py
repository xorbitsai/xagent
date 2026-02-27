#!/usr/bin/env python3
"""Main entry point for the xagent Web module

Usage:
    python -m xagent.web
    python -m xagent.web --host 0.0.0.0 --port 8000
    python -m xagent.web --reload --debug
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


JWT_ENV_KEYS = (
    "XAGENT_JWT_SECRET",
    "XAGENT_JWT_ALGORITHM",
    "XAGENT_ACCESS_TOKEN_EXPIRE_MINUTES",
    "XAGENT_REFRESH_TOKEN_EXPIRE_DAYS",
    "XAGENT_PASSWORD_MIN_LENGTH",
)


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _load_example_env_values() -> dict[str, str]:
    example_env_path = Path(__file__).resolve().parents[3] / "example.env"
    if not example_env_path.exists():
        return {}

    values: dict[str, str] = {}
    for line in example_env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if key not in JWT_ENV_KEYS:
            continue

        values[key] = _strip_wrapping_quotes(raw_value.strip())

    return values


def warn_if_example_jwt_config(logger: logging.Logger) -> None:
    example_values = _load_example_env_values()
    if not example_values:
        return

    matched_keys = [
        key
        for key, example_value in example_values.items()
        if os.getenv(key) == example_value
    ]

    if not matched_keys:
        return

    logger.warning(
        "⚠️ JWT-related environment variables are still using example defaults: %s. Please update your .env for production.",
        ", ".join(matched_keys),
    )


def setup_logging(debug: bool = False) -> None:
    """Configure logging"""
    level = logging.DEBUG if debug else logging.INFO

    # Basic logging configuration
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if debug:
        # Debug mode: configure detailed logging for specific modules
        # XAgent modules - set to DEBUG for all submodules
        logging.getLogger("xagent.core").setLevel(logging.DEBUG)
        logging.getLogger("xagent.web").setLevel(logging.DEBUG)
        logging.getLogger("xagent.entrypoint").setLevel(logging.DEBUG)
        logging.getLogger("xagent.skills").setLevel(logging.DEBUG)

        # LangChain/LangGraph detailed logging
        logging.getLogger("langchain").setLevel(logging.DEBUG)
        logging.getLogger("langgraph").setLevel(logging.DEBUG)
        logging.getLogger("langchain_core").setLevel(logging.DEBUG)

        # Suppress verbose logs from third-party libraries
        logging.getLogger("aiohttp").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

        print("🐛 Debug mode enabled")
        print("🔍 LLM responses and tool call details will be logged")
        print("-" * 50)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Start xagent Web service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m xagent.web                     # Start with default configuration
    python -m xagent.web --port 8001         # Specify port
    python -m xagent.web --reload --debug    # Development mode + debug mode
    python -m xagent.web --host 0.0.0.0      # Listen on all interfaces
    python -m xagent.web --debug             # Enable verbose logging (LLM responses, etc.)
        """,
    )

    parser.add_argument(
        "--host", default="127.0.0.1", help="Server host address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload (development mode)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (verbose logging, including LLM responses)",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Log level (default: info)",
    )

    return parser.parse_args()


def main() -> None:
    """Main function"""
    args = parse_args()

    # Configure logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    warn_if_example_jwt_config(logger)

    logger.info("🚀 Starting xagent Web service...")
    logger.info(f"📍 Service URL: http://{args.host}:{args.port}")

    if args.reload:
        logger.info("🔄 Development mode: auto-reload enabled")

    if args.debug:
        logger.info("🐛 Debug mode: verbose logging enabled")

    try:
        uvicorn.run(
            "xagent.web.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
        )
    except KeyboardInterrupt:
        logger.info("⏹️  Service stopped")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
