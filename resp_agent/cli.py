"""
Resp-Agent Command Line Interface

This module provides the CLI entry point for the resp-agent package.
"""

import argparse
import sys


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog="resp-agent",
        description="Resp-Agent: A multi-agent framework for respiratory sound diagnosis and generation",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Diagnose command
    diagnose_parser = subparsers.add_parser(
        "diagnose", help="Run respiratory sound diagnosis"
    )
    diagnose_parser.add_argument(
        "--audio_dir", type=str, required=True, help="Directory containing audio files"
    )
    diagnose_parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for results"
    )
    diagnose_parser.add_argument(
        "--metadata_csv", type=str, required=True, help="Path to metadata CSV file"
    )
    diagnose_parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration file"
    )
    diagnose_parser.add_argument(
        "--device", type=str, default="cuda:0", help="Computation device"
    )

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate", help="Generate respiratory sounds"
    )
    generate_parser.add_argument(
        "--ref_audio", type=str, required=True, help="Path to reference audio file"
    )
    generate_parser.add_argument(
        "--disease", type=str, required=True, help="Disease type for generation"
    )
    generate_parser.add_argument(
        "--out_dir", type=str, required=True, help="Output directory"
    )
    generate_parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration file"
    )
    generate_parser.add_argument(
        "--device", type=str, default="cuda:0", help="Computation device"
    )

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat agent")
    chat_parser.add_argument(
        "--lang", type=str, choices=["zh", "en"], default="zh", help="Language (zh/en)"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "diagnose":
        from .diagnoser import run_diagnoser

        result = run_diagnoser(
            audio_dir=args.audio_dir,
            output_dir=args.output_dir,
            metadata_csv=args.metadata_csv,
            config_path=args.config,
            device=args.device,
        )
        print(f"Diagnosis complete. Results saved to: {result}")

    elif args.command == "generate":
        from .generator import run_generator

        result = run_generator(
            ref_audio=args.ref_audio,
            disease=args.disease,
            out_dir=args.out_dir,
            config_path=args.config,
            device=args.device,
        )
        print(f"Generation complete. Audio saved to: {result}")

    elif args.command == "chat":
        if args.lang == "zh":
            from .agent.chinese import main as chat_main
        else:
            from .agent.english import main as chat_main
        chat_main()


if __name__ == "__main__":
    main()
