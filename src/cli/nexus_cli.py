#!/usr/bin/env python3
"""
nexus CLI

Main entry point for the Nexus CLI with integrated polish features.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Optional, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main(args: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog='nexus',
        description='Nexus - Universal Model Training and Inference Platform'
    )
    
    parser.add_argument('--version', '-v', action='version', version='%(prog)s 1.0.0')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--install-completion', action='store_true', help='Install shell completion')
    parser.add_argument('--shell', choices=['bash', 'zsh', 'fish', 'auto'], default='auto')
    
    args = parser.parse_args(args)
    
    if args.install_completion:
        from src.cli.completion import install_completion
        shell = None if args.shell == 'auto' else args.shell
        success = install_completion(shell, prog_name='nexus')
        return 0 if success else 1
    
    print("Nexus CLI - Use --help for available commands")
    return 0

if __name__ == '__main__':
    sys.exit(main())
