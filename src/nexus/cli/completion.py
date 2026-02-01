"""
src/cli/completion.py

Shell completion script generation for Nexus CLI.
Supports Bash and Zsh completion with auto-completion for:
- Model names (from registry)
- Dataset names
- File paths
- CLI arguments

Usage:
    nexus --install-completion
    nexus --install-completion --shell bash
    nexus --install-completion --shell zsh
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import logging

logger = logging.getLogger(__name__)


class CompletionGenerator:
    """Generates shell completion scripts for Nexus CLI."""
    
    # Main CLI commands and their arguments
    COMMANDS = {
        'train': {
            'args': ['--model', '--dataset', '--output', '--epochs', '--batch-size', 
                     '--learning-rate', '--device', '--quantize', '--lora', '--resume'],
            'help': 'Train a model on a dataset'
        },
        'eval': {
            'args': ['--model', '--dataset', '--benchmark', '--output', '--device'],
            'help': 'Evaluate a model on benchmarks'
        },
        'infer': {
            'args': ['--model', '--prompt', '--image', '--audio', '--video', 
                     '--output', '--device', '--max-tokens'],
            'help': 'Run inference with a model'
        },
        'convert': {
            'args': ['--model', '--format', '--output', '--quantize'],
            'help': 'Convert model to different formats (GGUF, ONNX, etc.)'
        },
        'download': {
            'args': ['--model', '--dataset', '--output', '--cache-dir'],
            'help': 'Download models or datasets'
        },
        'serve': {
            'args': ['--model', '--host', '--port', '--workers', '--device'],
            'help': 'Start API server'
        },
        'config': {
            'args': ['--get', '--set', '--list', '--validate'],
            'help': 'Manage configuration'
        },
        'cache': {
            'args': ['--clear', '--list', '--size', '--path'],
            'help': 'Manage cache'
        },
        'benchmark': {
            'args': ['--suite', '--model', '--output', '--device'],
            'help': 'Run benchmark suite'
        }
    }
    
    # Global arguments available for all commands
    GLOBAL_ARGS = [
        '--help', '-h',
        '--version', '-v',
        '--verbose',
        '--quiet',
        '--config',
        '--log-level',
        '--install-completion',
        '--show-completion'
    ]
    
    # Device options
    DEVICE_OPTIONS = ['cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 
                      'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7', 'auto']
    
    # Quantization options
    QUANTIZE_OPTIONS = ['none', 'int8', 'int4', 'nf4', 'fp16', 'bf16', 'gptq', 'awq']
    
    # Model formats for conversion
    FORMAT_OPTIONS = ['gguf', 'onnx', 'tensorrt', 'coreml', 'tflite']
    
    # Log levels
    LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    
    def __init__(self):
        """Initialize the completion generator."""
        self.model_registry = self._load_model_registry()
        self.dataset_registry = self._load_dataset_registry()
    
    def _load_model_registry(self) -> List[str]:
        """Load available models from the registry."""
        models = []
        
        # Try to load from nexus_final registry
        try:
            from src.nexus_final.registry import ModelRegistry
            registry = ModelRegistry()
            models.extend(registry.list_models())
        except Exception:
            pass
        
        # Common HuggingFace models used in Nexus
        common_models = [
            # Text models
            'gpt2', 'gpt2-medium', 'gpt2-large',
            'meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-13b-hf',
            'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3-70B',
            'mistralai/Mistral-7B-v0.1', 'mistralai/Mistral-7B-Instruct-v0.1',
            'mistralai/Mixtral-8x7B-v0.1',
            'Qwen/Qwen2-7B', 'Qwen/Qwen2-72B', 'Qwen/Qwen2.5-7B',
            'microsoft/phi-2', 'microsoft/Phi-3-mini-4k-instruct',
            'google/gemma-2b', 'google/gemma-7b', 'google/gemma-2-9b',
            'tiiuae/falcon-7b', 'tiiuae/falcon-40b',
            # Multimodal models
            'llava-hf/llava-1.5-7b-hf', 'llava-hf/llava-v1.6-vicuna-7b-hf',
            'Qwen/Qwen-VL', 'Qwen/Qwen2-VL-7B',
            'openbmb/MiniCPM-Llama3-V-2_5',
            'Salesforce/instructblip-vicuna-7b',
            # Audio models
            'openai/whisper-base', 'openai/whisper-small', 'openai/whisper-large-v3',
            'facebook/wav2vec2-base-960h',
            # Vision encoders
            'openai/clip-vit-base-patch32', 'openai/clip-vit-large-patch14',
            'google/siglip-base-patch16-224', 'google/siglip-large-patch16-256'
        ]
        models.extend(common_models)
        
        return list(set(models))  # Remove duplicates
    
    def _load_dataset_registry(self) -> List[str]:
        """Load available datasets from the registry."""
        datasets = []
        
        # Try to load from dataset configs
        try:
            import yaml
            config_path = Path('config/datasets.yaml')
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                    if 'datasets' in config:
                        datasets.extend(config['datasets'].keys())
        except Exception:
            pass
        
        # Common datasets
        common_datasets = [
            'glue', 'squad', 'squad_v2', 'ms_marco',
            'imdb', 'yelp_polarity', 'ag_news',
            'wikitext', 'wikitext-2', 'wikitext-103',
            'openwebtext', 'c4', 'the_pile',
            'alpaca', 'dolly', 'sharegpt', 'ultrachat',
            'codeparrot/github-code', 'code_search_net',
            'lvkaivot/unnatural-instructions',
            'FLAN', 'natural_instructions',
            'common_voice', 'librispeech',
            'coco', 'visual_genome', 'laion',
            # Nexus-specific
            'general', 'reasoning', 'coding', 'multimodal',
            'audio', 'vision', 'tool_use'
        ]
        datasets.extend(common_datasets)
        
        return list(set(datasets))
    
    def _get_file_completions(self, extension: Optional[str] = None) -> str:
        """Generate file path completion code."""
        base_template = """    # Complete file paths
    if [[ ${{prev}} == --model || ${{prev}} == --dataset || ${{prev}} == --output || ${{prev}} == --config ]]; then
        COMPREPLY=( $(compgen -f -- "${{cur}}") )
        return 0
    fi"""
        
        if extension:
            return f"""    # Complete file paths with {extension} extension
    if [[ ${{prev}} == *"="* ]]; then
        cur="${{cur}}"
        prev="${{prev%%=*}}="
    fi
    
    if [[ ${{prev}} == --model || ${{prev}} == --dataset || ${{prev}} == --output || ${{prev}} == --config ]]; then
        if [[ ${{cur}} == *.yaml || ${{cur}} == *.yml || ${{cur}} == *.json ]]; then
            COMPREPLY=( $(compgen -f -- "${{cur}}") )
        else
            COMPREPLY=( $(compgen -f -X '!*.{yaml,yml,json}' -- "${{cur}}") )
        fi
        return 0
    fi"""
        
        return base_template
    
    def generate_bash_completion(self, prog_name: str = "nexus") -> str:
        """Generate Bash completion script."""
        commands = ' '.join(self.COMMANDS.keys())
        global_args = ' '.join(self.GLOBAL_ARGS)
        
        # Generate command-specific arguments
        command_cases = []
        for cmd, info in self.COMMANDS.items():
            args = ' '.join(info['args'])
            command_cases.append(f'''        {cmd})
            COMPREPLY=( $(compgen -W "{args} {global_args}" -- "${{cur}}") )
            return 0
            ;;''')
        
        command_cases_str = '\n'.join(command_cases)
        
        # Generate model completions
        model_list = ' '.join(self.model_registry[:100])  # Limit to prevent script bloat
        
        # Generate dataset completions
        dataset_list = ' '.join(self.dataset_registry[:50])  # Limit to prevent script bloat
        
        # Generate device options
        device_list = ' '.join(self.DEVICE_OPTIONS)
        
        # Generate quantize options
        quantize_list = ' '.join(self.QUANTIZE_OPTIONS)
        
        # Generate format options
        format_list = ' '.join(self.FORMAT_OPTIONS)
        
        # Generate log levels
        log_level_list = ' '.join(self.LOG_LEVELS)
        
        script = f'''#!/bin/bash
# Bash completion script for {prog_name}
# Auto-generated by Nexus CLI
# Installation: source this file or copy to /etc/bash_completion.d/{prog_name}

_{prog_name}_completion() {{
    local cur prev opts
    COMPREPLY=()
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev="${{COMP_WORDS[COMP_CWORD-1]}}"
    
    # Global options
    local global_opts="{global_args}"
    local commands="{commands}"
    local models="{model_list}"
    local datasets="{dataset_list}"
    local devices="{device_list}"
    local quantize_opts="{quantize_list}"
    local formats="{format_list}"
    local log_levels="{log_level_list}"
    
    # Complete based on previous argument
    case "${{prev}}" in
        --model)
            COMPREPLY=( $(compgen -W "${{models}}" -- "${{cur}}") )
            return 0
            ;;
        --dataset)
            COMPREPLY=( $(compgen -W "${{datasets}}" -- "${{cur}}") )
            return 0
            ;;
        --device)
            COMPREPLY=( $(compgen -W "${{devices}}" -- "${{cur}}") )
            return 0
            ;;
        --quantize|--quantization)
            COMPREPLY=( $(compgen -W "${{quantize_opts}}" -- "${{cur}}") )
            return 0
            ;;
        --format)
            COMPREPLY=( $(compgen -W "${{formats}}" -- "${{cur}}") )
            return 0
            ;;
        --log-level)
            COMPREPLY=( $(compgen -W "${{log_levels}}" -- "${{cur}}") )
            return 0
            ;;
        --shell)
            COMPREPLY=( $(compgen -W "bash zsh fish" -- "${{cur}}") )
            return 0
            ;;
        --output|--config|--cache-dir|--path)
            COMPREPLY=( $(compgen -f -- "${{cur}}") )
            return 0
            ;;
    esac
    
    # Complete subcommands
    local found_command=""
    for ((i=1; i<COMP_CWORD; i++)); do
        if [[ " ${{commands}} " =~ " ${{COMP_WORDS[i]}} " ]]; then
            found_command="${{COMP_WORDS[i]}}"
            break
        fi
    done
    
    if [[ -z ${{found_command}} ]]; then
        # Complete with commands or global options
        if [[ ${{cur}} == -* ]]; then
            COMPREPLY=( $(compgen -W "${{global_opts}}" -- "${{cur}}") )
        else
            COMPREPLY=( $(compgen -W "${{commands}}" -- "${{cur}}") )
        fi
        return 0
    fi
    
    # Complete based on command
    case "${{found_command}}" in
{command_cases_str}
    esac
    
    # Default: complete with files
    COMPREPLY=( $(compgen -f -- "${{cur}}") )
}}

complete -F _{prog_name}_completion {prog_name}
'''
        return script
    
    def generate_zsh_completion(self, prog_name: str = "nexus") -> str:
        """Generate Zsh completion script."""
        # Generate command specifications
        command_specs = []
        for cmd, info in self.COMMANDS.items():
            args_spec = []
            for arg in info['args']:
                if arg in ['--model']:
                    args_spec.append(f"'{arg}[Model name]:model:->models'")
                elif arg in ['--dataset']:
                    args_spec.append(f"'{arg}[Dataset name]:dataset:->datasets'")
                elif arg in ['--device']:
                    args_spec.append(f"'{arg}[Device to use]:device:->devices'")
                elif arg in ['--quantize', '--quantization']:
                    args_spec.append(f"'{arg}[Quantization type]:quant:->quantize'")
                elif arg in ['--format']:
                    args_spec.append(f"'{arg}[Output format]:format:->formats'")
                elif arg in ['--output', '--cache-dir', '--path']:
                    args_spec.append(f"'{arg}[Path]:path:_files'")
                elif arg in ['--config']:
                    args_spec.append(f"'{arg}[Config file]:config:_files -g \"*.{{yaml,yml,json}}\"'")
                elif arg in ['--host']:
                    args_spec.append(f"'{arg}[Host address]:host:'")
                elif arg in ['--port']:
                    args_spec.append(f"'{arg}[Port number]:port:'")
                elif arg in ['--epochs', '--batch-size', '--max-tokens', '--workers']:
                    args_spec.append(f"'{arg}[Integer value]:int:'")
                elif arg in ['--learning-rate']:
                    args_spec.append(f"'{arg}[Learning rate]:float:'")
                else:
                    args_spec.append(f"'{arg}[{arg}]:value:'")
            
            args_str = ' '.join(args_spec)
            command_specs.append(f'''        '{cmd}:{info['help']}":{args_str}"')
''')
        
        command_specs_str = ''.join(command_specs)
        
        # Generate model list for zsh
        model_list = ' '.join(f'"{m}"' for m in self.model_registry[:100])
        
        # Generate dataset list for zsh
        dataset_list = ' '.join(f'"{d}"' for d in self.dataset_registry[:50])
        
        script = f'''#compdef {prog_name}
# Zsh completion script for {prog_name}
# Auto-generated by Nexus CLI
# Installation: Copy to $fpath directory or source this file

_{prog_name}() {{
    local curcontext="$curcontext" state line
    typeset -A opt_args
    
    local -a commands
    commands=(
{command_specs_str}
    )
    
    _arguments -C \\
        '(-h --help)'{{-h,--help}}'[Show help message]' \\
        '(-v --version)'{{-v,--version}}'[Show version]' \\
        '--verbose[Enable verbose output]' \\
        '--quiet[Suppress output]' \\
        '--config[Configuration file]:config:_files -g "*.{{yaml,yml,json}}"' \\
        '--log-level[Log level]:level:(DEBUG INFO WARNING ERROR CRITICAL)' \\
        '--install-completion[Install shell completion]' \\
        '--show-completion[Show completion script]' \\
        '--shell[Shell type]:shell:(bash zsh fish)' \\
        '1: :->command' \\
        '*:: :->args'
    
    case "$state" in
        command)
            _describe -t commands '{prog_name} commands' commands
            ;;
        args)
            case "$line[1]" in
                *)
                    # Handle dynamic completions
                    ;;
            esac
            ;;
    esac
}}

# Dynamic completion functions
_{prog_name}_models() {{
    local -a models
    models=({model_list})
    _describe -t models 'available models' models
}}

_{prog_name}_datasets() {{
    local -a datasets
    datasets=({dataset_list})
    _describe -t datasets 'available datasets' datasets
}}

_{prog_name}_devices() {{
    local -a devices
    devices=('cpu' 'cuda' 'cuda:0' 'cuda:1' 'cuda:2' 'cuda:3' 'auto')
    _describe -t devices 'compute devices' devices
}}

# Complete the completion function
_{prog_name} "$@"
'''
        return script
    
    def generate_fish_completion(self, prog_name: str = "nexus") -> str:
        """Generate Fish completion script."""
        lines = [
            f"# Fish completion script for {prog_name}",
            f"# Auto-generated by Nexus CLI",
            ""
        ]
        
        # Global options
        for arg in self.GLOBAL_ARGS:
            if arg.startswith('--'):
                lines.append(f"complete -c {prog_name} -l {arg.lstrip('-')} -f")
        
        # Commands
        for cmd, info in self.COMMANDS.items():
            lines.append(f"complete -c {prog_name} -n '__fish_use_subcommand' -a '{cmd}' -d '{info['help']}'")
            
            for arg in info['args']:
                if arg.startswith('--'):
                    arg_name = arg.lstrip('-')
                    if arg in ['--model']:
                        models = ' '.join(self.model_registry[:50])
                        lines.append(f"complete -c {prog_name} -n '__fish_seen_subcommand_from {cmd}' -l {arg_name} -a '{models}'")
                    elif arg in ['--dataset']:
                        datasets = ' '.join(self.dataset_registry[:30])
                        lines.append(f"complete -c {prog_name} -n '__fish_seen_subcommand_from {cmd}' -l {arg_name} -a '{datasets}'")
                    elif arg in ['--device']:
                        devices = ' '.join(self.DEVICE_OPTIONS)
                        lines.append(f"complete -c {prog_name} -n '__fish_seen_subcommand_from {cmd}' -l {arg_name} -a '{devices}'")
                    elif arg in ['--quantize', '--quantization']:
                        quant_opts = ' '.join(self.QUANTIZE_OPTIONS)
                        lines.append(f"complete -c {prog_name} -n '__fish_seen_subcommand_from {cmd}' -l {arg_name} -a '{quant_opts}'")
                    elif arg in ['--output', '--config', '--cache-dir', '--path']:
                        lines.append(f"complete -c {prog_name} -n '__fish_seen_subcommand_from {cmd}' -l {arg_name} -F")
                    else:
                        lines.append(f"complete -c {prog_name} -n '__fish_seen_subcommand_from {cmd}' -l {arg_name}")
        
        return '\n'.join(lines)
    
    def get_completion_script(self, shell: str = 'bash', prog_name: str = "nexus") -> str:
        """Get completion script for the specified shell."""
        shell = shell.lower()
        
        if shell == 'bash':
            return self.generate_bash_completion(prog_name)
        elif shell == 'zsh':
            return self.generate_zsh_completion(prog_name)
        elif shell == 'fish':
            return self.generate_fish_completion(prog_name)
        else:
            raise ValueError(f"Unsupported shell: {shell}. Supported: bash, zsh, fish")
    
    def install_completion(self, shell: Optional[str] = None, 
                          prog_name: str = "nexus") -> bool:
        """Install completion script for the current shell."""
        if shell is None:
            # Auto-detect shell
            shell = os.path.basename(os.environ.get('SHELL', 'bash'))
        
        shell = shell.lower()
        script = self.get_completion_script(shell, prog_name)
        
        # Determine installation path
        home = Path.home()
        
        if shell == 'bash':
            # Try common bash completion directories
            completion_dirs = [
                home / '.bash_completion.d',
                home / '.local' / 'share' / 'bash-completion' / 'completions',
                Path('/etc/bash_completion.d'),
                Path('/usr/share/bash-completion/completions'),
            ]
            
            # Fall back to .bashrc
            bashrc = home / '.bashrc'
            completion_file = home / f'.{prog_name}_completion'
            
            for comp_dir in completion_dirs:
                if comp_dir.parent.exists() or comp_dir.exists():
                    try:
                        comp_dir.mkdir(parents=True, exist_ok=True)
                        completion_path = comp_dir / prog_name
                        with open(completion_path, 'w') as f:
                            f.write(script)
                        print(f"✓ Bash completion installed to: {completion_path}")
                        print(f"  Run 'source {completion_path}' or restart your shell")
                        return True
                    except PermissionError:
                        continue
            
            # Fall back to sourcing in .bashrc
            with open(completion_file, 'w') as f:
                f.write(script)
            
            source_line = f"\n# Nexus CLI completion\nsource {completion_file}\n"
            
            try:
                with open(bashrc, 'r') as f:
                    content = f.read()
                
                if str(completion_file) not in content:
                    with open(bashrc, 'a') as f:
                        f.write(source_line)
                
                print(f"✓ Bash completion installed to: {completion_file}")
                print(f"  Sourced from: {bashrc}")
                print(f"  Run 'source {bashrc}' or restart your shell")
                return True
            except Exception as e:
                print(f"✗ Failed to install bash completion: {e}")
                return False
        
        elif shell == 'zsh':
            # Try zsh completion directories
            completion_dirs = [
                home / '.zsh' / 'completions',
                home / '.local' / 'share' / 'zsh' / 'site-functions',
            ]
            
            # Check fpath
            fpath_dirs = os.environ.get('FPATH', '').split(':')
            for fp_dir in fpath_dirs:
                if 'site-functions' in fp_dir or 'completions' in fp_dir:
                    completion_dirs.append(Path(fp_dir))
            
            completion_file = home / f'.{prog_name}_completion'
            zshrc = home / '.zshrc'
            
            for comp_dir in completion_dirs:
                if comp_dir.parent.exists() or comp_dir.exists():
                    try:
                        comp_dir.mkdir(parents=True, exist_ok=True)
                        completion_path = comp_dir / f'_{prog_name}'
                        with open(completion_path, 'w') as f:
                            f.write(script)
                        print(f"✓ Zsh completion installed to: {completion_path}")
                        print(f"  Run 'source {completion_path}' or restart your shell")
                        return True
                    except PermissionError:
                        continue
            
            # Fall back to sourcing in .zshrc
            with open(completion_file, 'w') as f:
                f.write(script)
            
            source_line = f"\n# Nexus CLI completion\nfpath+=({completion_file.parent})\nsource {completion_file}\n"
            
            try:
                with open(zshrc, 'r') as f:
                    content = f.read()
                
                if str(completion_file) not in content:
                    with open(zshrc, 'a') as f:
                        f.write(source_line)
                
                print(f"✓ Zsh completion installed to: {completion_file}")
                print(f"  Config added to: {zshrc}")
                print(f"  Run 'source {zshrc}' or restart your shell")
                return True
            except Exception as e:
                print(f"✗ Failed to install zsh completion: {e}")
                return False
        
        elif shell == 'fish':
            fish_dir = home / '.config' / 'fish' / 'completions'
            try:
                fish_dir.mkdir(parents=True, exist_ok=True)
                completion_path = fish_dir / f'{prog_name}.fish'
                with open(completion_path, 'w') as f:
                    f.write(script)
                print(f"✓ Fish completion installed to: {completion_path}")
                print(f"  Restart your shell or run 'source {completion_path}'")
                return True
            except Exception as e:
                print(f"✗ Failed to install fish completion: {e}")
                return False
        
        else:
            print(f"✗ Unsupported shell: {shell}")
            return False


def show_completion_script(shell: str = 'bash', prog_name: str = "nexus"):
    """Show the completion script for manual installation."""
    generator = CompletionGenerator()
    script = generator.get_completion_script(shell, prog_name)
    print(script)


def install_completion(shell: Optional[str] = None, prog_name: str = "nexus") -> bool:
    """Install shell completion for Nexus CLI."""
    generator = CompletionGenerator()
    return generator.install_completion(shell, prog_name)


def main():
    """CLI entry point for completion management."""
    parser = argparse.ArgumentParser(
        description="Manage shell completion for Nexus CLI"
    )
    parser.add_argument(
        '--shell',
        choices=['bash', 'zsh', 'fish', 'auto'],
        default='auto',
        help='Shell type (default: auto-detect)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show completion script instead of installing'
    )
    parser.add_argument(
        '--install',
        action='store_true',
        help='Install completion script'
    )
    parser.add_argument(
        '--prog-name',
        default='nexus',
        help='Program name for completion (default: nexus)'
    )
    
    args = parser.parse_args()
    
    shell = None if args.shell == 'auto' else args.shell
    
    if args.show:
        show_completion_script(args.shell if args.shell != 'auto' else 'bash', args.prog_name)
    elif args.install:
        success = install_completion(shell, args.prog_name)
        exit(0 if success else 1)
    else:
        # Default: show help
        parser.print_help()
        print("\nExamples:")
        print(f"  {args.prog_name} --install-completion --shell bash")
        print(f"  {args.prog_name} --install-completion --shell zsh")
        print(f"  {args.prog_name} --show-completion --shell bash")


if __name__ == "__main__":
    main()
