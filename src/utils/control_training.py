import os
import argparse
import sys
import threading

def create_flag_file(flag_dir, flag_name):
    """Creates a flag file."""
    path = os.path.join(flag_dir, flag_name)
    try:
        # Create the file if it doesn't exist
        with open(path, "w") as f:
            f.write("1")
    except Exception as e:
        print(f"Error creating flag file {path}: {e}", file=sys.stderr)

def remove_flag_file(flag_dir, flag_name):
    """Removes a flag file."""
    path = os.path.join(flag_dir, flag_name)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"Error removing flag file {path}: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Robust training controller. Reads commands from stdin.")
    parser.add_argument("--flag-dir", required=True, help="Directory to write flag files.")
    args = parser.parse_args()

    if not os.path.isdir(args.flag_dir):
        print(f"Error: Flag directory not found at '{args.flag_dir}'", file=sys.stderr)
        sys.exit(1)

    print("✅ Robust Controller Active.")
    print("   Enter 'p' to pause, 'r' to resume, 'n' for next, 'q' to quit.")
    print("   Press Enter after each command.")
    print("----------------------------------------------------")

    # This script does NOT need sudo. Run it as the normal user.
    while True:
        try:
            command = input().strip().lower()
            if command == 'p':
                create_flag_file(args.flag_dir, "pause.flag")
                print(">> Pause requested.")
            elif command == 'r':
                remove_flag_file(args.flag_dir, "pause.flag")
                remove_flag_file(args.flag_dir, "next.flag")
                print(">> Resume requested.")
            elif command == 'n':
                create_flag_file(args.flag_dir, "next.flag")
                print(">> Next script requested.")
            elif command == 'q':
                break
        except (EOFError, KeyboardInterrupt):
            break

    print("👋 Controller terminated.")

if __name__ == "__main__":
    main()