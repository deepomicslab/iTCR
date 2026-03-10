"""
Command-line interface for TCR analysis tools
"""
import sys

# Import all main functions from different modules based on actual file structure
from .core.calculate_MCR import main as entropy_mcr_main_func
from .analysis.PLS import main as PLS_analysis_func
from .visualization.mcr_display import main as mcr_display_func
from .visualization.entropy_display import main as entropy_display_func


def entropy_mcr_main():
    """Entry point for entropy and MCR analysis command"""
    entropy_mcr_main_func()

def mcr_display_main():
    """Entry point for MCR display command"""
    mcr_display_func()

def entropy_display_main():
    """Entry point for entropy display command"""
    entropy_display_func()

def PLS_main():
    """Entry point for timepoint analysis command"""
    PLS_analysis_func()


def show_usage():
    """Display usage information"""
    print("iTCR - T-cell Receptor Analysis Toolkit")
    print("=" * 40)
    print("Usage: python -m iTCR [command] [options]")
    print("\nAvailable commands:")
    print("  npmi              - Run NPMI analysis")
    print("  mcr               - Run entropy and MCR analysis")
    print("  mcr-display       - Display MCR results")
    print("  entropy-display   - Display entropy results")
    print("  timepoint         - Run timepoint analysis")
    print("\nExamples:")
    print("  python -m iTCR npmi --input data.pickle --output results/")
    print("  python -m iTCR mcr --input data.pickle --output results/")
    print("  python -m iTCR mcr-display --mcr_path results.pickle --save_dir figures/")
    print("  python -m iTCR entropy-display --entropy_path results.pickle --save_dir figures/")
    print("\nFor detailed help on each command, use:")
    print("  python -m iTCR [command] --help")

def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        show_usage()
        sys.exit(1)
    
    command = sys.argv[1]
    # Remove command from args so subcommands can parse their own arguments
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    command_map = {
        "mcr": entropy_mcr_main,
        "PLS": PLS_main,
        "mcr-display": mcr_display_main,
        "entropy-display": entropy_display_main,
        "help": lambda: show_usage(),
        "--help": lambda: show_usage(),
        "-h": lambda: show_usage(),
    }
    
    if command in command_map:
        try:
            command_map[command]()
        except KeyboardInterrupt:
            print("\n⚠️  Analysis interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error running command '{command}': {e}")
            print("\nFor help, run: python -m iTCR --help")
            sys.exit(1)
    else:
        print(f"❌ Unknown command: {command}")
        print("\nAvailable commands:")
        for cmd in sorted(command_map.keys()):
            if not cmd.startswith('-'):
                print(f"  {cmd}")
        print("\nFor detailed help, run: python -m iTCR --help")
        sys.exit(1)

if __name__ == "__main__":
    main()
