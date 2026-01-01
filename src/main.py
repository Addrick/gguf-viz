import argparse
import sys
from analyzer import GGUFAnalyzer
from renderer import GraphRenderer, format_size

def run_cli(args):
    """The command-line execution logic."""
    try:
        # 1. Analyze the GGUF file
        print(f"Analyzing GGUF file: {args.input_file}...")
        analyzer = GGUFAnalyzer(args.input_file)
        analysis_data = analyzer.analyze()
        print("Analysis complete.")

        # 2. Print summary to console
        summary = analysis_data['summary']
        total_params = summary['total_params']
        total_size_bytes = summary['total_size_bytes']
        layer_count = len(analysis_data['layers'])

        print("\n--- GGUF Model Summary ---")
        print(f"Total Parameters: {total_params / 1e9:.2f} B")
        print(f"Total Size:       {format_size(total_size_bytes)}")
        print(f"Layer Count:      {layer_count}")
        print("--------------------------\n")

        # 3. Render the diagram
        print(f"Rendering diagram to {args.output_file}.png...")
        renderer = GraphRenderer(analysis_data)
        renderer.render(args.output_file)

    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_file}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    """Main function to run the GGUF visualization CLI or GUI."""
    parser = argparse.ArgumentParser(
        description="Create a visual architecture diagram from a GGUF file."
    )
    parser.add_argument(
        "input_file",
        type=str,
        nargs='?',  # Make positional argument optional for GUI mode
        default=None,
        help="Path to the GGUF model file (required for CLI mode)."
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default="model_diagram",
        help="Path to the output PNG file (without extension). Default: model_diagram"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the graphical user interface."
    )

    args = parser.parse_args()

    if args.gui:
        # Dynamically import GUI to avoid tkinter dependency for CLI users
        try:
            from gui import GGUFVizGUI
            app = GGUFVizGUI()
            app.mainloop()
        except ImportError:
            print("Could not import the GUI module. Please ensure tkinter is installed.")
            sys.exit(1)
    else:
        if not args.input_file:
            parser.error("the following arguments are required in CLI mode: input_file")
        run_cli(args)

if __name__ == "__main__":
    main()
