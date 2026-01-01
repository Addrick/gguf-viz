import argparse
from analyzer import GGUFAnalyzer
from renderer import GraphRenderer, format_size

def main():
    """Main function to run the GGUF visualization CLI."""
    parser = argparse.ArgumentParser(
        description="Create a visual architecture diagram from a GGUF file."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the GGUF model file."
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default="model_diagram",
        help="Path to the output PNG file (without extension). Default: model_diagram"
    )

    args = parser.parse_args()

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

if __name__ == "__main__":
    main()
