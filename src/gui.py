import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import sys
from analyzer import GGUFAnalyzer
from renderer import GraphRenderer, format_size

class GGUFVizGUI(tk.Tk):
    """A tkinter-based GUI for the GGUF Visualizer."""

    def __init__(self):
        super().__init__()
        self.title("GGUF Visualizer")
        self.geometry("700x500")

        # Frame for input/output selection
        file_frame = tk.Frame(self, padx=10, pady=10)
        file_frame.pack(fill=tk.X)

        # Input file selection
        tk.Label(file_frame, text="GGUF File:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.input_path = tk.StringVar()
        tk.Entry(file_frame, textvariable=self.input_path, width=60).grid(row=0, column=1, sticky=tk.EW)
        tk.Button(file_frame, text="Browse...", command=self.browse_input_file).grid(row=0, column=2, padx=5)

        # Output file selection
        tk.Label(file_frame, text="Output PNG:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.output_path = tk.StringVar()
        tk.Entry(file_frame, textvariable=self.output_path, width=60).grid(row=1, column=1, sticky=tk.EW)
        tk.Button(file_frame, text="Save As...", command=self.browse_output_file).grid(row=1, column=2, padx=5)

        file_frame.columnconfigure(1, weight=1)

        # Generate button
        self.generate_button = tk.Button(self, text="Generate Diagram", command=self.start_generation, font=("Arial", 12, "bold"))
        self.generate_button.pack(pady=10)

        # Status/log text area
        self.log_area = scrolledtext.ScrolledText(self, wrap=tk.WORD, height=20, state='disabled')
        self.log_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    def log(self, message):
        """Appends a message to the log area."""
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.config(state='disabled')
        self.log_area.see(tk.END)
        self.update_idletasks()

    def browse_input_file(self):
        """Opens a dialog to select a GGUF file."""
        filepath = filedialog.askopenfilename(
            title="Select a GGUF File",
            filetypes=(("GGUF files", "*.gguf"), ("All files", "*.*"))
        )
        if filepath:
            self.input_path.set(filepath)
            # Suggest a default output path based on the input
            output_suggestion = filepath.rsplit('.', 1)[0] + "_diagram"
            self.output_path.set(output_suggestion)

    def browse_output_file(self):
        """Opens a dialog to select a location to save the PNG file."""
        filepath = filedialog.asksaveasfilename(
            title="Save Diagram As",
            filetypes=(("PNG files", "*.png"),),
            defaultextension=".png"
        )
        if filepath:
            # The dialog adds the extension, but our renderer adds it too.
            # So, we remove it here to avoid a double extension.
            self.output_path.set(filepath.rsplit('.', 1)[0])

    def start_generation(self):
        """Starts the analysis and rendering process in a separate thread."""
        input_file = self.input_path.get()
        output_file = self.output_path.get()

        if not input_file or not output_file:
            messagebox.showerror("Error", "Please select both an input and output file.")
            return

        self.generate_button.config(state='disabled', text="Generating...")
        self.log_area.config(state='normal')
        self.log_area.delete('1.0', tk.END)
        self.log_area.config(state='disabled')

        # Run the core logic in a thread to keep the GUI responsive
        thread = threading.Thread(target=self.run_generation, args=(input_file, output_file))
        thread.start()

    def run_generation(self, input_file, output_file):
        """The core logic that runs in a separate thread."""
        try:
            self.log(f"Analyzing GGUF file: {input_file}...")
            analyzer = GGUFAnalyzer(input_file)
            analysis_data = analyzer.analyze()
            self.log("Analysis complete.")

            summary = analysis_data['summary']
            total_params = summary['total_params']
            total_size_bytes = summary['total_size_bytes']
            layer_count = len(analysis_data['layers'])

            summary_text = (
                "\n--- GGUF Model Summary ---\n"
                f"Total Parameters: {total_params / 1e9:.2f} B\n"
                f"Total Size:       {format_size(total_size_bytes)}\n"
                f"Layer Count:      {layer_count}\n"
                "--------------------------\n"
            )
            self.log(summary_text)

            self.log(f"Rendering diagram to {output_file}.png...")
            renderer = GraphRenderer(analysis_data)

            # Redirect renderer's print output to our log
            original_stdout = sys.stdout
            sys.stdout = self

            renderer.render(output_file)

            sys.stdout = original_stdout # Restore stdout
            self.log("Done.")
            messagebox.showinfo("Success", f"Diagram saved successfully to {output_file}.png")

        except Exception as e:
            self.log(f"\nAn error occurred: {e}")
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        finally:
            self.generate_button.config(state='normal', text="Generate Diagram")
            sys.stdout = sys.__stdout__ # Ensure stdout is always restored

    def write(self, text):
        """Allows redirecting stdout to the log widget."""
        if text.strip(): # Avoid logging empty lines
            self.log(text.strip())

    def flush(self):
        """Needed for stdout redirection."""
        pass
