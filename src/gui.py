import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import sys
import os
from analyzer import GGUFAnalyzer
from renderer import GraphRenderer, format_size

class GGUFVizGUI(tk.Tk):
    """A tkinter-based GUI for the GGUF Visualizer."""

    def __init__(self):
        super().__init__()
        self.title("GGUF Visualizer")
        self.geometry("700x500")
        self.renderer = None # To store the renderer instance after generation

        # Frame for input/output selection
        file_frame = tk.Frame(self, padx=10, pady=10)
        file_frame.pack(fill=tk.X)

        # Input file selection
        tk.Label(file_frame, text="GGUF File:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.input_path = tk.StringVar()
        tk.Entry(file_frame, textvariable=self.input_path, width=60).grid(row=0, column=1, sticky=tk.EW)
        tk.Button(file_frame, text="Browse...", command=self.browse_input_file).grid(row=0, column=2, padx=5)

        file_frame.columnconfigure(1, weight=1)

        # --- Action Buttons ---
        action_frame = tk.Frame(self)
        action_frame.pack(pady=10)

        self.generate_button = tk.Button(action_frame, text="Generate & Display Diagram", command=self.start_generation, font=("Arial", 12, "bold"))
        self.generate_button.pack(side=tk.LEFT, padx=5)

        self.save_button = tk.Button(action_frame, text="Save Diagram As...", command=self.save_diagram, state='disabled')
        self.save_button.pack(side=tk.LEFT, padx=5)

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

    def save_diagram(self):
        """Opens a dialog to save the currently generated diagram."""
        if not self.renderer:
            messagebox.showwarning("Warning", "Please generate a diagram first.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Diagram As",
            filetypes=(("PNG files", "*.png"),),
            defaultextension=".png",
            initialfile=os.path.basename(self.input_path.get()).rsplit('.', 1)[0] + "_diagram.png"
        )

        if save_path:
            try:
                self.log(f"Saving diagram to {save_path}...")
                # Use the stored renderer to save the diagram to the new path
                self.renderer.render(save_path=save_path)
                self.log("Save complete.")
                messagebox.showinfo("Success", f"Diagram saved successfully to\n{save_path}")
            except Exception as e:
                self.log(f"Error during save: {e}")
                messagebox.showerror("Error", f"Could not save the diagram: {e}")

    def start_generation(self):
        """Starts the analysis and rendering process in a separate thread."""
        input_file = self.input_path.get()
        if not input_file:
            messagebox.showerror("Error", "Please select an input GGUF file.")
            return

        self.generate_button.config(state='disabled', text="Generating...")
        self.save_button.config(state='disabled')
        self.renderer = None # Reset previous renderer

        self.log_area.config(state='normal')
        self.log_area.delete('1.0', tk.END)
        self.log_area.config(state='disabled')

        # Run the core logic in a thread to keep the GUI responsive
        thread = threading.Thread(target=self.run_generation, args=(input_file,))
        thread.start()

    def run_generation(self, input_file):
        """The core logic that runs in a separate thread."""
        try:
            self.log(f"Analyzing GGUF file: {input_file}...")
            analyzer = GGUFAnalyzer(input_file)
            analysis_data = analyzer.analyze()
            self.log("Analysis complete.")

            summary = analysis_data['summary']
            summary_text = (
                "\n--- GGUF Model Summary ---\n"
                f"Total Parameters: {summary['total_params'] / 1e9:.2f} B\n"
                f"Total Size:       {format_size(summary['total_size_bytes'])}\n"
                f"Layer Count:      {len(analysis_data['layers'])}\n"
                "--------------------------\n"
            )
            self.log(summary_text)

            self.log("Building and displaying diagram...")

            # Store the renderer instance so we can save it later
            self.renderer = GraphRenderer(analysis_data)

            # Redirect print output to our log for the rendering phase
            original_stdout = sys.stdout
            sys.stdout = self

            # Calling render() without a path will display it
            self.renderer.render()

            sys.stdout = original_stdout # Restore stdout
            self.log("Display complete. You can now save the diagram.")
            self.save_button.config(state='normal') # Enable the save button

        except Exception as e:
            self.log(f"\nAn error occurred: {e}")
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        finally:
            self.generate_button.config(state='normal', text="Generate & Display Diagram")
            sys.stdout = sys.__stdout__ # Ensure stdout is always restored

    def write(self, text):
        """Allows redirecting stdout to the log widget."""
        if text.strip(): # Avoid logging empty lines
            self.log(text.strip())

    def flush(self):
        """Needed for stdout redirection."""
        pass
