import graphviz
from typing import Dict, Any, Optional

def format_size(size_bytes: int) -> str:
    """Formats size in bytes to KB, MB, or GB with increased precision."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/1024**2:.2f} MB"
    else:
        return f"{size_bytes/1024**3:.2f} GB"

class GraphRenderer:
    """Renders the GGUF analysis data into a visual graph."""

    def __init__(self, analysis_data: Dict[str, Any]):
        self.data = analysis_data
        self.dot = graphviz.Digraph(comment='GGUF Model Architecture')
        self.dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2')
        self.dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
        self.dot.attr('edge', color='gray40')

    def _add_tensor_node(self, tensor: Dict[str, Any], parent_graph):
        """Adds a single tensor node to the specified graph."""
        name = tensor['name']
        shape = 'x'.join(map(str, tensor['shape']))
        size = format_size(tensor['size_bytes'])
        label = f"{{ {name} | {{Shape: {shape} | Size: {size}}} }}"
        parent_graph.node(name, label=label, shape='record')

    def _render_layer_0_detail(self):
        """Renders the first layer in detail as a single vertical column."""
        if not self.data['layers']:
            return

        layer_0 = self.data['layers'][0]
        layer_id = f"layer_{layer_0['index']}"

        with self.dot.subgraph(name=f"cluster_{layer_id}") as c:
            c.attr(label=f"Layer {layer_0['index']} (Detailed View)\\nSize: {format_size(layer_0['size'])}", style='filled', fillcolor='lightgrey')

            all_layer_tensors = layer_0['tensors']

            # Add all tensor nodes to the subgraph
            for tensor in all_layer_tensors:
                self._add_tensor_node(tensor, c)

            # Create invisible edges between consecutive tensors to enforce a strict vertical order
            if len(all_layer_tensors) > 1:
                for i in range(len(all_layer_tensors) - 1):
                    # Connect the current tensor to the next one
                    c.edge(all_layer_tensors[i]['name'], all_layer_tensors[i+1]['name'], style='invis')

    def _render_layer_stack_summary(self):
        """Renders a summary node for the stack of remaining layers with detailed sizes and shapes."""
        if len(self.data['layers']) <= 1:
            return

        remaining_layers = self.data['layers'][1:]
        num_layers = len(remaining_layers)
        total_size = sum(layer['size'] for layer in remaining_layers)

        from collections import Counter

        # --- Structure Signature Analysis ---
        first_sig = remaining_layers[0].get('structure_signature')
        all_same_structure = all(layer.get('structure_signature') == first_sig for layer in remaining_layers)

        details_label = ""
        if all_same_structure and first_sig:
            # All layers are the same, show a consolidated summary
            # The signature is (internal_name, shape, type). We only want the shape.
            shapes = [s[1] for s in first_sig]
            shape_counts = Counter(shapes)
            shape_summary = [f"{count}x {' x '.join(map(str, shape))}" for shape, count in sorted(shape_counts.items())]

            size_summary = [f"Layer {layer['index']}: {format_size(layer['size'])}" for layer in remaining_layers]

            details_label = (
                f"--- Common Tensor Shapes ---\\n"
                f"{'\\n'.join(shape_summary)}\\n"
                f"--- Individual Layer Sizes ---\\n"
                f"{'\\n'.join(size_summary)}\\n"
            )
        else:
            # Layers differ, show a detailed tensor breakdown for each.
            # This provides full transparency and avoids confusing summaries.
            layer_details = []
            for layer in remaining_layers:
                # Use left-alignment for clean, readable blocks of text.
                size_info = f"Layer {layer['index']}: {format_size(layer['size'])}\\l"
                tensor_breakdown = [size_info]
                # Sort tensors by name for consistent ordering.
                for tensor in sorted(layer['tensors'], key=lambda t: t['name']):
                    internal_name = tensor['name'].split('.', 2)[-1]
                    shape_str = 'x'.join(map(str, tensor['shape']))
                    size_str = format_size(tensor['size_bytes'])
                    tensor_breakdown.append(f"  • {internal_name}: [{shape_str}] ({size_str})\\l")
                layer_details.append('\\n'.join(tensor_breakdown))

            details_label = (
                f"--- Layer Details (Structures Vary) ---\\n"
                f"{'\\n\\n'.join(layer_details)}\\n" # Separate layers with a blank line for readability.
            )

        label = (
            f"Stack of {num_layers} Layers\\n"
            f"{details_label}"
            f"------------------------------\\n"
            f"Total Stack Size: {format_size(total_size)}"
        )
        self.dot.node('layer_stack_summary', label=label, shape='box3d', fillcolor='moccasin', fontsize='10')

    def _build_graph(self):
        """Constructs the graphviz graph object from the analysis data."""
        # Pre-computation Globals
        with self.dot.subgraph(name="cluster_globals_pre") as c:
            c.attr(label="Input Tensors", style='filled', fillcolor='whitesmoke')
            for tensor in self.data['globals_pre']:
                self._add_tensor_node(tensor, c)

        # Detailed Layer 0
        self._render_layer_0_detail()

        # Summary of Layer Stack
        self._render_layer_stack_summary()

        # Post-computation Globals
        with self.dot.subgraph(name="cluster_globals_post") as c:
            c.attr(label="Output Tensors", style='filled', fillcolor='whitesmoke')
            for tensor in self.data['globals_post']:
                self._add_tensor_node(tensor, c)

        # --- Connect the main components ---
        if not self.data['layers']:
            return

        layer_0_cluster_name = f"cluster_layer_{self.data['layers'][0]['index']}"

        if self.data['globals_pre']:
            self.dot.edge(
                self.data['globals_pre'][-1]['name'],
                self.data['layers'][0]['tensors'][0]['name'],
                lhead=layer_0_cluster_name
            )

        if len(self.data['layers']) > 1:
            self.dot.edge(
                self.data['layers'][0]['tensors'][-1]['name'],
                'layer_stack_summary',
                ltail=layer_0_cluster_name
            )
            if self.data['globals_post']:
                self.dot.edge(
                    'layer_stack_summary',
                    self.data['globals_post'][0]['name'],
                    lhead='cluster_globals_post'
                )
        elif self.data['globals_post']:
            self.dot.edge(
                self.data['layers'][0]['tensors'][-1]['name'],
                self.data['globals_post'][0]['name'],
                ltail=layer_0_cluster_name,
                lhead='cluster_globals_post'
            )

    def render(self, save_path: Optional[str] = None):
        """
        Generates the diagram. If save_path is provided, it saves to a file.
        Otherwise, it displays the diagram in the default image viewer.
        """
        self._build_graph()
        try:
            if save_path:
                # The render method adds the extension, so we pass the path without it
                # to avoid a double extension (e.g., 'model.png.png').
                path_without_ext = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
                output_filename = self.dot.render(path_without_ext, format='png', view=False, cleanup=True)
                print(f"Diagram saved to {output_filename}")
            else:
                # Render to a temporary file and display
                print("Displaying diagram...")
                self.dot.render(format='png', view=True, cleanup=True)
        except Exception as e:
            print(f"Error rendering graph: {e}")
            print("Please ensure Graphviz is installed and in your system's PATH.")
            raise
