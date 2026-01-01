import graphviz
from typing import Dict, Any

def format_size(size_bytes: int) -> str:
    """Formats size in bytes to KB, MB, or GB with increased precision."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/1024**2:.4f} MB"
    else:
        return f"{size_bytes/1024**3:.4f} GB"

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
        """Renders the first layer in detail with internal tensor structure."""
        if not self.data['layers']:
            return

        layer_0 = self.data['layers'][0]
        layer_id = f"layer_{layer_0['index']}"

        with self.dot.subgraph(name=f"cluster_{layer_id}") as c:
            c.attr(label=f"Layer {layer_0['index']} (Detailed View)\\nSize: {format_size(layer_0['size'])}", style='filled', fillcolor='lightgrey')

            # Sub-groups for Attention and FFN
            attn_tensors = [t for t in layer_0['tensors'] if 'attn' in t['name']]
            ffn_tensors = [t for t in layer_0['tensors'] if 'ffn' in t['name']]
            other_tensors = [t for t in layer_0['tensors'] if 'attn' not in t['name'] and 'ffn' not in t['name']]

            if attn_tensors:
                with c.subgraph(name=f"cluster_{layer_id}_attn") as attn_sg:
                    attn_sg.attr(label="Attention", style='rounded,filled', fillcolor='lightblue2')
                    for tensor in attn_tensors:
                        self._add_tensor_node(tensor, attn_sg)

            if ffn_tensors:
                with c.subgraph(name=f"cluster_{layer_id}_ffn") as ffn_sg:
                    ffn_sg.attr(label="Feed-Forward Network", style='rounded,filled', fillcolor='lightgreen2')
                    for tensor in ffn_tensors:
                        self._add_tensor_node(tensor, ffn_sg)

            # Add an invisible edge to enforce vertical alignment
            if attn_tensors and ffn_tensors:
                c.edge(attn_tensors[-1]['name'], ffn_tensors[0]['name'], style='invis')

            for tensor in other_tensors:
                self._add_tensor_node(tensor, c)

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
            shapes = [s[0] for s in first_sig] # Extract shapes from structure signature
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
            # Layers differ, show details for each
            layer_details = []
            for layer in remaining_layers:
                size_info = f"Layer {layer['index']}: {format_size(layer['size'])}"

                sig = layer.get('structure_signature')
                if sig:
                    shapes = [s[0] for s in sig]
                    shape_counts = Counter(shapes)
                    shape_summary = ", ".join(f"{c}x[{'x'.join(map(str, s))}]" for s, c in sorted(shape_counts.items()))
                    layer_details.append(f"{size_info} (Shapes: {shape_summary})")
                else:
                    layer_details.append(size_info)

            details_label = (
                f"--- Layer Details (Structures Vary) ---\\n"
                f"{'\\n'.join(layer_details)}\\n"
            )

        label = (
            f"Stack of {num_layers} Layers\\n"
            f"{details_label}"
            f"------------------------------\\n"
            f"Total Stack Size: {format_size(total_size)}"
        )
        self.dot.node('layer_stack_summary', label=label, shape='box3d', fillcolor='moccasin', fontsize='10')

    def render(self, output_path: str):
        """Generates and saves the final architectural diagram."""
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
        layer_0_cluster_name = f"cluster_layer_{self.data['layers'][0]['index']}"

        if self.data['globals_pre'] and self.data['layers']:
            # Connect from the last pre-global tensor to the first tensor in Layer 0
            self.dot.edge(
                self.data['globals_pre'][-1]['name'],
                self.data['layers'][0]['tensors'][0]['name'],
                lhead=layer_0_cluster_name
            )

        if len(self.data['layers']) > 1:
            # Connect from the last tensor in Layer 0 to the stack summary
            self.dot.edge(
                self.data['layers'][0]['tensors'][-1]['name'],
                'layer_stack_summary',
                ltail=layer_0_cluster_name
            )
            if self.data['globals_post']:
                # Connect from the stack summary to the first post-global tensor
                self.dot.edge(
                    'layer_stack_summary',
                    self.data['globals_post'][0]['name'],
                    lhead='cluster_globals_post'
                )
        elif self.data['layers'] and self.data['globals_post']:
            # If there's no stack, connect Layer 0 directly to the post-globals
            self.dot.edge(
                self.data['layers'][0]['tensors'][-1]['name'],
                self.data['globals_post'][0]['name'],
                ltail=layer_0_cluster_name,
                lhead='cluster_globals_post'
            )

        try:
            self.dot.render(output_path, format='png', view=False, cleanup=True)
            print(f"Diagram saved to {output_path}.png")
        except Exception as e:
            print(f"Error rendering graph: {e}")
            print("Please ensure Graphviz is installed and in your system's PATH.")
            raise
