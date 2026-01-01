import graphviz
from typing import Dict, Any

def format_size(size_bytes: int) -> str:
    """Formats size in bytes to KB, MB, or GB."""
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

            for tensor in other_tensors:
                self._add_tensor_node(tensor, c)

    def _render_layer_stack_summary(self):
        """Renders a summary node for the stack of remaining layers."""
        if len(self.data['layers']) <= 1:
            return

        remaining_layers = self.data['layers'][1:]
        num_layers = len(remaining_layers)
        total_size = sum(layer['size'] for layer in remaining_layers)

        label = (
            f"Stack of {num_layers} Layers\\n"
            f"(Layers 1 to {num_layers})\\n"
            f"Total Stack Size: {format_size(total_size)}"
        )
        self.dot.node('layer_stack_summary', label=label, shape='box3d', fillcolor='moccasin')

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

        # Connect the main components
        if self.data['globals_pre'] and self.data['layers']:
            self.dot.edge(self.data['globals_pre'][0]['name'], self.data['layers'][0]['tensors'][0]['name'], lhead=f"cluster_layer_{self.data['layers'][0]['index']}")

        if len(self.data['layers']) > 1:
            last_tensor_l0 = self.data['layers'][0]['tensors'][-1]['name']
            self.dot.edge(last_tensor_l0, 'layer_stack_summary')

            if self.data['globals_post']:
                self.dot.edge('layer_stack_summary', self.data['globals_post'][0]['name'])
        elif self.data['layers'] and self.data['globals_post']:
            last_tensor_l0 = self.data['layers'][0]['tensors'][-1]['name']
            self.dot.edge(last_tensor_l0, self.data['globals_post'][0]['name'])

        try:
            self.dot.render(output_path, format='png', view=False, cleanup=True)
            print(f"Diagram saved to {output_path}.png")
        except Exception as e:
            print(f"Error rendering graph: {e}")
            print("Please ensure Graphviz is installed and in your system's PATH.")
            raise
