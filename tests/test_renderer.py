import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os

# Add src directory to path to import renderer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from renderer import GraphRenderer, format_size

class TestGraphRenderer(unittest.TestCase):

    def setUp(self):
        """Set up sample analysis data for testing."""
        self.analysis_data = {
            'summary': {'total_params': 1000, 'total_size_bytes': 1024000},
            'globals_pre': [
                {'name': 'token_embd', 'shape': [10, 10], 'size_bytes': 100}
            ],
            'layers': [
                {
                    'index': 0,
                    'size': 500,
                    'tensors': [
                        {'name': 'layer.0.attn', 'shape': [5, 5], 'size_bytes': 250},
                        {'name': 'layer.0.ffn', 'shape': [5, 5], 'size_bytes': 250},
                    ]
                },
                {
                    'index': 1,
                    'size': 600,
                    'tensors': [
                        {'name': 'layer.1.attn', 'shape': [6, 6], 'size_bytes': 300},
                    ]
                }
            ],
            'globals_post': [
                {'name': 'output', 'shape': [10, 10], 'size_bytes': 100}
            ],
            'miscellaneous': []
        }

    def test_format_size(self):
        """Test the format_size utility function."""
        self.assertEqual(format_size(500), "500 B")
        self.assertEqual(format_size(2048), "2.00 KB")
        self.assertEqual(format_size(1048576 * 2.5), "2.5000 MB")
        self.assertEqual(format_size(1073741824 * 3), "3.0000 GB")

    @patch('renderer.graphviz.Digraph')
    def test_renderer_initialization(self, MockDigraph):
        """Test if the Digraph object is initialized with correct attributes."""
        renderer = GraphRenderer(self.analysis_data)
        mock_dot = MockDigraph.return_value

        mock_dot.attr.assert_any_call(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2')
        mock_dot.attr.assert_any_call('node', shape='box', style='rounded,filled', fillcolor='lightblue')

    @patch('renderer.graphviz.Digraph')
    def test_render_calls(self, MockDigraph):
        """Test the main render method to ensure all sub-render methods are called."""
        mock_dot_instance = MockDigraph.return_value

        renderer = GraphRenderer(self.analysis_data)

        # Mock the sub-methods to check if they are called
        renderer._add_tensor_node = MagicMock()
        renderer._render_layer_0_detail = MagicMock()
        renderer._render_layer_stack_summary = MagicMock()

        renderer.render('test_output')

        # Check that nodes for globals are created
        self.assertEqual(renderer._add_tensor_node.call_count, 2)

        # Check that main rendering components are called
        renderer._render_layer_0_detail.assert_called_once()
        renderer._render_layer_stack_summary.assert_called_once()

        # Check that the final render call is made
        mock_dot_instance.render.assert_called_once_with('test_output', format='png', view=False, cleanup=True)

    @patch('renderer.graphviz.Digraph')
    def test_layer_0_detail_vertical_rendering(self, MockDigraph):
        """Test detailed rendering of Layer 0 enforces a vertical layout."""
        mock_dot_instance = MockDigraph.return_value

        # Use a MagicMock to simulate the subgraph context manager
        mock_subgraph = MagicMock()
        mock_dot_instance.subgraph.return_value.__enter__.return_value = mock_subgraph

        renderer = GraphRenderer(self.analysis_data)
        # We patch _add_tensor_node because we're not testing its functionality here,
        # only that it's called correctly.
        renderer._add_tensor_node = MagicMock()

        renderer._render_layer_0_detail()

        # 1. Check that a cluster is created for Layer 0
        mock_dot_instance.subgraph.assert_any_call(name='cluster_layer_0')

        # 2. Check that tensor nodes were added for each tensor in layer 0
        self.assertEqual(renderer._add_tensor_node.call_count, 2)

        # 3. Check that an invisible edge was created to enforce vertical layout
        # This is the key change from the previous implementation.
        layer_0_tensors = self.analysis_data['layers'][0]['tensors']
        expected_edge_call = call(
            layer_0_tensors[0]['name'],
            layer_0_tensors[1]['name'],
            style='invis'
        )
        mock_subgraph.edge.assert_has_calls([expected_edge_call])

    @patch('renderer.graphviz.Digraph')
    def test_layer_stack_summary_node(self, MockDigraph):
        """Test if the layer stack summary node is created correctly."""
        mock_dot_instance = MockDigraph.return_value

        renderer = GraphRenderer(self.analysis_data)
        renderer._render_layer_stack_summary()

        # Verify the node call
        args, kwargs = mock_dot_instance.node.call_args
        self.assertEqual(args[0], 'layer_stack_summary')
        self.assertIn("Stack of 1 Layers", kwargs['label'])
        self.assertIn("Total Stack Size", kwargs['label'])
        self.assertEqual(kwargs['shape'], 'box3d')

if __name__ == '__main__':
    unittest.main()
