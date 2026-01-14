import unittest
from unittest.mock import MagicMock, patch
from collections import namedtuple
import sys
import os

# Add src directory to path to import analyzer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from analyzer import GGUFAnalyzer

# Mock tensor object structure that GGUFReader returns
MockTensor = namedtuple('MockTensor', ['name', 'shape', 'n_elements', 'tensor_type'])

class TestGGUFAnalyzer(unittest.TestCase):

    def setUp(self):
        """Set up a mock GGUFReader and some mock tensors."""
        # Define mock quantization types for clarity in tests
        self.Q8_0 = MagicMock()
        self.F32 = MagicMock()

        self.mock_tensors = [
            MockTensor('token_embd.weight', [4096, 32000], 131072000, self.Q8_0),
            MockTensor('blk.0.attn_q.weight', [4096, 4096], 16777216, self.Q8_0),
            MockTensor('blk.0.ffn_down.weight', [14336, 4096], 58720256, self.Q8_0),
            MockTensor('layers.1.attn_k.weight', [4096, 1024], 4194304, self.Q8_0),
            MockTensor('layers.1.ffn_gate.weight', [4096, 14336], 58720256, self.Q8_0),
            MockTensor('output_norm.weight', [4096], 4096, self.F32),
            MockTensor('output.weight', [32000, 4096], 131072000, self.Q8_0),
            MockTensor('some.other.tensor', [10], 10, self.F32),
        ]

        # Mock gguf.constants.GGML_QUANT_SIZES for size calculation
        self.mock_quant_sizes = {
            self.Q8_0: (32, 34),  # (block_size, type_size) -> (32, 2 + 32)
            self.F32: (1, 4),    # (block_size, type_size)
        }

    @patch('analyzer.gguf.GGUFReader')
    @patch('analyzer.GGML_QUANT_SIZES')
    def test_analyze_structure_and_grouping(self, MockGGMLQuantSizes, MockGGUFReader):
        """Test if tensors are correctly grouped into layers, globals, and misc."""

        # Configure the mock reader and quant sizes
        mock_reader_instance = MockGGUFReader.return_value
        mock_reader_instance.tensors = self.mock_tensors
        MockGGMLQuantSizes.get.side_effect = lambda key, default: self.mock_quant_sizes.get(key, default)

        analyzer = GGUFAnalyzer('fake_path.gguf')
        analyzer.reader = mock_reader_instance # Manually set the mocked reader

        result = analyzer.analyze()

        # Check top-level keys
        self.assertIn('summary', result)
        self.assertIn('layers', result)
        self.assertIn('globals_pre', result)
        self.assertIn('globals_post', result)
        self.assertIn('miscellaneous', result)

        # Check globals
        self.assertEqual(len(result['globals_pre']), 1)
        self.assertEqual(result['globals_pre'][0]['name'], 'token_embd.weight')

        self.assertEqual(len(result['globals_post']), 2)
        self.assertIn(result['globals_post'][0]['name'], ['output_norm.weight', 'output.weight'])

        # Check layers
        self.assertEqual(len(result['layers']), 2)
        self.assertEqual(result['layers'][0]['index'], 0)
        self.assertEqual(len(result['layers'][0]['tensors']), 2)
        self.assertEqual(result['layers'][0]['tensors'][0]['name'], 'blk.0.attn_q.weight')

        self.assertEqual(result['layers'][1]['index'], 1)
        self.assertEqual(len(result['layers'][1]['tensors']), 2)
        self.assertEqual(result['layers'][1]['tensors'][0]['name'], 'layers.1.attn_k.weight')

        # Check miscellaneous
        self.assertEqual(len(result['miscellaneous']), 1)
        self.assertEqual(result['miscellaneous'][0]['name'], 'some.other.tensor')

    @patch('analyzer.gguf.GGUFReader')
    @patch('analyzer.GGML_QUANT_SIZES')
    def test_analyze_calculations(self, MockGGMLQuantSizes, MockGGUFReader):
        """Test the total parameter and size calculations."""
        mock_reader_instance = MockGGUFReader.return_value
        mock_reader_instance.tensors = self.mock_tensors
        MockGGMLQuantSizes.get.side_effect = lambda key, default: self.mock_quant_sizes.get(key, default)

        analyzer = GGUFAnalyzer('fake_path.gguf')
        analyzer.reader = mock_reader_instance

        result = analyzer.analyze()

        # Expected total elements
        expected_total_params = sum(t.n_elements for t in self.mock_tensors)
        self.assertEqual(result['summary']['total_params'], expected_total_params)

        # Expected total size
        expected_total_size = sum(
            (t.n_elements * self.mock_quant_sizes[t.tensor_type][1]) // self.mock_quant_sizes[t.tensor_type][0]
            for t in self.mock_tensors
        )
        self.assertEqual(result['summary']['total_size_bytes'], expected_total_size)

        # Expected size for layer 0
        layer_0_tensors = [t for t in self.mock_tensors if 'blk.0' in t.name]
        expected_layer_0_size = sum(
            (t.n_elements * self.mock_quant_sizes[t.tensor_type][1]) // self.mock_quant_sizes[t.tensor_type][0]
            for t in layer_0_tensors
        )
        self.assertEqual(result['layers'][0]['size'], expected_layer_0_size)

    @patch('analyzer.gguf.GGUFReader')
    @patch('analyzer.GGML_QUANT_SIZES')
    def test_structure_signature_logic(self, MockGGMLQuantSizes, MockGGUFReader):
        """Test that structure signatures are the same for identical layers and different for others."""
        mock_reader_instance = MockGGUFReader.return_value
        MockGGMLQuantSizes.get.side_effect = lambda key, default: self.mock_quant_sizes.get(key, default)

        # Layer 0 and 2 are identical, Layer 1 is different (missing a tensor)
        tensors = [
            MockTensor('blk.0.attn_q.weight', [10, 10], 100, self.Q8_0),
            MockTensor('blk.0.ffn_down.weight', [20, 10], 200, self.Q8_0),

            MockTensor('blk.1.attn_q.weight', [10, 10], 100, self.Q8_0), # Missing ffn_down

            MockTensor('blk.2.attn_q.weight', [10, 10], 100, self.Q8_0),
            MockTensor('blk.2.ffn_down.weight', [20, 10], 200, self.Q8_0),
        ]
        mock_reader_instance.tensors = tensors

        analyzer = GGUFAnalyzer('fake_path.gguf')
        analyzer.reader = mock_reader_instance
        result = analyzer.analyze()

        self.assertEqual(len(result['layers']), 3)

        sig_0 = result['layers'][0]['structure_signature']
        sig_1 = result['layers'][1]['structure_signature']
        sig_2 = result['layers'][2]['structure_signature']

        # Signatures for identical layers should be the same
        self.assertEqual(sig_0, sig_2)

        # Signature for the different layer should be different
        self.assertNotEqual(sig_0, sig_1)
        self.assertNotEqual(sig_2, sig_1)

if __name__ == '__main__':
    unittest.main()
