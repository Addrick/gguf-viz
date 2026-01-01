import gguf
import re
from collections import defaultdict
from gguf.constants import GGML_QUANT_SIZES

class GGUFAnalyzer:
    """Parses a GGUF file to extract and categorize tensor metadata."""

    def __init__(self, file_path: str):
        """
        Initializes the analyzer with the path to a GGUF file.

        Args:
            file_path: The path to the GGUF file.
        """
        self.file_path = file_path
        self.reader = None

    def load(self):
        """Loads the GGUF file using gguf.GGUFReader."""
        try:
            self.reader = gguf.GGUFReader(self.file_path, 'r')
        except Exception as e:
            print(f"Error loading GGUF file: {e}")
            raise

    def analyze(self) -> dict:
        """
        Analyzes the tensors in the loaded GGUF file, grouping them and calculating sizes.

        Returns:
            A dictionary containing the analysis, structured into summary, layers, and globals.
        """
        if not self.reader:
            self.load()

        total_params = 0
        total_size = 0

        layers = defaultdict(lambda: {'size': 0, 'tensors': []})
        globals_pre = []
        globals_post = []
        misc = []

        for tensor in self.reader.tensors:
            tensor_name = tensor.name
            shape = tensor.shape[::-1]  # GGUF shape is reversed
            n_elements = tensor.n_elements

            # Calculate size in bytes using GGML_QUANT_SIZES
            block_size, type_size = GGML_QUANT_SIZES.get(tensor.tensor_type, (1, 0))
            if block_size == 0:
                size_bytes = 0
            else:
                size_bytes = (n_elements * type_size) // block_size

            total_params += n_elements
            total_size += size_bytes

            tensor_info = {
                'name': tensor_name,
                'shape': shape,
                'size_bytes': size_bytes,
                'n_elements': n_elements,
            }

            # Grouping logic
            layer_match = re.match(r'^(blk|layers)\.(\d+)\.(.*)', tensor_name)

            if layer_match:
                layer_index = int(layer_match.group(2))
                layers[layer_index]['size'] += size_bytes
                layers[layer_index]['tensors'].append(tensor_info)
            elif 'token_embd' in tensor_name or 'tok_embeddings' in tensor_name:
                globals_pre.append(tensor_info)
            elif 'output' in tensor_name or 'norm' in tensor_name:
                globals_post.append(tensor_info)
            else:
                misc.append(tensor_info)

        # Add shape signatures to each layer before sorting
        for data in layers.values():
            shapes = [tuple(t['shape']) for t in data['tensors']]
            data['shape_signature'] = tuple(sorted(shapes))

        # Convert defaultdict to a sorted list for predictable order
        sorted_layers = [
            {'index': index, **data} for index, data in sorted(layers.items())
        ]

        return {
            'summary': {
                'total_params': total_params,
                'total_size_bytes': total_size,
            },
            'layers': sorted_layers,
            'globals_pre': globals_pre,
            'globals_post': globals_post,
            'miscellaneous': misc,
        }
