"""FX16 simulation from the final experiment notebook, not packed integer inference."""
import torch
from torch import nn
class QuantizedVisionTransformer(nn.Module):
    """
    Comprehensive FX16 quantization wrapper for Vision Transformer.

    Quantizes:
    - All weights (Linear, Conv2d, Embedding, LayerNorm params)
    - All biases
    - All activations (Linear, Conv2d, Embedding, LayerNorm, GELU, Attention outputs)
    - Input and output
    """

    def __init__(self, model, weight_quantizer, act_quantizer):
        super().__init__()
        self.model = model
        self.wq = weight_quantizer
        self.aq = act_quantizer
        self._quantize_all_weights()
        self._register_activation_hooks()

    def _quantize_all_weights(self):
        """
        Quantize weights and biases across ALL layer types:
        - Linear layers
        - Conv2d layers
        - Embedding layers
        - LayerNorm parameters (scale & shift)
        """
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                module.weight.data = self.wq(module.weight.data)
                if module.bias is not None:
                    module.bias.data = self.wq(module.bias.data)
            elif isinstance(module, nn.Conv2d):
                module.weight.data = self.wq(module.weight.data)
                if module.bias is not None:
                    module.bias.data = self.wq(module.bias.data)
            elif isinstance(module, nn.Embedding):
                module.weight.data = self.wq(module.weight.data)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    module.weight.data = self.wq(module.weight.data)
                if module.bias is not None:
                    module.bias.data = self.wq(module.bias.data)

    def _activation_hook(self, module, input, output):
        """
        Hook function to quantize activations after forward pass.
        Handles both single tensor and tuple outputs.
        """
        if isinstance(output, torch.Tensor):
            return self.aq(output)
        elif isinstance(output, tuple):
            return tuple((self.aq(o) if isinstance(o, torch.Tensor) else o for o in output))
        return output

    def _register_activation_hooks(self):
        """
        Register hooks on ALL layers that produce activations:
        - Linear (QKV, output projections, MLP layers)
        - Conv2d (patch embedding)
        - Embedding (token embeddings)
        - LayerNorm (normalization outputs)
        - GELU (activation function)
        - Softmax (attention weights) - if accessible
        """
        hook_layers = (nn.Linear, nn.Conv2d, nn.Embedding, nn.LayerNorm, nn.GELU, nn.ReLU, nn.Softmax)
        for module in self.model.modules():
            if isinstance(module, hook_layers):
                module.register_forward_hook(self._activation_hook)

    def forward(self, x, mask=None):
        """
        Forward pass with comprehensive quantization:
        1. Quantize input
        2. Process through model (weights already quantized, activations quantized via hooks)
        3. Quantize output
        """
        x = self.aq(x)
        if mask is not None:
            x = self.model(x, mask)
        else:
            x = self.model(x)
        x = self.aq(x)
        return x

    def get_quantization_stats(self):
        """
        Print statistics about quantized layers.
        """
        layer_count = {'Linear': 0, 'Conv2d': 0, 'Embedding': 0, 'LayerNorm': 0, 'Activation': 0}
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                layer_count['Linear'] += 1
            elif isinstance(module, nn.Conv2d):
                layer_count['Conv2d'] += 1
            elif isinstance(module, nn.Embedding):
                layer_count['Embedding'] += 1
            elif isinstance(module, nn.LayerNorm):
                layer_count['LayerNorm'] += 1
        layer_count['Activation'] = sum((1 for m in self.model.modules() if isinstance(m, (nn.Linear, nn.Conv2d, nn.Embedding, nn.LayerNorm, nn.GELU))))
        print('\n' + '=' * 60)
        print('QUANTIZATION STATISTICS')
        print('=' * 60)
        print(f"Linear Layers (Quantized):      {layer_count['Linear']}")
        print(f"Conv2d Layers (Quantized):      {layer_count['Conv2d']}")
        print(f"Embedding Layers (Quantized):   {layer_count['Embedding']}")
        print(f"LayerNorm Layers (Quantized):   {layer_count['LayerNorm']}")
        print(f"Total Activation Hooks:         {layer_count['Activation']}")
        print('=' * 60)
        print(f'Weight Quantization: FX16 (FixedPoint 16,8)')
        print(f'Activation Quantization: FX16 (FixedPoint 16,8)')
        print('=' * 60 + '\n')
        return layer_count

    def disable_quantization(self):
        """
        Remove all activation hooks (keeps weights quantized).
        """
        for module in self.model.modules():
            module._forward_hooks.clear()

    def enable_quantization(self):
        """
        Re-register activation hooks.
        """
        self._register_activation_hooks()


def quantize(model):
    from qtorch.quant import quantizer
    from qtorch import FixedPoint
    q = quantizer(forward_number=FixedPoint(16, 8), forward_rounding='nearest')
    return QuantizedVisionTransformer(model, q, q).eval()
