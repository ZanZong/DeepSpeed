import torch
from torch.nn import Module

class IngestLayerWrapper(Module):
    def __init__(self, name, layer_module, preds, ports):
        super().__init__()
        self.module_name = name
        self.mod = layer_module
        self.pred_layers = sorted(preds)
        self.ports = ports
    
    def forward(self, inputs):
        if isinstance(inputs, torch.Tensor):
            print(f"Ingest Layer wrapper {self.module_name} got a Tensor input! \n\n\n")
            return self.mod(inputs)
        elif inputs is None:
            print(f"Ingest Layer wrapper {self.module_name} got a None input! \n\n\n")
            assert len(self.pred_layers) == 1
            _input = self.input_buffers[self.pred_layers[0]][self.buffer_id]
            print(_input)
            return self.mod(_input)
        elif isinstance(inputs, tuple):
            print(f"Ingest Layer wrapper {self.module_name} got a tuple input! \n\n\n")
            # Tuple of tensors includes [None]
            assert hasattr(self, 'input_buffers')
            _inputs = list(inputs)
            cur_pred_layer_idx = 0
            for i in range(len(_inputs)):
                if _inputs[i] is not None:
                    continue
                for (out_, in_) in self.ports[self.pred_layers[cur_pred_layer_idx]]:
                    # Current impl. will send all output of the splitted layer to next stage in Tuple(...).
                    # Only pick referenced tensor here.
                    data = self.input_buffers[cur_pred_layer_idx][self.buffer_id]
                    if isinstance(data, torch.Tensor):
                        _inputs[in_] = data
                    else:
                        _inputs[in_] = data[out_]
                cur_pred_layer_idx += 1
            input_ready = [t is not None for t in _inputs]
            if not all(input_ready):
                print(f"Layer {self.module_name}'s input isn't ready when exec: {input_ready}.")    
            _inputs = tuple(_inputs)
            return self.mod(_inputs)
        else:
            pass
            
        
        
        