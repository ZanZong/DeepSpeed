import os
import glob

import re as regex

from functools import partial, lru_cache

import torch
import torch.nn as nn
from deepspeed import comm as dist

from deepspeed.utils import logger
from .. import utils as ds_utils
from ..activation_checkpointing import checkpointing
from .topology import PipeDataParallelTopology, PipelineParallelGrid
from deepspeed.runtime.state_dict_factory import SDLoaderFactory

import time
import networkx as nx
from collections import defaultdict
from . import p2p
from functools import lru_cache
from .layer_wrapper import IngestLayerWrapper

class PipelineError(Exception):
    """Errors related to the use of deepspeed.PipelineModule """


class LayerSpec:
    """Building block for specifying pipeline-parallel modules.

    LayerSpec stores the type information and parameters for each stage in a
    PipelineModule. For example:

    .. code-block:: python

        nn.Sequence(
            torch.nn.Linear(self.in_dim, self.hidden_dim, bias=False),
            torch.nn.Linear(self.hidden_hidden, self.out_dim)
        )

    becomes

    .. code-block:: python

        layer_specs = [
            LayerSpec(torch.nn.Linear, self.in_dim, self.hidden_dim, bias=False),
            LayerSpec(torch.nn.Linear, self.hidden_hidden, self.out_dim)]
        ]
    """
    def __init__(self, typename, *module_args, **module_kwargs):
        self.typename = typename
        self.module_args = module_args
        self.module_kwargs = module_kwargs

        if not issubclass(typename, nn.Module):
            raise RuntimeError('LayerSpec only supports torch.nn.Module types.')

        if dist.is_initialized():
            self.global_rank = dist.get_rank()
        else:
            self.global_rank = -1

    def __repr__(self):
        return ds_utils.call_to_str(self.typename.__name__,
                                    self.module_args,
                                    self.module_kwargs)

    def build(self, log=False):
        """Build the stored specification."""
        if log:
            logger.info(f'RANK={self.global_rank} building {repr(self)}')

        return self.typename(*self.module_args, **self.module_kwargs)


class TiedLayerSpec(LayerSpec):
    def __init__(self,
                 key,
                 typename,
                 *module_args,
                 forward_fn=None,
                 tied_weight_attr='weight',
                 **module_kwargs):
        super().__init__(typename, *module_args, **module_kwargs)
        self.key = key
        self.forward_fn = forward_fn
        self.tied_weight_attr = tied_weight_attr

class StageInput:
    def __init__(self) -> None:
        pass
    
MAX_LAYER_NUM = 1000 # This layer is loss layer
MIN_LAYER_NUM = -1 # This layer directly reads data sets, TODO: a better notation
class PipelineBranchModule(nn.Module):
    """Enabling pipeline parallelism for heavy multi-branch modules.

    """
    def __init__(self,
                 layers,
                 graph_info=None,
                 see_baseline_perf=False,
                 is_constrastive_loss=False,
                 timeline_path=None,
                 num_stages=None,
                 topology=None,
                 loss_fn=None,
                 seed_layers=False,
                 seed_fn=None,
                 base_seed=1234,
                 partition_method='parameters',
                 activation_checkpoint_interval=0,
                 activation_checkpoint_func=checkpointing.checkpoint,
                 checkpointable_layers=None):

        super().__init__()

        if num_stages is None and topology is None:
            raise RuntimeError('must provide num_stages or topology')

        self.micro_offset = 0

        self.loss_fn = loss_fn

        self.checkpointable_layers = checkpointable_layers
        if checkpointable_layers is not None:
            assert isinstance(checkpointable_layers, list), "param `checkpointable_layers` must be type of list."

        self.seed_layers = seed_layers
        self.graph = graph_info
        self.see_baseline_perf = see_baseline_perf
        self.is_constrastive_loss = is_constrastive_loss
        self.timeline_path = timeline_path
        self.seed_fn = seed_fn
        self.base_seed = base_seed
        if dist.get_rank() == 0:
            try:
                seed_str = self.seed_fn.__name__
            except AttributeError:
                seed_str = None
            print(
                f'SEED_LAYERS={self.seed_layers} BASE_SEED={self.base_seed} SEED_FN={seed_str}'
            )

        # Setup world info
        self.world_group = dist.new_group(ranks=range(dist.get_world_size()))
        self.global_rank = dist.get_rank(group=self.world_group)
        self.world_size = dist.get_world_size(group=self.world_group)
        self.local_rank = int(os.environ.get("LOCAL_RANK", None))
        assert self.local_rank != None

        if topology:
            self._topo = topology
            self.num_stages = self._topo.get_dim('pipe')
        else:
            self.num_stages = num_stages
            if topology is None:
                if self.world_size % self.num_stages != 0:
                    raise RuntimeError(
                        f'num_stages ({self.num_stages}) must divide distributed world size ({self.world_size})'
                    )
                dp = self.world_size // num_stages
                topology = PipeDataParallelTopology(num_pp=num_stages, num_dp=dp)
                self._topo = topology

        # Construct communicators for pipeline topology
        self._grid = PipelineParallelGrid(process_group=self.world_group,
                                          topology=self._topo)

        self.stage_id = self._topo.get_coord(self.global_rank).pipe

        # Initialize partition information
        self._layer_specs = list(layers)
        self._num_layers = len(self._layer_specs)
        self._local_start = 0
        self._local_stop = None
        self._partition_layers(method=partition_method)

        self.forward_funcs = []
        self.fwd_map = {}
        self.tied_modules = nn.ModuleDict()
        self.tied_weight_attrs = {}

        # Offset the random seed by the stage ID.
        #newseed = torch.cuda.initial_seed() + self._grid.get_stage_id()
        #ds_utils.set_random_seed(newseed)

        #with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        self._build()
        self.to(f'cuda:{self.local_rank}')
        self.extract_graph_info(self.graph)
        self.buffer_num = None # pipeline buffer number for micro-batches
        self.build_stage_buffer()

        self.tied_comms = self._index_tied_modules()
        self._synchronize_tied_weights()

        self.activation_checkpoint_interval = activation_checkpoint_interval
        self.activation_checkpoint_func = activation_checkpoint_func
        
        ds_utils.see_memory_usage("Memory after build branch module", True)
        # self.debug_hook_for_backward()
        
    def _build(self):
        specs = self._layer_specs
        self.name2local_idx = dict()
        for local_idx, layer in enumerate(specs[self._local_start:self._local_stop]):
            layer_idx = local_idx + self._local_start
            if self.seed_layers:
                if self.seed_fn:
                    self.seed_fn(self.base_seed + layer_idx)
                else:
                    ds_utils.set_random_seed(self.base_seed + layer_idx)

            # LayerSpec objects contain an nn.Module that should be allocated now.
            elif isinstance(layer, nn.Module):
                name = str(layer_idx)
                self.name2local_idx[name] = local_idx
                self.forward_funcs.append(layer)
                self.fwd_map.update({name: len(self.forward_funcs) - 1})
                self.add_module(name, layer)

            # TiedLayerSpec objects contain an nn.Module that should be allocated now.
            elif isinstance(layer, TiedLayerSpec):
                # Build and register the module if we haven't seen it before.
                if layer.key not in self.tied_modules:
                    self.tied_modules[layer.key] = layer.build()
                    self.tied_weight_attrs[layer.key] = layer.tied_weight_attr

                if layer.forward_fn is None:
                    # Just use forward()
                    self.forward_funcs.append(self.tied_modules[layer.key])
                else:
                    # User specified fn with args (module, input)
                    self.forward_funcs.append(
                        partial(layer.forward_fn,
                                self.tied_modules[layer.key]))

            # LayerSpec objects contain an nn.Module that should be allocated now.
            elif isinstance(layer, LayerSpec):
                module = layer.build()
                name = str(layer_idx)
                self.forward_funcs.append(module)
                self.fwd_map.update({name: len(self.forward_funcs) - 1})
                self.add_module(name, module)

            # Last option: layer may be a functional (e.g., lambda). We do nothing in
            # that case and just use it in forward()
            else:
                self.forward_funcs.append(layer)

        # All pipeline parameters should be considered as model parallel in the context
        # of our FP16 optimizer
        for p in self.parameters():
            p.ds_pipe_replicated = False

    def _count_layer_params(self):
        """Count the trainable parameters in individual layers.

        This routine will only build one layer at a time.

        Returns:
            A list of the number of parameters in each layer.
        """
        param_counts = [0] * len(self._layer_specs)
        for idx, layer in enumerate(self._layer_specs):
            if isinstance(layer, LayerSpec):
                l = layer.build()
                params = filter(lambda p: p.requires_grad, l.parameters())
                param_counts[idx] = sum(p.numel() for p in params)
            elif isinstance(layer, nn.Module):
                params = filter(lambda p: p.requires_grad, layer.parameters())
                param_counts[idx] = sum(p.numel() for p in params)
        return param_counts

    def _find_layer_type(self, layername):
        idxs = []
        typeregex = regex.compile(layername, regex.IGNORECASE)
        for idx, layer in enumerate(self._layer_specs):
            name = None
            if isinstance(layer, LayerSpec):
                name = layer.typename.__name__
            elif isinstance(layer, nn.Module):
                name = layer.__class__.__name__
            else:
                try:
                    name = layer.__name__
                except AttributeError:
                    continue
            if typeregex.search(name):
                idxs.append(idx)

        if len(idxs) == 0:
            raise RuntimeError(
                f"Partitioning '{layername}' found no valid layers to partition.")
        return idxs
    
    # def check_stage_load_inputs(self):
    #     if len(self.layer_inputs[self._local_start]) == 0:
    #         return True
    #     elif self.stage_id == self.num_stages - 1 and not self.is_constrastive_loss:
    #         return True
    #     else:
    #         return False
        
    def __parse_param_order(self, mapping_str: str):
        params_mapping = []
        # This slice remove string quotation mark
        for mapping in mapping_str[1: -1].split(","):
            out_in_indexes = mapping.split("_")
            assert len(out_in_indexes) == 2, \
                f"The edge attribute {mapping_str} must be specified with '_' between two indexes."
            params_mapping.append((int(out_in_indexes[0]), int(out_in_indexes[1])))
        return params_mapping
    
    def extract_graph_info(self, graph):
        """Construct self.layer_outputs as 
                    {layer0: 
                        {adj_layer0:[(layer0_out0, adj_in0), 
                                    (layer0_out1, adj_in1)], 
                         adj_layer1:...},
                    layer1: 
                        {adj_layer0:[(layer0_out0, adj_in0), 
                                    (layer0_out1, adj_in1)], 
                         adj_layer1:...},
                            ...
                    }. 
        Args:
            graph (DiGraph): DiGraph built with networkx.
        """
        # Get each layer's outputs.
        if not hasattr(self, "layer_outputs"):
            self.layer_outputs = {}
            for node, nbrs in self.graph.adj.items():
                if node == "\\n": continue
                adj_param_mapping = {}
                for adj, map_out_in in nbrs.items():
                    adj_param_mapping[int(adj)] = self.__parse_param_order(map_out_in["map_out_in"])
                self.layer_outputs[int(node)] = adj_param_mapping
                
        # Get each layer's inputs.
        reverse_graph = self.graph.reverse(copy=True)
        if not hasattr(self, "layer_inputs"):
            self.layer_inputs = {}
            for node, nbrs in reverse_graph.adj.items():
                if node == "\\n": continue
                adj_param_mapping = {}
                for adj, map_out_in in nbrs.items():
                    adj_param_mapping[int(adj)] = self.__parse_param_order(map_out_in["map_out_in"])
                self.layer_inputs[int(node)] = adj_param_mapping
    
    def in_stage(self, layer_idx):    
            return True if layer_idx >= self._local_start and layer_idx < self._local_stop else False
    
    def get_layer_module(self, layer_idx):
        return self.get_submodule(str(layer_idx))

    def replace_layer_module(self, layer_idx, new_module):
        assert hasattr(self, str(layer_idx)), f"To be replaced layer {layer_idx} isn't in module."
        delattr(self, str(layer_idx))
        if isinstance(new_module, nn.Module):
            self.add_module(str(layer_idx), new_module)
            self.forward_funcs[self.name2local_idx[str(layer_idx)]] = new_module

    def __find_outstage(self):
        # Find layers with out-stage connections
        outstage_outputs = defaultdict(list)
        outstage_inputs = defaultdict(list)
       
        for layer_id in range(self._local_start, self._local_stop):
            # Output pass
            if len(self.layer_outputs[layer_id].keys()) == 0:
                outstage_outputs[layer_id].append(MAX_LAYER_NUM)
            else:
                for succ_layer in sorted(self.layer_outputs[layer_id].keys()):
                    if not self.in_stage(succ_layer):
                        outstage_outputs[layer_id].append(succ_layer)
            # # Input pass
            if len(self.layer_inputs[layer_id].keys()) == 0:
                outstage_inputs[layer_id].append(MIN_LAYER_NUM) 
            else:
                for predecs_layer in sorted(self.layer_inputs[layer_id].keys()):
                    if not self.in_stage(predecs_layer):
                        outstage_inputs[layer_id].append(predecs_layer)
        self.output_ref = outstage_outputs
        self.ordered_output_ref = sorted([key for key in outstage_outputs.keys() if key != MAX_LAYER_NUM])
        self.input_ref = outstage_inputs
        
        # Find an input layer depends on previous stage.
        # # Wrap the layer to handle forward().
        # for layer_id, preds in self.input_ref.items():
        #     mod = self.get_layer_module(layer_id)
        #     wrapped_mod = IngestLayerWrapper(name=layer_id,
        #                                      layer_module=mod,
        #                                      preds=preds,
        #                                      ports=self.layer_inputs[layer_id])
        #     self.replace_layer_module(layer_id, wrapped_mod)
        # print(f"Rank {self.local_rank}, replace module with wrapper: {self.input_ref.keys()}")
    
    def __init_layer_buffers(self):
        """ We build output buffers for every layer in self.layer_output_buffer, and
            out-stage input buffer in layer modules that have out-stage connections.
            Scheduling algorithm will call expand_buffers_wrt_sched() to lengthen the buffers
            according to the number of buffers.
        """
        # Output pass
        # for layer_id, succes in self.output_ref.items():
            # layer_mod = self.get_layer_module(layer_id)
            # layer_mod.output_buffers = dict() # {succ0: [buffer0, buffer1], ...}
            # if len(self.layer_outputs[layer_id].keys()) == 0:
            #     layer_mod.output_buffers[MAX_LAYER_NUM] = [tuple([None])]
            # else:
            #     for succ_layer in sorted(succes):
            #         layer_mod.output_buffers[succ_layer] = [tuple([None] * \
            #                                     len(self.layer_outputs[layer_id][succ_layer]))]
        # Input buffer
        for layer_id, preds in self.input_ref.items():
            layer_mod = self.get_layer_module(layer_id)
            if hasattr(layer_mod, "layer_inputs"):
                del layer_mod.layer_inputs
            layer_mod.input_buffers = dict() # {pred0: [buffer0, buffer1], ...}
            if len(self.layer_inputs[layer_id].keys()) == 0:
                layer_mod.input_buffers[MIN_LAYER_NUM] = [None] # TODO: assume a layer only eat single tensor.
            else:
                for predecs_layer in sorted(preds):
                    out_num_from_pred = len(self.layer_inputs[layer_id][predecs_layer])
                    if out_num_from_pred == 1:
                        layer_mod.input_buffers[predecs_layer] = [None]
                    else:
                        layer_mod.input_buffers[predecs_layer] = [tuple([None] * out_num_from_pred)]
        
        for layer_id, succes in self.output_ref.items():
            layer_mod = self.get_layer_module(layer_id)
            if hasattr(layer_mod, "grad_layer"):
                del layer_mod.grad_layer
        
        if hasattr(self, "layer_output_buffer"):
            del self.layer_output_buffer
        # For layers do not have out-stage connections, just init None tuple
        self.layer_output_buffer = dict()
        for idx in range(self._local_start, self._local_stop):
            self.layer_output_buffer[idx] = [None]
        
    def expand_buffers_wrt_sched(self, buffer_num):
        """Expand `num_buffers`` times when scheudling."""
        self.buffer_num = buffer_num
        
        for cur, preds in self.input_ref.items():
            for key in preds:
                self.get_layer_module(cur).input_buffers[key] *= buffer_num
            # print(f"Layer {cur} expand buffer, {self.get_layer_module(cur).input_buffers}")
        
        # form output as a dict for easy to empty
        for idx in range(self._local_start, self._local_stop):
            self.layer_output_buffer[idx] = {i: None for i in range(buffer_num)}
        
    def __register_hooks(self):
        hooked_layers = list()
        from .register import gen_module_hook_activation_recv, \
            gen_module_hook_grad_recv_hook, gen_module_hook_activation_send
        for layer_id in self.input_ref.keys():
            if MIN_LAYER_NUM in self.input_ref[layer_id]:
                continue
            layer_mod = self.get_layer_module(layer_id)
            '''
            for pred_layer in self.input_ref[layer_id]:
                # if pred_layer != MIN_LAYER_NUM:
                layer_mod.register_forward_pre_hook(
                                    gen_module_hook_activation_recv(
                                        pred_layer,
                                        self.stage_id - 1, 
                                        self.local_rank))
            '''
            # Fill some necessary info to this module
            # layer_mod.pipe_recv_buf = None
            hooked_layers.append(layer_id)

        for layer_id in self.output_ref.keys():
            if MAX_LAYER_NUM in self.output_ref[layer_id]:
                continue
            layer_mod = self.get_layer_module(layer_id)
            '''
            for succ_layer in self.output_ref[layer_id]:
                # if succ_layer != MAX_LAYER_NUM:
                layer_mod.register_forward_hook(
                                        gen_module_hook_activation_send(
                                            self.local_rank,
                                            self.stage_id + 1))
                layer_mod.register_full_backward_pre_hook(
                                        gen_module_hook_grad_recv_hook(
                                            succ_layer,
                                            self.stage_id + 1,
                                            self.local_rank))
            '''
            # layer_mod.grad_layer = None
            hooked_layers.append(layer_id)
            
        self.hooked_layers = list(set(hooked_layers))
        print(f"Rank {self.local_rank}, hooked layers:{self.hooked_layers}")

    @lru_cache(maxsize=512)
    def query_stage(self, layer_id):
        num_stages = self._topo.get_dim('pipe')
        for stage in range(num_stages):
            start = self.parts[stage]
            stop = self.parts[stage + 1]
            if layer_id >= start and layer_id < stop:
                return stage
            elif stage == num_stages - 1 and layer_id >= start and layer_id <= stop:
                return num_stages - 1
            else:
                continue
            
    @lru_cache(maxsize=512)
    def query_output_successors(self, layer_id):
        """Get successor infomation. 
        Args:
            layer_id (_type_): which layer to query.
        Returns:
            out_tensor_to_layer: {out_tensor_id: (successor_layer_id, input_index)}
        """
        out_tensor_to_layer = {}
        for succ in sorted(self.layer_outputs[layer_id].keys()):
            for out_, in_ in self.layer_outputs[layer_id][succ]:
                if out_ in out_tensor_to_layer.keys():
                    print("[WARN] Need support: repeat access of a tensor for multiple successors.")
                out_tensor_to_layer[out_] = (succ, in_)
        return out_tensor_to_layer
    
    def build_stage_buffer(self):
        self.__find_outstage()
        self.__init_layer_buffers()
        # self.__register_hooks()
        self.stage_read_data_layers = dict()
        # check if local stage needs original input data
        for stage in range(self.num_stages):
            self.stage_read_data_layers[stage] = list()
        for layer_id, predecs_layer in self.layer_inputs.items():
            if len(predecs_layer.keys()) == 0:
                stage = self.query_stage(layer_id)
                self.stage_read_data_layers[stage].append(layer_id)
        # TODO what if normal loss layer
        # if not self.is_constrastive_loss:
        #     self.stage_read_data_layers[self.num_stages - 1].append(self.stage_id)
        logger.info(f"Local rank={self.local_rank}, input_={self.input_ref}, output_={self.output_ref}, read_data_layer={self.stage_read_data_layers}")

    def rebuild_buffers(self):
        self.__init_layer_buffers()

    def debug_hook_for_backward(self):
        def backward_hook(module, res, grad_out):
            ds_utils.see_memory_usage(f"Call backward layers={module._get_name()}", True)
            
        for name, module in self.named_children():
            module.register_full_backward_hook(backward_hook)
    
    def form_layer_input(self, layer_id, buffer_id, check_graph_tail=False):    
        param_count = sum([len(maps) for _, maps in self.layer_inputs[layer_id].items()])
        # print(f"Layer {layer_id}: {param_count} parameters are required")
        # Handle the training input of each branch.
        if param_count == 0:
            inputs_ = []
            for ingest_layer in sorted(self.input_ref.keys()):
                if ingest_layer != layer_id:
                    continue
                for pred_layer in self.input_ref[ingest_layer]:
                    data = self.get_layer_module(ingest_layer).input_buffers[pred_layer][buffer_id]
                    if isinstance(data, tuple):
                        inputs_.append([t for t in data])
                    else:
                        inputs_.append(data)
            for tensor in inputs_:
                if tensor.grad is not None:
                    tensor.grad.data.zero_()
            return inputs_[0] if len(inputs_) == 1 else tuple(inputs_)
        
        inputs = [None] * param_count
        for predecs_layer in sorted(self.layer_inputs[layer_id].keys()):
            # Find input within stage
            if predecs_layer in self.layer_output_buffer.keys():
                pred_mod = self.get_layer_module(predecs_layer)
                # print(f"find pred {predecs_layer} in layer_output_buffer, for buffer id {buffer_id}")
                for (out_, in_) in self.layer_inputs[layer_id][predecs_layer]:
                    if check_graph_tail and hasattr(pred_mod, 'tail_outputs'):
                        # print(f"{in_}^th tensor out from {predecs_layer}'s tail_outputs")
                        if isinstance(pred_mod.tail_outputs[buffer_id], torch.Tensor):
                            inputs[in_] = pred_mod.tail_outputs[buffer_id]
                        else:
                            inputs[in_] = pred_mod.tail_outputs[buffer_id][out_]
                    else:
                        # print(f"{in_}^th tensor out from {predecs_layer}'s layer_output_buffer")
                        if isinstance(self.layer_output_buffer[predecs_layer][buffer_id], torch.Tensor):
                            inputs[in_] = self.layer_output_buffer[predecs_layer][buffer_id]
                        else:  
                            inputs[in_] = self.layer_output_buffer[predecs_layer][buffer_id][out_]
            
            # Find input tensor through out-stage buffers
            if predecs_layer in self.input_ref[layer_id]:
                # print(f"find pred {predecs_layer} in module {layer_id}'s layer_inputs, for buffer id {buffer_id}")
                module_buffer = self.get_layer_module(layer_id).input_buffers[predecs_layer][buffer_id]   
                if isinstance(module_buffer, torch.Tensor):
                    out_, in_ = self.layer_inputs[layer_id][predecs_layer][0]
                    # print(f"{in_}^th tensor out from {layer_id}'s layer_inputs")
                    inputs[in_] = module_buffer
                else:
                    assert isinstance(module_buffer, tuple)
                    for (out_, in_) in self.layer_inputs[layer_id][predecs_layer]:
                        # print(f"{in_}^th tensor out from {layer_id}'s layer_inputs")
                        inputs[in_] = module_buffer[out_]
                        
        input_ready = [t is not None for t in inputs]
        if not all(input_ready):
            print(f"Layer {layer_id}'s input not satisfied: {input_ready}.")    
            
        return inputs[0] if isinstance(inputs, list) and len(inputs) == 1 else tuple(inputs)
        
    def sink_layer_output(self, outputs, layer_id, buffer_id):
        self.layer_output_buffer[layer_id][buffer_id] = outputs
        
    def form_layer_output_to_send(self, send_layer, recv_layer, buffer_id):
        """ Generate a tensor tuple that need to be sent for `layer_id` layer."""
        assert send_layer in self.output_ref.keys()
        sender_buffer = self.layer_output_buffer[send_layer][buffer_id]
        if isinstance(sender_buffer, torch.Tensor):
            return sender_buffer
        else:
            send_tensors = []
            for out_, in_ in self.layer_outputs[send_layer][recv_layer]:
                send_tensors.append(self.layer_output_buffer[send_layer][buffer_id][out_])
            return tuple(send_tensors)
        
    def naive_forward(self, start, end, buffer_id, graph_tail=False):
        # for testing forward one-by-one
        local_micro_offset = self.micro_offset + 1
        start_func_idx = start - self._local_start
        end_func_idx = end - self._local_start
        for idx, layer in enumerate(self.forward_funcs[start_func_idx:end_func_idx]):
            curr_layer = idx + start
            # print(f"Rank {self.local_rank} exec layer {curr_layer}")
            if self.seed_layers:
                new_seed = (self.base_seed *
                            local_micro_offset) + curr_layer
                if self.seed_fn:
                    self.seed_fn(new_seed)
                else:
                    ds_utils.set_random_seed(new_seed)
            if idx == 0:
                inputs = self.form_layer_input(curr_layer, buffer_id, check_graph_tail=graph_tail)
            
            outputs = layer(inputs)
            inputs = outputs
        return outputs
        
    def partial_forward(self, start, end, buffer_id, graph_tail=False):
        local_micro_offset = self.micro_offset + 1
        assert start >= self._local_start
        start_func_idx = start - self._local_start
        end_func_idx = end - self._local_start
        for idx, layer in enumerate(self.forward_funcs[start_func_idx:end_func_idx]):
            curr_layer = idx + start
            # print(f"Rank {self.local_rank} exec layer {curr_layer}")
            if self.seed_layers:
                new_seed = (self.base_seed *
                            local_micro_offset) + curr_layer
                if self.seed_fn:
                    self.seed_fn(new_seed)
                else:
                    ds_utils.set_random_seed(new_seed)

            inputs = self.form_layer_input(curr_layer, buffer_id, check_graph_tail=True)
            outputs = layer(inputs)
            self.sink_layer_output(outputs, curr_layer, buffer_id)
            # ds_utils.see_memory_usage(f"Rank={self.local_rank}, memory after exec layer {curr_layer}", True)
        
        # Leave a tail for feeding next graph
        if graph_tail:
            mod = self.get_layer_module(curr_layer)
            if isinstance(outputs, torch.Tensor):
                tail_outputs = outputs.clone().detach()
                tail_outputs.requires_grad = tail_outputs.is_floating_point()
            else:
                tail_outputs = tuple([t.clone().detach() for t in outputs])
                for t in tail_outputs:
                    t.requires_grad = t.is_floating_point()
            if not hasattr(mod, "tail_outputs"):
                setattr(mod, "tail_outputs", {i: None for i in range(self.buffer_num)})
            mod.tail_outputs[buffer_id] = tail_outputs
            # print(f"Layer {curr_layer} breaks the graph with a tail. tail_outputs={mod.tail_outputs}")
                
        return outputs
    
    def forward(self, forward_input):
        # TODO: activation checkpointing feature to be impl.
        self.micro_offset += 1

        if self.activation_checkpoint_interval == 0:
            pass
        else:
            raise ValueError("Activation checkpointing is not supported for now.")            
            num_layers = len(self.forward_funcs)
            x = forward_input
            for start_idx in range(0, num_layers, self.activation_checkpoint_interval):
                end_idx = min(start_idx + self.activation_checkpoint_interval,
                              num_layers)

                funcs = self.forward_funcs[start_idx:end_idx]
                # Since we either pass tensors or tuples of tensors without unpacking, we
                # need to be careful not to double-wrap tensors with tuple.
                if not isinstance(x, tuple):
                    x = (x, )

                if self._is_checkpointable(funcs):
                    x = self.activation_checkpoint_func(
                        exec_range_func(start_idx,
                                        end_idx),
                        *x)
                else:
                    x = exec_range_func(start_idx, end_idx)(*x)
        return x

    def _partition_layers(self, method='uniform'):
        num_stages = self._topo.get_dim('pipe')
        stage_id = self._topo.get_coord(self.global_rank).pipe

        if isinstance(method, str):
            method = method.lower()

        # Each stage gets a simple uniform number of layers.
        if isinstance(method, list):
            self.parts = method
        elif method == 'uniform':
            num_layers = len(self._layer_specs)
            self.parts = ds_utils.partition_uniform(num_items=num_layers,
                                                    num_parts=num_stages)
        elif method == 'parameters':
            param_counts = self._count_layer_params()
            self.parts = ds_utils.partition_balanced(weights=param_counts,
                                                     num_parts=num_stages)
        elif method.startswith('type:'):
            layertype = method.split(':')[1]
            binary_weights = [0] * len(self._layer_specs)
            for idx in self._find_layer_type(layertype):
                binary_weights[idx] = 1
            self.parts = ds_utils.partition_balanced(weights=binary_weights,
                                                     num_parts=num_stages)
        elif method == 'profile':
            raise NotImplementedError(f'Partitioning method {method} not implemented.')
        else:
            raise NotImplementedError(f'Partitioning method {method} not implemented.')

        # Print some information on the partitioning.
        if self.global_rank == 0:
            logger.info(f'Partitioning pipeline stages with method {method}, get partition: {self.parts}')
            for stage in range(num_stages):
                start = self.parts[stage]
                stop = self.parts[stage + 1]
                print(f'stage={stage} layers={stop - start}')
                for idx, layer in enumerate(self._layer_specs[start:stop]):
                    name = str(layer)
                    if isinstance(layer, LayerSpec):
                        name = layer.typename.__name__
                    if isinstance(layer, nn.Module):
                        name = layer.__class__.__name__
                    else:
                        try:
                            name = layer.__name__
                        except AttributeError:
                            pass
                    print(f'    {idx+start:2d}: {name}')
            if self.loss_fn:
                try:
                    print(f'  loss: {self.loss_fn.__name__}')
                except AttributeError:
                    print(f'  loss: {self.loss_fn.__class__.__name__}')

        self._set_bounds(start=self.parts[stage_id], stop=self.parts[stage_id + 1])

    def allreduce_tied_weight_gradients(self):
        '''All reduce the gradients of the tied weights between tied stages'''
        for key, comm in self.tied_comms.items():
            weight = getattr(self.tied_modules[key], comm['weight_attr'])
            dist.all_reduce(weight.grad, group=comm['group'])

    def get_tied_weights_and_groups(self):
        weight_group_list = []
        for key, comm in self.tied_comms.items():
            weight = getattr(self.tied_modules[key], comm['weight_attr'])
            weight_group_list.append((weight, comm['group']))
        return weight_group_list

    def _synchronize_tied_weights(self):
        for key, comm in self.tied_comms.items():
            dist.broadcast(
                getattr(comm['module'],
                        comm['weight_attr']),
                src=min(comm['ranks']),
                group=comm['group'],
            )

    def _index_tied_modules(self):
        ''' Build communication structures for tied modules. '''
        tied_comms = {}
        if self._topo.get_dim('pipe') == 1:
            return tied_comms

        specs = self._layer_specs
        tie_keys = set(s.key for s in specs if isinstance(s, TiedLayerSpec))
        for key in tie_keys:
            # Find the layers that the tied module appears in
            tied_layers = []
            for idx, layer in enumerate(specs):
                if isinstance(layer, TiedLayerSpec) and layer.key == key:
                    tied_layers.append(idx)
            # Find all stages with this tied module
            # TODO: Would be nice to remove the nested data/model parallelism loops and
            # TODO: instead generalize in some way, since we really just care about the
            # TODO: stage that owns the tied layer. Then loop over each (dp, mp, ...)
            # TODO: fiber to generate process groups.
            tied_stages = set(self.stage_owner(idx) for idx in tied_layers)
            for dp in range(self._grid.data_parallel_size):
                for mp in range(self._grid.get_slice_parallel_world_size()):
                    tied_ranks = []
                    for s in sorted(tied_stages):
                        if self._grid.get_slice_parallel_world_size() > 1:
                            tied_ranks.append(
                                self._grid.stage_to_global(stage_id=s,
                                                           data=dp,
                                                           model=mp))
                        else:
                            tied_ranks.append(
                                self._grid.stage_to_global(stage_id=s,
                                                           data=dp))
                    group = dist.new_group(ranks=tied_ranks)

                    # Record this tied module if we own a local copy of it.
                    if self.global_rank in tied_ranks:
                        assert key in self.tied_modules
                        if key in self.tied_modules:
                            tied_comms[key] = {
                                'ranks': tied_ranks,
                                'group': group,
                                'weight_attr': self.tied_weight_attrs[key],
                                'module': self.tied_modules[key],
                            }
                            # Only count the tied module once in the eyes of the FP16 optimizer
                            if self.global_rank != tied_ranks[0]:
                                for p in self.tied_modules[key].parameters():
                                    p.ds_pipe_replicated = True
        '''
        if len(tied_comms) > 0:
            print(f'RANK={self.global_rank} tied_comms={tied_comms}')
        '''

        return tied_comms

    def partitions(self):
        return self.parts

    def stage_owner(self, layer_idx):
        assert 0 <= layer_idx < self._num_layers
        for stage in range(self._topo.get_dim('pipe')):
            if self.parts[stage] <= layer_idx < self.parts[stage + 1]:
                return stage
        raise RuntimeError(f'Layer {layer_idx} not owned? parts={self.parts}')

    def _set_bounds(self, start=None, stop=None):
        """Manually define the range of layers that will be built on this process.

        These boundaries are treated as list slices and so start is inclusive and stop is
        exclusive. The default of None for both results in all layers being built
        locally.
        """
        self._local_start = start
        self._local_stop = stop

    def set_checkpoint_interval(self, interval):
        assert interval >= 0
        self.checkpoint_interval = interval

    def topology(self):
        """ ProcessTopology object to query process mappings. """
        return self._topo

    def mpu(self):
        return self._grid

    def num_pipeline_stages(self):
        return self._topo.get_dim('pipe')

    def ckpt_prefix(self, checkpoints_path, tag):
        """Build a prefix for all checkpoint files written by this module. """
        # All checkpoint files start with this
        rank_name = 'module'

        # Data parallelism is omitted from the naming convention because we are agnostic
        # to this in the checkpoint.
        omit_dims = frozenset(['data'])
        axes = [a for a in self._grid._topo.get_axis_names() if a not in omit_dims]
        for dim in axes:
            rank = getattr(self._grid._topo.get_coord(rank=self.global_rank), dim)
            rank_name += f'-{dim}_{rank:02d}'

        ckpt_name = os.path.join(checkpoints_path, str(tag), rank_name)
        return ckpt_name

    def ckpt_layer_path(self, ckpt_dir, local_layer_idx):
        """Customize a prefix for a specific pipeline module layer. """
        idx = local_layer_idx + self._local_start
        layer_ckpt_path = os.path.join(ckpt_dir, f'layer_{idx:02d}')
        rank_repr = self._grid._topo.get_rank_repr(rank=self.global_rank)
        if rank_repr != '':
            layer_ckpt_path += f'-{rank_repr}'
        layer_ckpt_path += '-model_states.pt'
        return layer_ckpt_path

    def ckpt_layer_path_list(self, ckpt_dir, local_layer_idx):
        """Get all ckpt file list for a specific pipeline module layer. """
        idx = local_layer_idx + self._local_start
        layer_ckpt_path = os.path.join(ckpt_dir, f'layer_{idx:02d}-')
        layer_ckpt_path += "*model_states.pt"
        ckpt_files = glob.glob(layer_ckpt_path)
        ckpt_files.sort()
        return ckpt_files

    def save_state_dict(self, save_dir, checkpoint_engine):
        # Processes having the same model parallel rank on different data parallel instances
        # have identical layer weights.  We can distribute the task of saving the layer weights
        # among the data parallel ranks.  For example, if a pipeline stage has 9 layers and
        # if there are 2 data parallel instances, rank 0 will save the first 5 layers and
        # rank 1 will save the last 4.
        dp_rank = self._grid.data_parallel_id
        dp_size = self._grid.data_parallel_size
        num_layers = len(self.forward_funcs)
        if self.checkpoint_parallel_write_pipeline:
            # spread layers evenly across data parallel ranks
            offsets = ds_utils.partition_uniform(num_layers, dp_size)
            start, end = offsets[dp_rank], offsets[dp_rank + 1]
        else:
            # data parallel rank 0 writes all layers
            if dp_rank != 0:
                return
            start, end = 0, num_layers
        layer_list = self.forward_funcs[start:end]

        os.makedirs(save_dir, exist_ok=True)
        for idx, layer in enumerate(layer_list):
            model_ckpt_path = self.ckpt_layer_path(save_dir, start + idx)
            if not hasattr(layer, 'state_dict'):
                continue
            # We pass cloned tensors to torch.save() to avoid checkpoint bloat which occurs because torch.save()
            # saves the underlying storage rather than the slice of the storage corresponding to individual tensors.
            # This is a problem in DeepSpeed because we often allocate tensors using slices of large flattened buffers.
            # Tensor cloning helps to avoid this problem because the storage of cloned tensors are closer to the true size.
            # It is expected that the garbage collector will reclaim the cloned tensor storage to avoid memory bloat.
            # See https://pytorch.org/docs/stable/notes/serialization.html#preserve-storage-sharing
            orig_state_dict = layer.state_dict()
            final_state_dict = type(orig_state_dict)(
                {k: v.clone()
                 for k,
                 v in orig_state_dict.items()})
            checkpoint_engine.save(final_state_dict, model_ckpt_path)

    def load_state_dir(self, load_dir, checkpoint_engine, strict=True):
        for idx, layer in enumerate(self.forward_funcs):
            # Functions, etc. will not have state_dicts
            if not hasattr(layer, 'load_state_dict'):
                continue

            # get all checkpoint files for the layer.
            model_ckpt_list = self.ckpt_layer_path_list(load_dir, idx)
            mp_rank = self._grid.get_slice_parallel_rank()
            mp_world_size = self._grid.get_slice_parallel_world_size()

            sd_loader = SDLoaderFactory.get_sd_loader(
                model_ckpt_list,
                version=2.0,
                checkpoint_engine=checkpoint_engine)
            load_path, checkpoint, _ = sd_loader.load(mp_world_size, mp_rank, module_key=None, is_pipe_parallel=True)

            layer.load_state_dict(checkpoint)

            # if self._grid.data_parallel_id == 0:
            #     logger.info(
            #         f'RANK={self.global_rank} Loaded layer={idx+self._local_start} file={load_path}'
            #     )

        self._synchronize_tied_weights()

    def _is_checkpointable(self, funcs):
        # This is an unfortunate hack related to torch and deepspeed activation checkpoint implementations.
        # Some layers like torch.nn.Embedding will not receive grads if checkpointed, which breaks things.
        # I presume it's related to the discrete inputs that cannot require_grad? Need to revisit.
        if self.__class__.__name__ in ('GPTModelPipe', 'GPT2ModelPipe'):
            return all('ParallelTransformerLayerPipe' in f.__class__.__name__
                       for f in funcs)
        if self.checkpointable_layers is not None:
            return all(f.__class__.__name__ in self.checkpointable_layers for f in funcs)

        params = [f.parameters() for f in funcs if isinstance(f, torch.nn.Module)]
        return any(len(list(p)) > 0 for p in params)
