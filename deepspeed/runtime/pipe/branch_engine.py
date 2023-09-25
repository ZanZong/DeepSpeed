# Copyright 2019 The Microsoft DeepSpeed Team

from types import MethodType

import torch
from deepspeed import comm as dist

from deepspeed.utils import logger
from deepspeed.utils.timer import ThroughputTimer

from ..engine import DeepSpeedEngine, MEMORY_OPT_ALLREDUCE_SIZE
from ..utils import PartitionedTensor
from ..dataloader import RepeatingLoader

from .module import PipelineModule, PipelineError
from .branch_module import PipelineBranchModule
from . import p2p
from . import schedule
from . import flex_schedule
from .. import utils as ds_utils
import time

TARGET_ID = -2
LOG_STAGE = -2
DATA_PARALLEL_ID = -2


def is_even(number):
    return number % 2 == 0


mem_alloced = 0
mem_cached = 0


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


class PipelineBranchEngine(DeepSpeedEngine):
    """ A training engine hybrid pipeline, data, and model parallel training.

    This engine is created by ``deepspeed.initialize()`` when a :class:`PipelineModule`
    is provided.
    """
    ID_TO_DTYPE = [
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
        torch.float16,
        torch.bfloat16,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool
    ]
    DTYPE_TO_ID = {dtype: id_ for id_, dtype in enumerate(ID_TO_DTYPE)}

    def __init__(self, has_bool_tensors=False, *super_args, **super_kwargs):
        super().__init__(*super_args, **super_kwargs)
        assert isinstance(self.module, PipelineBranchModule), \
            "model must be PipelineBranchModule"

        assert self.zero_optimization_stage() < 2, "ZeRO-2 and ZeRO-3 are incompatible with pipeline parallelism"

        # We schedule the all-reduces, so disable it in super().backward()
        self.enable_backward_allreduce = False
        self.has_bool_tensors = has_bool_tensors
        self.eval_return_logits = False
        self.outputs = None

        # used to disable the pipeline all-reduce when used with 1-bit Adam/1-bit LAMB
        self.pipeline_enable_backward_allreduce = True

        if self.elasticity_enabled():
            if not self.is_elastic_model_parallel_supported():
                assert not self.elasticity_enabled(), "Elasticity is not currently supported" \
                " with pipeline parallelism."

        # pipeline step for logging
        self.log_batch_step_id = -1

        self.micro_batch_size = self.train_micro_batch_size_per_gpu()
        self.micro_batches = self.gradient_accumulation_steps()

        # Set Grid and Communication Groups
        self.grid = self.module._grid
        if self.grid.get_global_rank() == 0:
            logger.info(f'CONFIG: micro_batches={self.micro_batches} '
                        f'micro_batch_size={self.micro_batch_size}')

        self.global_rank = self.grid.get_global_rank()

        assert self.dp_world_size == self.grid.data_parallel_size
        assert self.train_batch_size() == \
            self.micro_batch_size * self.micro_batches * self.grid.data_parallel_size

        #  Set Stage Inf
        self.num_stages = self.grid.pipe_parallel_size
        self.stage_id = self.grid.get_stage_id()
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

        self.data_iterator = None
        self.batch_fn = None

        self._force_grad_boundary = False

        self.batch_timer = ThroughputTimer(batch_size=self.micro_batch_size *
                                           self.micro_batches,
                                           num_workers=self.dp_world_size,
                                           logging_fn=self.tput_log,
                                           monitor_memory=False,
                                           steps_per_output=self.steps_per_print())

        # PipelineEngine needs to handle data loading specially due to only the first
        # and last stages loading inputs/labels. We construct a sampler that uses
        if self.training_data:
            self._build_data_iter(self.training_data)

        self.is_pipe_parallel = self.grid.pipe_parallel_size > 1
        self.is_data_parallel = self.grid.data_parallel_size > 1
        self.is_model_parallel = self.grid.model_parallel_size > 1

        # Partition input/output buffers
        # XXX temporarily disable while I revert some partition hacks.
        self.is_pipe_partitioned = self.is_model_parallel
        self.is_grad_partitioned = self.is_model_parallel

        model_parameters = filter(lambda p: p.requires_grad, self.module.parameters())
        num_params = sum([p.numel() for p in model_parameters])
        unique_params = num_params
        # Subtract tied parameters if we don't own them
        if self.module.tied_comms:
            tied_params = 0
            for key, d in self.module.tied_comms.items():
                if self.global_rank != min(d['ranks']):
                    tied_params += sum(p.numel() for p in d['module'].parameters())
            unique_params -= tied_params
        params_tensor = torch.LongTensor(data=[num_params,
                                               unique_params]).to(self.device)
        dist.all_reduce(params_tensor, group=self.grid.get_model_parallel_group())
        params_tensor = params_tensor.tolist()
        total_params = params_tensor[0]
        unique_params = params_tensor[1]
        if self.grid.data_parallel_id == 0:
            logger.info(f'RANK={self.global_rank} '
                        f'STAGE={self.stage_id} '
                        f'LAYERS={self.module._local_stop - self.module._local_start} '
                        f'[{self.module._local_start}, {self.module._local_stop}) '
                        f'STAGE_PARAMS={num_params} ({num_params/1e6:0.3f}M) '
                        f'TOTAL_PARAMS={total_params} ({total_params/1e6:0.3f}M) '
                        f'UNIQUE_PARAMS={unique_params} ({unique_params/1e6:0.3f}M)')

        #initialize peer-2-peer communication and allreduce groups
        if self.is_pipe_parallel:
            p2p.init_process_groups(self.grid)

        # Pipeline buffers
        self.num_pipe_buffers = 0
        self.pipe_buffers = {
            # 'inputs' : [],   # batch input and received activations
            'labels' : [],   # labels from batch input
            # 'outputs' : [],  # activations
            'output_tensors' : [], # tensor object to preserve backward graph
        }
        
        self.meta_buffer = None
        # Identify whether is the first time to comm.
        self.first_output_send_recorder = set()
        self.first_grad_recv_recoder = set()
        if self.module.timeline_path is not None:
            ds_utils.init_timeline_logger(self.module.timeline_path)
            self.timeline_skip = 6
            self.timeline_dur = 2
        self.curr_batch_num = 0

        #stores the loss for the current micro batch being processed
        self.loss = torch.tensor(0.0).to(self.device)

        #stores the loss for the entire batch
        self.total_loss = None
        self.agg_loss = torch.tensor(0.0, requires_grad=False).to(self.device)
        self.dp_group_loss = torch.tensor(0.0, requires_grad=False).to(self.device)

        if self._config.pipeline['activation_checkpoint_interval'] > 0:
            self.module.activation_checkpoint_interval = self._config.pipeline[
                'activation_checkpoint_interval']

        self.module.checkpoint_parallel_write_pipeline = self._config.checkpoint_parallel_write_pipeline

        if self.is_last_stage():
            self.loss_model = self.module.loss_fn

        self.has_attention_mask = self.module.__class__.__name__ == 'GPT2ModelPipe'
        # Initialize pipeline communicators. Just send a 0.
        if is_even(self.stage_id):
            if not self.is_last_stage():
                p2p.send(self.loss, self.next_stage)
            if not self.is_first_stage():
                p2p.recv(self.loss, self.prev_stage)
        else:
            if not self.is_first_stage():
                p2p.recv(self.loss, self.prev_stage)
            if not self.is_last_stage():
                p2p.send(self.loss, self.next_stage)

        # XXX look into timer reporting timing
        # Initialize some timers because of early weirdness.
        if self.wall_clock_breakdown():
            self.timers('forward_microstep').start()
            self.timers('forward_microstep').stop()
            self.timers('backward_microstep').start()
            self.timers('backward_microstep').stop()
            self.timers('backward_inner_microstep').start()
            self.timers('backward_inner_microstep').stop()
            self.timers('backward_allreduce_microstep').start()
            self.timers('backward_allreduce_microstep').stop()
            self.timers('backward_allreduce').start()
            self.timers('backward_allreduce').stop()
            self.timers('step_microstep').start()
            self.timers('step_microstep').stop()

    def set_has_attention_mask(self, value):
        assert isinstance(value, bool)
        self.has_attention_mask = value

    def _build_data_iter(self, dataset):
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=self.dp_world_size,
            rank=self.mpu.get_data_parallel_rank(),
            shuffle=False)
        # Build a loader and make it repeating.
        pipe_dataloader = self.deepspeed_io(dataset, data_sampler=sampler)
        pipe_dataloader = RepeatingLoader(pipe_dataloader)
        self.set_dataloader(pipe_dataloader)

    def _exec_reduce_tied_grads(self):
        # We need to run this first to write to self.averaged_gradients;
        # since this class turns `enable_backward_allreduce` off,
        # `self.overlapping_partition_gradients_reduce_epilogue()` defined in the DeepSpeedEngine
        # never actually runs. I suspect this is because of efficiency problems; get_flat_partition in
        # stage2.py might do something expensive; someone will have to look into that later. But
        # in the meantime, this fixes ZeRO2 + Pipelining enough to run a demo. Further profiling
        # needed to decide if it actually breaks everything.
        # (see https://github.com/EleutherAI/gpt-neox/issues/62#issuecomment-761471944)
        if self.zero_optimization_partition_gradients():
            self.optimizer.overlapping_partition_gradients_reduce_epilogue()

        weight_group_list = self.module.get_tied_weights_and_groups()
        for weight, group in weight_group_list:
            grad = weight._hp_grad if self.bfloat16_enabled() else weight.grad
            dist.all_reduce(grad, group=group)

    def _exec_reduce_grads(self):
        self._force_grad_boundary = True
        if self.pipeline_enable_backward_allreduce:
            if self.bfloat16_enabled():
                if self.zero_optimization_stage() == 0:
                    self._bf16_reduce_grads()
                else:
                    assert self.zero_optimization_stage() == 1, "only bf16 + z1 are supported"
                    raise NotImplementedError()
            else:
                self.allreduce_gradients(bucket_size=MEMORY_OPT_ALLREDUCE_SIZE)
        self._force_grad_boundary = False

    def _bf16_reduce_grads(self):
        # Make our own list of gradients from the optimizer's FP32 grads
        grads = []
        self.buffered_allreduce_fallback(grads=self.optimizer.get_grads_for_reduction(),
                                         elements_per_buffer=MEMORY_OPT_ALLREDUCE_SIZE)
    
    def _check_if_need_grad(self, tensor):
        # TODO a better way to identify whether the tensor needs grads
        if len(tensor.shape) == 2:
            if tensor.grad_fn is not None:
                return True
            else:
                print(f"No grad_fn or no grad tensor:{tensor}")
                return False
        else:
            return True
    
    def _reserve_pipe_info(self, num_buffers, exec_interrupt):
        """Ensure that each pipeline buffer has at least ``num_buffers`` slots.

        This method only reserves slots and does not allocate tensors.

        Args:
            num_buffers (int): The number of buffers to reserve.
        """
        
        # for key in self.pipe_buffers:
        #     self.pipe_buffers[key].extend([None] * num_buffers)
        print(f"\nCreate {num_buffers} buffers for scheudling.\n")
        self.module.expand_buffers_wrt_sched(num_buffers)
        self.num_pipe_buffers = num_buffers
        self.exec_interrupt = exec_interrupt


    def reset_activation_shape(self):
        """Reset the buffers when the shape of activation and gradient change.
        For example, for curriculum learning that changes the seqlen of each
        sample, we need to call this whenever the seqlen is going to change.
        """
        self.first_output_send_recorder = set()
        self.first_grad_recv_recoder = set()
        # for layer, succes in self.module.output_ref.items():
        #     for succ in succes:
        #         self.first_output_send_recorder[(layer, succ)] = True
        #         self.first_grad_recv_recoder[(succ, layer)] = True
    
        self.meta_buffer = None

    def train_batch(self, data_iter=None):
        """Progress the pipeline to train the next batch of data. The engine will ingest
        ``self.train_batch_size()`` total samples collectively across all workers.


        An iterator that over training data should be provided as an argument
        unless ``deepspeed.initialize()`` was provided a training set. In that event,
        the training data will automatically be read.


        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be pulled
            from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.RepeatingLoader`
            that wraps data loaders to automatically restart upon a ``StopIteration``.

        Args:
            data_iter (Iterator, optional): Iterator of training data.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """
        if not torch._C.is_grad_enabled():
            raise RuntimeError(
                f'train_batch() requires gradients enabled. Use eval_batch() instead.')

        # Curriculum learning could change activation shape
        if self.curriculum_enabled():
            assert not self.curriculum_enabled(), "BranchEngine does not support curriculum."
            new_difficulty = self.curriculum_scheduler.update_difficulty( \
                self.global_steps + 1)
            if self.global_steps == 0 or self.curriculum_scheduler.first_step:
                self.reset_activation_shape()
                self.curriculum_scheduler.first_step = False
            elif new_difficulty != self.curriculum_scheduler.get_difficulty( \
                self.global_steps):
                self.reset_activation_shape()

        if data_iter:
            self.set_dataiterator(data_iter)
                
        self.module.train()
        self.total_loss = None
        self._compute_loss = True
        # Do the work
        self.timers('train_batch').start()
        if self.module.see_baseline_perf:
            sched = flex_schedule.TrainNaiveSchedule(micro_batches=self.micro_batches,
                                        stages=self.num_stages,
                                        stage_id=self.stage_id,
                                        stage_read_data_layers=self.module.stage_read_data_layers)
        else:    
            sched = flex_schedule.TrainFlexSchedule(micro_batches=self.micro_batches,
                                        stages=self.num_stages,
                                        stage_id=self.stage_id,
                                        stage_read_data_layers=self.module.stage_read_data_layers)
        self._exec_schedule(sched)
        self.agg_train_loss = self._aggregate_total_loss()
        self.curr_batch_num += 1
        self.timers('train_batch').stop()
        torch.distributed.barrier()
        
        if self.wall_clock_breakdown():
            self.timers.log([
                'pipe_send_output',
                'pipe_send_grad',
                'pipe_recv_input',
                # 'pipe_recv_input_tensor',
                'pipe_recv_grad',
                'batch_input_iter',
                'forward_microstep',
                'backward_microstep',
                # 'backward_inner',
                'backward_allreduce',
                'step'
            ], ranks=[self.global_rank])
            
        if self.global_steps % self.steps_per_print() == 0:
            if self.global_rank == 0:
                elapsed = self.timers('train_batch').elapsed(reset=True) / 1000.0
                iter_time = elapsed / self.steps_per_print()
                tput = self.train_batch_size() / iter_time
                print(f'[^^] steps: {self.global_steps} '
                      f'loss: {self.agg_train_loss:0.4f} '
                      f'iter time (s): {iter_time:0.3f} '
                      f'samples/sec: {tput:0.3f}, bs={self.train_batch_size()} [^^]\n')
        ds_utils.see_memory_usage(f"Rank={self.local_rank}, after batch", True)
        
        # Monitoring
        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [(f'Train/Samples/train_loss',
                                    self.agg_train_loss.mean().item(),
                                    self.global_samples)]
            self.monitor.write_events(self.summary_events)
        
        
        

        # TODO: should return precisely what loss returned and allow others to be queried?
        return self.agg_train_loss

    def eval_batch(self,
                   data_iter,
                   return_logits=False,
                   compute_loss=True,
                   reduce_output='avg'):
        """Evaluate the pipeline on a batch of data from ``data_iter``. The
        engine will evaluate ``self.train_batch_size()`` total samples
        collectively across all workers.

        This method is equivalent to:

        .. code-block:: python

            module.eval()
            with torch.no_grad():
                output = module(batch)

        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be pulled
            from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.RepeatingLoader`
            that wraps data loaders to automatically restart upon a ``StopIteration``.

        Args:
            data_iter (Iterator): Iterator of data to evaluate.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """
        self.eval_return_logits = return_logits
        self.module.eval()

        # Curriculum learning could change activation shape
        if self.curriculum_enabled():
            assert not self.curriculum_enabled(), "BranchEngine does not support curriculum."
            new_difficulty = self.curriculum_scheduler.update_difficulty( \
                self.global_steps + 1)
            if self.global_steps == 0 or self.curriculum_scheduler.first_step:
                self.reset_activation_shape()
                self.curriculum_scheduler.first_step = False
            elif new_difficulty != self.curriculum_scheduler.get_difficulty( \
                self.global_steps):
                self.reset_activation_shape()

        eval_output = None

        self._compute_loss = compute_loss

        # Use the provided data iterator
        train_iterator = self.data_iterator
        self.set_dataiterator(data_iter)

        # Do the work
        sched = schedule.InferenceSchedule(micro_batches=self.micro_batches,
                                           stages=self.num_stages,
                                           stage_id=self.stage_id)

        # prevent dead-lock with multiple evals sequence
        dist.barrier()

        with torch.no_grad():
            self._exec_schedule(sched)

        if self.is_last_stage():
            eval_output = self._reduce_outputs(self.fwd_outputs, reduce=reduce_output)

        if compute_loss:
            eval_output = self._bcast_pipe_scalar(eval_output)

        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [(f'Train/Samples/eval_loss',
                                    eval_output.mean().item(),
                                    self.global_samples)]
            self.monitor.write_events(self.summary_events)

        # Restore the training iterator
        self.set_dataiterator(train_iterator)

        # Reset any buffers that may have been populated during the forward passes.
        #ds_checkpointing.reset()
        self.eval_return_logits = False
        if return_logits:
            outputs = self.outputs
            self.outputs = None
            return eval_output, outputs
        return eval_output

    def set_train_batch_size(self, train_batch_size):
        """Adjust the global batch size by increasing or decreasing the number of
        micro-batches (i.e., gradient accumulation steps). The size of each micro-batch
        (i.e., ``train_micro_batch_size_per_gpu``) is not changed.
        Args:
            train_batch_size (int): The new global batch size for training.
        Raises:
            ValueError: if ``train_batch_size`` is not divisible by the
                configured micro-batch size and data parallelism.
        """
        super().set_train_batch_size(train_batch_size)
        self.micro_batches = self.gradient_accumulation_steps()

    def is_first_stage(self):
        """True if this process is in the first stage in the pipeline."""
        return self.stage_id == 0

    def is_last_stage(self):
        """True if this process is in the last stage in the pipeline."""
        return self.stage_id == self.num_stages - 1

    def _reduce_outputs(self, outputs, reduce='avg', reduce_dp=True):
        if reduce is None:
            return outputs

        if reduce.lower() == 'avg':
            # first sum over all microbatches
            if torch.is_tensor(outputs[0]):
                reduced = sum(outputs)
            else:
                assert isinstance(outputs, (list, tuple))
                reduced = [torch.zeros_like(o) for o in outputs[0]]
                for idx, out in outputs:
                    reduced[idx] += out

            # Average over the microbatches
            reduced = self._scale_loss_by_gas(reduced)

            # Average over DP groups
            if reduce_dp and self.is_data_parallel:
                if torch.is_tensor(reduced):
                    dist.all_reduce(reduced, group=self.mpu.get_data_parallel_group())
                    reduced /= self.dp_world_size
                else:
                    for idx in range(len(reduced)):
                        dist.all_reduce(reduced[idx],
                                        group=self.mpu.get_data_parallel_group())
                        reduced[idx] /= self.dp_world_size

            return reduced
        else:
            raise NotImplementedError(f'reduction type {reduce} not supported.')

    def _bcast_pipe_scalar(self, data, src_rank=None, dtype=torch.float32):
        # Default to last stage (e.g., for broadcasting loss)
        if src_rank is None:
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
        assert src_rank in self.grid.pp_group

        if self.global_rank == src_rank:
            result = data.clone().detach()
        else:
            result = torch.Tensor([0.]).type(dtype).to(self.device)

        dist.broadcast(tensor=result,
                       src=src_rank,
                       group=self.mpu.get_pipe_parallel_group())

        return result

    def _aggregate_total_loss(self):
        # Scale loss, average among DP ranks, and bcast loss to the rest of my DP group
        if self.is_last_stage():
            loss = self._scale_loss_by_gas(self.total_loss)
            self.dp_group_loss = loss.clone().detach()

            ## Average loss across all data-parallel groups
            agg_loss = self.dp_group_loss.clone().detach()
            #print(f'RANK={self.global_rank} bcast SENDER src={self.global_rank} group={self.grid.pp_group}', flush=True)
            if self.is_data_parallel:
                dist.all_reduce(agg_loss, group=self.mpu.get_data_parallel_group())
                agg_loss /= self.dp_world_size

            assert self.global_rank in self.grid.pp_group
            losses = torch.Tensor([self.dp_group_loss, agg_loss]).to(self.device)
            dist.broadcast(tensor=losses,
                           src=self.global_rank,
                           group=self.mpu.get_pipe_parallel_group())

        else:
            # Get loss from last stage
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
            assert src_rank in self.grid.pp_group
            losses = torch.Tensor([0., 0.]).to(self.device)
            dist.broadcast(tensor=losses,
                           src=src_rank,
                           group=self.grid.get_pipe_parallel_group())
            self.dp_group_loss = losses[0].clone().detach()
            agg_loss = losses[1].clone().detach()

        return agg_loss

    def set_dataloader(self, loader):
        """"""
        if len(self.module.stage_read_data_layers[self.stage_id]) > 0:
            self.training_dataloader = loader
            self.data_iterator = iter(self.training_dataloader)

    def set_dataiterator(self, iterator):
        """ Store an iterator to sample for training data. """
        if len(self.module.stage_read_data_layers[self.stage_id]) > 0:
            self.training_dataloader = None
            self.data_iterator = iterator

    def set_batch_fn(self, fn):
        """Execute a post-processing function on input data.

        Args:
            fn (function): The function to run.
        """
        self.batch_fn = fn

    def is_gradient_accumulation_boundary(self):
        """True if the engine is executing a gradient reduction or optimizer step instruction.

        This is overridden from :class:`DeepSpeedEngine` to force reductions
        and steps when the pipeline engine is instructed to do so.

        Returns:
            bool: whether reductions and optimizer steps should occur.
        """
        return self._force_grad_boundary

    def log_for_device(self, *msg):
        if LOG_STAGE == self.stage_id or LOG_STAGE == -1:
            if DATA_PARALLEL_ID == self.grid.data_parallel_id or DATA_PARALLEL_ID == -1:
                print(
                    f'RANK={dist.get_rank()} '
                    f'PIPE-ID={self.stage_id} '
                    f'DATA-ID={self.grid.data_parallel_id} '
                    f'MBATCH-ID={self.microbatch_id} '
                    f'STEP-ID={self.log_batch_step_id} '
                    '::',
                    *msg,
                    flush=True)

    def tput_log(self, *msg):
        if self.global_rank == 0 and self.global_steps % self.steps_per_print() == 0:
            print(*msg)

    def _next_batch(self):
        # If using 3D parallelism, only some first-stage ranks may do IO
        batch = None
        if self.data_iterator is not None:
            batch = next(self.data_iterator)

        # Any post-processing, like broadcasting across a slice-parallel group.
        if self.batch_fn:
            batch = self.batch_fn(batch)

        return batch
    
    
    def count_input_num(self, layer_id, predecessor_id):
        if predecessor_id == -1:
            # read input data set
            return 1
        else:
            return len(self.module.layer_inputs[layer_id][predecessor_id])

    def _exec_forward_pass(self, buffer_id, start, end):
        assert start <= end
        self.tput_timer.start()
        if self.wall_clock_breakdown():
            self.timers('forward_microstep').start()
        self.mem_status('BEFORE FWD', reset_max=True)
        
        if self.stage_id in self.exec_interrupt.keys() and \
                end in self.exec_interrupt[self.stage_id]:
            graph_tail = True
        else:
            graph_tail = False
        outputs = self.module.partial_forward(start, end + 1, buffer_id, graph_tail)
        # ds_utils.see_memory_usage(f"Rank={self.local_rank}, after partial forward layers=[{start},{end}]", True)
        '''
        ###### DEBUG Memory ######
        if self.local_rank == 0:
            # ds_utils.see_memory_usage(f"Rank={self.local_rank}, before partial forward layers=[{start},{end}]", True)    
            outputs = self.module.partial_forward(start, end + 1, buffer_id, graph_tail)
            ds_utils.see_memory_usage(f"Rank={self.local_rank}, after partial forward layers=[{start},{end}]", True)
            del self.module.layer_output_buffer
            # for layer in self.module.layer_output_buffer.keys():
            #     for buffer_id in range(self.num_pipe_buffers):
            #         if isinstance(self.module.layer_output_buffer[layer][buffer_id], torch.Tensor):
            #             del self.module.layer_output_buffer[layer][buffer_id]
            #         else:
            #             for t in self.module.layer_output_buffer[layer][buffer_id]:
            #                 del t
            
            torch.autograd.backward(tensors=(torch.sum(outputs), ))
            ds_utils.see_memory_usage(f"Rank={self.local_rank}, after backward layers=[{start},{end}]", True)
            
            # ds_utils.see_memory_usage(f"Rank={self.local_rank}, before naive forward layers=[{start},{end}]", True)    
            # outputs = self.module.naive_forward(start, end + 1, buffer_id, graph_tail)
            # ds_utils.see_memory_usage(f"Rank={self.local_rank}, after naive forward layers=[{start},{end}]", True)        
            # torch.autograd.backward(tensors=(torch.sum(outputs), ))
            # ds_utils.see_memory_usage(f"Rank={self.local_rank}, after backward layers=[{start},{end}]", True)
            
        if self.local_rank == 1:
            outputs = self.module.partial_forward(start, end + 1, buffer_id, graph_tail)
        exit()
        '''
        
        # Compute loss after the last layer
        if end + 1 == self.module._num_layers:
            if self._compute_loss and self.module.loss_fn is not None:
                # TODO: handle various loss calcs.
                print("TODO: handle various loss calcs.")
                # labels = self.pipe_buffers['labels'][buffer_id]
                # self.loss = self.module.loss_fn(outputs, labels)
            else:
                # Some models just return loss from forward()
                self.loss = outputs
            if self.eval_return_logits:
                self.outputs = outputs
            if isinstance(self.loss, torch.Tensor):
                self.fwd_outputs.append(self.loss.detach())
                if self.total_loss is None:
                    self.total_loss = torch.zeros_like(self.loss)
                self.total_loss += self.loss.detach()
            else:
                raise Exception("Only suppot single tensor loss")
                # TODO: handle various loss calcs.
                self.fwd_outputs.append([l.detach() for l in self.loss])
                if self.total_loss is None:
                    self.total_loss = [torch.zeros_like(l) for l in self.loss]
                for idx, l in enumerate(self.loss):
                    self.total_loss[idx] += l.detach()
        print(f"finish forward pass {start}, {end}")
        if self.wall_clock_breakdown():
            self.timers('forward_microstep').stop()
            
    def _exec_backward_pass(self, buffer_id, left, right):
        """" Layers between the index of `left` and `right` will be backwarded and these layers must
             be a consecutive slice. The backward will stop after calculationg grads of `left` layer's input tensors."""
        assert self.optimizer is not None, "must provide optimizer during " \
                                           "init in order to use backward"
        assert right >= left
        self.mem_status('BEFORE BWD', reset_max=True)
        if self.wall_clock_breakdown():
            self.timers('backward_microstep').start()
            # self.timers('backward').start()
            # self.timers('backward_inner_microstep').start()
            # self.timers('backward_inner').start()

        if self.bfloat16_enabled() and not self.is_last_stage():
            # manually call because we don't call optimizer.backward()
            self.optimizer.clear_lp_grads()
        
        def clear_output_buffer_before_backward():
            """ Output buffers of each layer need to be cleaned before calling `backward`.
                Without cleaning buffers will cause a memory occupation even activation tensors
                are consumed by the backward pass, which leads to OOM after several training 
                iterations.
            """
            for layer_idx in range(left, right + 1):
                outtensor_refs = self.module.layer_output_buffer[layer_idx]
                del outtensor_refs[buffer_id]
                outtensor_refs[buffer_id] = None
            right_mod = self.module.get_layer_module(right)
            if hasattr(right_mod, 'tail_outputs'):
                del right_mod.tail_outputs[buffer_id]
                right_mod.tail_outputs[buffer_id] = None
                
        # print(f"Call backward index [{left}, {right}]")
        
        # if contain the loss layer.
        if right + 1 == self.module._num_layers:
            torch.autograd.backward(tensors=self.loss)    
            if self.wall_clock_breakdown():
                self.timers('backward_microstep').stop()
            return
        
        right_mod = self.module.get_layer_module(right)
        out_tensors = self.module.layer_output_buffer[right][buffer_id]
        if isinstance(out_tensors, torch.Tensor):
            if hasattr(right_mod, 'grad_layer'):
                # Case 1: received grad
                single_succ = list(right_mod.grad_layer.keys())[0]
                grad_tensor = right_mod.grad_layer[single_succ][buffer_id]
            elif hasattr(right_mod, 'tail_outputs'):
                # Case 2: grad from next interrupted graph's input buffer
                grad_tensor = right_mod.tail_outputs[buffer_id].grad
            else:
                print(f"ERR: Backward right bound {right} doesn't have grad_layer for receiving " +
                      "grad or tail_outputs for feeding new graph. Partial backward can only among two cases.")
            # clear_output_buffer_before_backward()
            torch.autograd.backward(tensors=(out_tensors, ), grad_tensors=(grad_tensor, ))
            print(f"backward done for [{left}, {right}]")
        else:
            tensors = list()
            grad_tensors = list()
            for idx, t in enumerate(out_tensors):
                if t.is_floating_point() and t.requires_grad and len(t.shape) != 1:
                    # Get real tensor
                    tensors.append(t)
                    # Get corresponding grad.
                    if hasattr(right_mod, 'grad_layer'):
                        # Case 1: received grad
                        t2layer = self.module.query_output_successors(right)
                        succ = t2layer[idx][0]
                        grad_tensors.append(right_mod.grad_layer[succ][buffer_id][idx])
                    elif hasattr(right_mod, 'tail_outputs'):
                        # Case 2: grad from next interrupted graph's input buffer
                        grad_tensors.append(right_mod.tail_outputs[buffer_id][idx].grad)
                    else:
                        print(f"ERR: Backward right bound {right} doesn't have grad_layer for receiving " +
                      "grad or tail_outputs for feeding new graph.")
                        
            assert len(tensors) == len(grad_tensors), f"out tensors={len(tensors)} don't have enough grads={len(grad_tensors)}"
            # clear_output_buffer_before_backward()
            torch.autograd.backward(tensors=tuple(tensors), grad_tensors=tuple(grad_tensors))
        # ds_utils.see_memory_usage(f"Rank={self.local_rank}, Backward [{left}, {right}] done", True)
        
        if self.bfloat16_enabled() and not self.is_last_stage():
            # manually call because we don't call optimizer.backward()
            self.optimizer.update_hp_grads(clear_lp_grads=False)

        if self.wall_clock_breakdown():
            #self.timers('backward_inner').stop()
            # self.timers('backward_inner_microstep').stop()
            # self.timers('backward').stop()
            self.timers('backward_microstep').stop()

        self.mem_status('AFTER BWD')

    def _exec_load_micro_batch(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('batch_input_iter').start()
        
        batch = self._next_batch()
        if len(self.module.stage_read_data_layers[self.stage_id]) == 0:
            print(f"Rank={self.global_rank}, stage {self.stage_id} doesn't need train data, but _exec_load_micro_batch is called!")
            pass
        else:
            data_start = sum([len(self.module.stage_read_data_layers[s]) for s in range(self.stage_id)])
            for i, layer_id in enumerate(self.module.stage_read_data_layers[self.stage_id]):
                cur_data_index = data_start + i
                # print(f"put data index {cur_data_index} to layer {layer_id}'s buffer {buffer_id}")
                loaded = batch[cur_data_index].to(self.local_rank).clone().detach()
                # print("after loaded")
                # loaded.requires_grad = loaded.is_floating_point()
                self.module.get_layer_module(layer_id).input_buffers[-1][buffer_id] = loaded
        if self.wall_clock_breakdown():
            self.timers('batch_input_iter').stop()

    def _send_tensor_meta(self, buffer, recv_stage):
        """ Communicate metadata about upcoming p2p transfers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape
        """
        send_bytes = 0
        if isinstance(buffer, torch.Tensor):
            type_tensor = torch.LongTensor(data=[0]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            send_shape = torch.LongTensor(data=buffer.size()).to(self.device)
            send_ndims = torch.LongTensor(data=[len(buffer.size())]).to(self.device)
            p2p.send(send_ndims, recv_stage)
            p2p.send(send_shape, recv_stage)
            send_bytes += _tensor_bytes(buffer)
        elif isinstance(buffer, list):
            assert (False)
            type_tensor = torch.LongTensor(data=[1]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            count_tensor = torch.LongTensor(data=[len(buffer)]).to(self.device)
            p2p.send(count_tensor, recv_stage)
            for tensor in buffer:
                assert isinstance(tensor, torch.Tensor)
                send_shape = torch.LongTensor(data=tensor.size()).to(self.device)
                send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(self.device)
                p2p.send(send_ndims, recv_stage)
                p2p.send(send_shape, recv_stage)
                send_bytes += _tensor_bytes(tensor)
        elif isinstance(buffer, tuple):
            type_tensor = torch.LongTensor(data=[2]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            count_tensor = torch.LongTensor(data=[len(buffer)]).to(self.device)
            p2p.send(count_tensor, recv_stage)
            for idx, tensor in enumerate(buffer):
                assert isinstance(tensor, torch.Tensor), f"Tensor not valid: {tensor}"
                send_shape = torch.LongTensor(data=tensor.size()).to(self.device)
                send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(self.device)
                send_dtype = torch.LongTensor(data=[self.DTYPE_TO_ID[tensor.dtype]]).to(
                    self.device)
                p2p.send(send_dtype, recv_stage)
                p2p.send(send_ndims, recv_stage)
                p2p.send(send_shape, recv_stage)
                # Useful for performance debugging.
                '''
                new_bytes = _tensor_bytes(tensor)
                send_bytes += _tensor_bytes(tensor)
                # Useful for performance debugging.
                if self.grid.data_parallel_id == 0:
                    print(
                        f'STAGE={self.stage_id} pipe-send-volume[{idx}]: shape={send_shape} {new_bytes/1024**2:0.2f}MB'
                    )
                '''
        else:
            raise NotImplementedError(f'Could not send meta type {type(buffer)}')

        # Useful for performance debugging.
        '''
        if self.grid.data_parallel_id == 0:
            print(f'STAGE={self.stage_id} pipe-send-volume: {send_bytes/1024**2:0.2f}MB')
        '''

    def _recv_tensor_meta(self, send_stage, buffer_num=None):
        """Receive metadata about upcoming p2p transfers and return allocated buffers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape

        Returns:
            Allocated buffer for receiving from send_stage.
        """

        type_tensor = torch.LongTensor(data=[0]).to(self.device)
        p2p.recv(type_tensor, send_stage)
        recv_type = type_tensor.item()

        # A single tensor will be sent.
        if recv_type == 0:
            recv_ndims = torch.LongTensor(data=[0]).to(self.device)
            p2p.recv(recv_ndims, send_stage)
            recv_ndims = recv_ndims.item()
            recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
            p2p.recv(recv_shape, send_stage)
            recv_shape = recv_shape.tolist()
            if buffer_num is not None and buffer_num > 0:
                return [self._allocate_buffer(recv_shape, num_buffers=1)[0] for i in range(buffer_num)]
            else:
                return self._allocate_buffer(recv_shape, num_buffers=1)[0]

        # List or tuple of tensors
        elif recv_type == 1 or recv_type == 2:
            count_tensor = torch.LongTensor(data=[0]).to(self.device)
            p2p.recv(count_tensor, send_stage)
            num_tensors = count_tensor.item()
            recv_shapes_and_dtypes = []
            for idx in range(num_tensors):
                recv_dtype = torch.LongTensor(data=[0]).to(self.device)
                p2p.recv(recv_dtype, send_stage)
                recv_dtype = self.ID_TO_DTYPE[recv_dtype.item()]
                recv_ndims = torch.LongTensor(data=[0]).to(self.device)
                p2p.recv(recv_ndims, send_stage)
                recv_ndims = recv_ndims.item()
                recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
                p2p.recv(recv_shape, send_stage)
                recv_shapes_and_dtypes.append((recv_shape.tolist(), recv_dtype))
            
            if buffer_num is not None and buffer_num > 0:
                multi_buffers = [self._allocate_buffers(recv_shapes_and_dtypes, num_buffers=1)[0] for i in range(buffer_num)]
                # Convert to tuples if requested.
                if recv_type == 2:
                    multi_buffers = [tuple(buffers) for buffers in multi_buffers]
                return multi_buffers
            else:
                buffers = self._allocate_buffer(recv_shapes_and_dtypes, num_buffers=1)[0]
                # Convert to tuples if requested.
                if recv_type == 2:
                    buffers = tuple(buffers)
                return buffers
        else:
            raise NotImplementedError(f'Could not receive type {type(recv_type)}')

    def _exec_send_activations(self, buffer_id, send, recv):
        
        if self.wall_clock_breakdown():
            self.timers('pipe_send_output').start()
        outputs = self.module.form_layer_output_to_send(send, recv, buffer_id)
        
        if (send, recv) not in self.first_output_send_recorder:
            self.first_output_send_recorder.add((send, recv))
            self._send_tensor_meta(outputs, self.next_stage)
        
        if isinstance(outputs, torch.Tensor):
            p2p.send(outputs, self.next_stage)
        elif isinstance(outputs, tuple):
            for idx, buffer in enumerate(outputs):
                p2p.send(buffer, self.next_stage)
        else:
            raise NotImplementedError('Could not send output of type '
                                      f'{type(outputs)}')
        print(f"{self.local_rank} send activations, buffer_id={buffer_id}")
        # print(f"DEBUG: send tenosr {outputs.shape} to next stage {self.next_stage}, tensor={outputs}")
        if self.wall_clock_breakdown():
            self.timers('pipe_send_output').stop()
            

    def _exec_recv_activations(self, buffer_id, send, recv):
        """"Current implementation only supports each layer to receive once for each iteration."""
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_input').start()
        
        mod = self.module.get_layer_module(recv)
        # Allocate the buffer if necessary
        if (send, recv) not in self.first_output_send_recorder:
            self.first_output_send_recorder.add((send, recv))
            mod.input_buffers[send] = self._recv_tensor_meta(self.prev_stage, buffer_num=self.num_pipe_buffers)
            # print(f"allocate recv buffer for layer {recv}, buffer_id={buffer_id}, {mod.input_buffers[send]}")
        
        self.timers('pipe_recv_input_tensor').start()
        if isinstance(mod.input_buffers[send][buffer_id], torch.Tensor):
            p2p.recv(mod.input_buffers[send][buffer_id], self.prev_stage)
            # mod.input_buffers[send][buffer_id] = mod.input_buffers[send][buffer_id].clone().detach()
            mod.input_buffers[send][buffer_id].requires_grad = mod.input_buffers[send][buffer_id].is_floating_point()
        else:
            for idx, tuple_buffer in enumerate(mod.input_buffers[send][buffer_id]):
                assert torch.is_tensor(tuple_buffer), f"receiver buffer is not a tensor: {tuple_buffer}"
                # XXX hardcode meta type
                # if self.is_pipe_partitioned and idx == 0 and tuple_buffer.dtype != torch.long:
                #     if self.meta_buffer is None:
                #         self.meta_buffer = torch.zeros(tuple_buffer.size(),
                #                                        dtype=torch.long,
                #                                        device=self.device)
                #     tuple_buffer = self.meta_buffer
                p2p.recv(tuple_buffer, self.prev_stage)
            
            # NCCL does not like to send torch.BoolTensor types, so un-cast the
            # attention mask
            # if self.has_attention_mask or self.has_bool_tensors:
            #     recvd[-1] = recvd[-1].bool()
            for idx, tuple_buffer in enumerate(mod.input_buffers[send][buffer_id]):
                tuple_buffer.requires_grad = tuple_buffer.is_floating_point() and idx < 2 # TODO a better way to identify non-grad tensor, e.g., mask
        # print(f"Recv tensors {mod.input_buffers[send][buffer_id]}")
        print(f"{self.local_rank} recv activations, buffer_id={buffer_id}")
        self.timers('pipe_recv_input_tensor').stop()
        
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_input').stop()

    def _exec_send_grads(self, buffer_id, send, recv):
        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad').start()
        target_stage = self.module.query_stage(recv)
        inputs = self.module.form_layer_input(send, buffer_id, check_graph_tail=True)
        # identify which tensor's grad to send
        if isinstance(inputs, torch.Tensor):
            data_recv = inputs
        else:
            data_recv = []
            data_idx = []
            for out_, in_ in self.module.layer_inputs[send][recv]:
                data_recv.append(inputs[in_])
                data_idx.append(in_)
            data_recv = data_recv[0] if len(data_recv) == 1 else tuple(data_recv)

        if isinstance(data_recv, torch.Tensor):
            assert data_recv.grad is not None
            p2p.send(data_recv.grad, target_stage)
        else:
            for idx, buffer in enumerate(data_recv):
                # Skip tensors that will not produce a grad
                if buffer.is_floating_point() and buffer.requires_grad:
                    p2p.send(buffer.grad, target_stage)
                else:
                    # print(f"[Send gards] Tensor in buffer doesn't have grad:{buffer}, {buffer.shape}")
                    continue
        print(f"Rank {self.local_rank} send grad,  buffer_id={buffer_id}")

        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad').stop()
        
    def _exec_recv_grads(self, buffer_id, send, recv):
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').start()
        layer_mod = self.module.get_layer_module(recv)
        data = self.module.layer_output_buffer[recv][buffer_id]
        # identify which tensor associate with the grad `send` layer.
        if isinstance(data, torch.Tensor):
            data_send = data
        else:
            data_send = []
            data_idx = []
            for out_, in_ in self.module.layer_outputs[recv][send]:
                data_send.append(data[out_])
                data_idx.append(out_)
            data_send = data_send[0] if len(data_send) == 1 else tuple(data_send)
        
        # Create grad buffers according to `num_pipe_buffers`
        if (send, recv) not in self.first_grad_recv_recoder:
            # print(f"First allocate grad layer for {(send, recv)}")
            layer_mod.grad_layer = dict()
            if isinstance(data_send, torch.Tensor):
                s = list(data_send.size())
                sizes_and_dtypes = [(s, data_send.dtype) for i in range(self.num_pipe_buffers)]
                layer_mod.grad_layer[send] = self._allocate_buffers(sizes_and_dtypes, num_buffers=1)[0]
            else:
                sizes_and_dtypes = [(list(t.size()),
                                        t.dtype) for t in data_send
                                        if t.is_floating_point() and t.requires_grad]
                layer_mod.grad_layer[send] = list()
                for i in range(self.num_pipe_buffers):
                    layer_mod.grad_layer[send].append(tuple(self._allocate_buffers(sizes_and_dtypes,
                                                        num_buffers=1)[0]))
            # print(f"create grad buffer for layer {send}, {layer_mod.grad_layer[send]}")
            self.first_grad_recv_recoder.add((send, recv))
        
        if isinstance(data_send, torch.Tensor):
            p2p.recv(layer_mod.grad_layer[send][buffer_id], self.next_stage)
        else:
            for idx, buffer in enumerate(layer_mod.grad_layer[send][buffer_id]):
                # XXX GPT-2 hack
                if self.is_grad_partitioned and idx == 0 and buffer.dtype != torch.long:
                    buffer.data = torch.zeros(buffer.size(),
                                              dtype=torch.long,
                                              device=self.device)
                p2p.recv(buffer, self.next_stage)

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').stop()
        print(f"Rank {self.local_rank} recv grad,  buffer_id={buffer_id}")
         

    def _exec_optimizer_step(self, lr_kwargs=None):
        if self.wall_clock_breakdown():
            self.timers('step_microstep').start()
            self.timers('step').start()
        self.mem_status('BEFORE STEP', reset_max=True)

        self._force_grad_boundary = True
        self._take_model_step(lr_kwargs)
        self._force_grad_boundary = False

        self.mem_status('AFTER STEP')

        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [(f'Train/Samples/lr',
                                    self.get_lr()[0],
                                    self.global_samples)]
            if self.fp16_enabled() and hasattr(self.optimizer, 'cur_scale'):
                self.summary_events.append((f'Train/Samples/loss_scale',
                                            self.optimizer.cur_scale,
                                            self.global_samples))
            self.monitor.write_events(self.summary_events)

        if self.wall_clock_breakdown():
            self.timers('step_microstep').stop()
            self.timers('step').stop()

    def _zero_grads(self, inputs):
        if isinstance(inputs, torch.Tensor):
            if inputs.grad is not None:
                inputs.grad.data.zero_()
        else:
            for t in inputs:
                if t.grad is not None:
                    t.grad.data.zero_()

    def _allocate_zeros(self, shape, **kwargs):
        """ Allocate a tensor of zeros on the engine's device.

        Arguments:
            shape: the shape of the tensor to allocate
            kwargs: passed to torch.zeros()

        Returns:
            A tensor from torch.zeros() allocated on self.device.
        """
        if "dtype" not in kwargs:
            if self.fp16_enabled():
                kwargs["dtype"] = torch.half
            if self.bfloat16_enabled():
                kwargs["dtype"] = torch.bfloat16

        return torch.zeros(shape, device=self.device, **kwargs)

    def _allocate_buffer(self, shape, num_buffers=-1, **kwargs):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_pipe_buffers
        for count in range(num_buffers):
            buffers.append(self._allocate_zeros(shape, **kwargs))
        return buffers

    def _allocate_buffers(self, shapes_and_dtypes, requires_grad=False, num_buffers=-1):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_pipe_buffers
        for count in range(num_buffers):
            buffer = []
            for shape, dtype in shapes_and_dtypes:
                buffer.append(
                    self._allocate_zeros(shape,
                                         dtype=dtype,
                                         requires_grad=requires_grad))
            buffers.append(buffer)
        return buffers

    def forward(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def backward(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def step(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def mem_status(self, msg, print_rank=-1, reset_max=False):
        return
        global mem_alloced, mem_cached
        if not self.global_steps == 0 or not self.global_steps == 9:
            #return
            pass
        if self.mpu.get_data_parallel_rank() != 0:
            return

        if self.global_rank != 0:
            return

        rank = self.global_rank
        if print_rank != -1 and rank != print_rank:
            return

        torch.cuda.synchronize()

        if reset_max:
            torch.cuda.reset_max_memory_cached()
            torch.cuda.reset_max_memory_allocated()

        new_alloced = torch.cuda.memory_allocated()
        new_cached = torch.cuda.memory_cached()

        delta_alloced = new_alloced - mem_alloced
        delta_cached = new_cached - mem_cached

        mem_cached = new_cached
        mem_alloced = new_alloced

        max_alloced = torch.cuda.max_memory_allocated()
        max_cached = torch.cuda.max_memory_cached()

        # convert to GB for printing
        new_alloced /= 1024**3
        new_cached /= 1024**3
        delta_alloced /= 1024**3
        delta_cached /= 1024**3
        max_alloced /= 1024**3
        max_cached /= 1024**3

        print(
            f'RANK={rank} STAGE={self.stage_id} STEP={self.global_steps} MEMSTATS',
            msg,
            f'current alloc={new_alloced:0.4f}GB (delta={delta_alloced:0.4f}GB max={max_alloced:0.4f}GB) '
            f'current cache={new_cached:0.4f}GB (delta={delta_cached:0.4f}GB max={max_cached:0.4f}GB)'
        )

    def module_state_dict(self):
        """Override hack to save a pipe model and return the directory path of the save.

        This method should only be called by DeepSpeed's ``save_checkpoint()``. The
        recommended way of saving a ``PipelineModule`` outside of ``save_checkpoint()``
        is ``save_state_dict()``.

        Returns:
            None
        """
        assert isinstance(self.module, PipelineModule)
        assert self._curr_ckpt_path is not None, \
            "PipelineEngine expects module_state_dict() to be called from save_checkpoint()"

        self.module.save_state_dict(self._curr_ckpt_path,
                                    checkpoint_engine=self.checkpoint_engine)
        return None

    def load_module_state_dict(self, state_dict, strict=True, custom_load_fn=None):
        """Override hack to instead use a directory path.

        This is important because pipeline models checkpoint by layer instead of rank.

        If ``state_dict`` is not ``None`` or a ``str``, we revert to ``super()`` expecting a ``dict``.

        Args:
            state_dict (str, None): unused
            strict (bool, optional): Strict state loading. Defaults to True.
        """
        assert custom_load_fn is None, "custom_load_fn not supported w. pipeline parallelism"
        if (state_dict is not None) and (not isinstance(state_dict, str)):
            super().load_module_state_dict(state_dict, strict)
            return

        self.module.load_state_dir(load_dir=self._curr_ckpt_path,
                                   strict=strict,
                                   checkpoint_engine=self.checkpoint_engine)

    # A map of PipeInstruction types to methods. Each method will be executed with the
    # kwargs provided to the PipeInstruction from the scheduler.
    _INSTRUCTION_MAP = {
        schedule.OptimizerStep: _exec_optimizer_step,
        schedule.ReduceGrads: _exec_reduce_grads,
        schedule.ReduceTiedGrads: _exec_reduce_tied_grads,
        schedule.LoadMicroBatch: _exec_load_micro_batch,
        schedule.ForwardPass: _exec_forward_pass,
        schedule.BackwardPass: _exec_backward_pass,
        schedule.SendActivation: _exec_send_activations,
        schedule.RecvActivation: _exec_recv_activations,
        schedule.SendGrad: _exec_send_grads,
        schedule.RecvGrad: _exec_recv_grads,
    }

    _OMIT_INSTRUCTION = [schedule.SendActivation, schedule.RecvActivation, \
                                        schedule.SendGrad, schedule.RecvGrad]
    
    def _exec_schedule(self, pipe_schedule):
        # Reserve and reset buffers.
        self._reserve_pipe_info(pipe_schedule.num_pipe_buffers(), pipe_schedule.interrupt_info())
        self.fwd_outputs = []
        # print("##### Pipeline _exec_schedule #####")
        cout = 0
        # For each step in the schedule
        for step_cmds in pipe_schedule:
            # For each instruction in the step
            print(f"\nStep {cout}, global rank {self.global_rank}, exec cmds:{step_cmds}")
            for cmd in step_cmds:
                if type(cmd) not in self._INSTRUCTION_MAP:
                    raise RuntimeError(
                        f'{self.__class__.__name__} does not understand instruction {repr(cmd)}'
                    )
                # print(f"rank {self.global_rank} exec cmd: {cmd}")
                if self.module.timeline_path is not None:
                    if self.curr_batch_num >= self.timeline_skip and \
                        self.curr_batch_num < self.timeline_skip + self.timeline_dur:
                        torch.cuda.synchronize()
                        ds_utils.log_rank2file(f"{cmd}|{time.time()}")
                    
                self._exec_instr = MethodType(self._INSTRUCTION_MAP[type(cmd)], self)
                self._exec_instr(**cmd.kwargs)
            cout += 1