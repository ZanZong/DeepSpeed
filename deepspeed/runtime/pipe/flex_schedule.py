from ..utils import call_to_str

from abc import ABC, abstractmethod
from .schedule import PipeSchedule, LoadMicroBatch, ForwardPass, BackwardPass, \
    ReduceTiedGrads, ReduceGrads, OptimizerStep, SendActivation, SendGrad, \
        RecvActivation, RecvGrad, _is_even, _is_odd

split = 19
split_succ = 20

class TrainFlexSchedule(PipeSchedule):
    """ Interruptible and structure aware scheduling method.
    """
    
    def __init__(self, micro_batches, stages, stage_id, stage_read_data_layers):
        super().__init__(micro_batches, stages, stage_id)
        self.stage_read_data_layers = stage_read_data_layers

    def interrupt_info(self):
        """ Execution interrupt within a stage."""
        stage_breaks = {1: [33, 59]} # {stage_id:[break_layer0, break_layer1, ...]}
        return stage_breaks

    def steps(self):
        """"""
        # how to be automatic?
        total_steps = 1
        for step_id in range(total_steps):
            cmds = []
            if self.stage_id == 0:
                cmds.append(LoadMicroBatch(0))
                cmds.append(ForwardPass(None, kwargs={'buffer_id': 0, 'start': 0, 'end': split}))
                cmds.append(SendActivation(None, kwargs={'buffer_id': 0, 'send': split, 'recv': split_succ}))
                cmds.append(LoadMicroBatch(1))
                cmds.append(ForwardPass(None, kwargs={'buffer_id': 1, 'start': 0, 'end': split}))
                cmds.append(RecvGrad(None, kwargs={'buffer_id': 0, 'send': split_succ, 'recv': split}))
                cmds.append(SendActivation(None, kwargs={'buffer_id': 1, 'send': split, 'recv': split_succ}))
                cmds.append(BackwardPass(None, kwargs={'buffer_id': 0, 'left': 0, 'right': split}))
                cmds.append(RecvGrad(None, kwargs={'buffer_id': 1, 'send': split_succ, 'recv': split}))
                cmds.append(BackwardPass(None, kwargs={'buffer_id': 1, 'left': 0, 'right': split}))
                
            else:
                cmds.append(LoadMicroBatch(0))
                cmds.append(ForwardPass(None, kwargs={'buffer_id': 0, 'start': 34, 'end': 59}))
                cmds.append(LoadMicroBatch(1))
                cmds.append(ForwardPass(None, kwargs={'buffer_id': 1, 'start': 34, 'end': 59}))
                cmds.append(RecvActivation(None, kwargs={'buffer_id': 0, 'send': split, 'recv': split_succ}))
                cmds.append(ForwardPass(None, kwargs={'buffer_id': 0, 'start': split_succ, 'end': 33}))
                cmds.append(ForwardPass(None, kwargs={'buffer_id': 0, 'start': 60, 'end': 60}))
                cmds.append(BackwardPass(None, kwargs={'buffer_id': 0, 'left': 60, 'right': 60}))
                cmds.append(BackwardPass(None, kwargs={'buffer_id': 0, 'left': split_succ, 'right': 33}))
                cmds.append(SendGrad(None, kwargs={'buffer_id': 0, 'send': split_succ, 'recv': split}))
                # to be planed
                cmds.append(RecvActivation(None, kwargs={'buffer_id': 1, 'send': split, 'recv': split_succ}))
                cmds.append(ForwardPass(None, kwargs={'buffer_id': 1, 'start': split_succ, 'end': 33}))
                cmds.append(ForwardPass(None, kwargs={'buffer_id': 1, 'start': 60, 'end': 60}))
                cmds.append(BackwardPass(None, kwargs={'buffer_id': 1, 'left': 60, 'right': 60}))
                cmds.append(BackwardPass(None, kwargs={'buffer_id': 1, 'left': split_succ, 'right': 33}))
                cmds.append(SendGrad(None, kwargs={'buffer_id': 1, 'send': split_succ, 'recv': split}))
                cmds.append(BackwardPass(None, kwargs={'buffer_id': 0, 'left': 34, 'right': 59}))
                cmds.append(BackwardPass(None, kwargs={'buffer_id': 1, 'left': 34, 'right': 59}))
                

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                cmds.append(ReduceTiedGrads())
                cmds.append(ReduceGrads())
                cmds.append(OptimizerStep())

            yield cmds

    def num_pipe_buffers(self):
        """As many buffers as the distance from this stage to the last stage.
        """
        buffers = min(self.stages - self.stage_id + 1, self.micro_batches)
        return buffers

    def _step_to_micro_batch(self, step_id):
        if _is_even(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._even_step_forward_id(step_id)
            is_forward = True

        elif _is_odd(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._odd_step_forward_id(step_id)
            is_forward = True

        elif _is_even(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._even_step_backward_id(step_id)
            is_forward = False

        elif _is_odd(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._odd_step_backward_id(step_id)
            is_forward = False

        else:
            assert False

        return micro_batch_id, is_forward

    def _even_step_forward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _odd_step_forward_id(self, step_id):
        base = (step_id - 1) // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _even_step_backward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stages + (self.stage_id + 1) // 2)
        return micro_batch_id

    def _odd_step_backward_id(self, step_id):
        base = ((step_id - 1) // 2) - self.stages + 1
        micro_batch_id = int(base + self.stage_id // 2)
        return micro_batch_id



class TrainNaiveSchedule(PipeSchedule):
    """ Equivalent to TrainSchedule of schedule.py.
    """
    
    def __init__(self, micro_batches, stages, stage_id, stage_read_data_layers):
        super().__init__(micro_batches, stages, stage_id)
        self.stage_read_data_layers = stage_read_data_layers
        
    def interrupt_info(self):
        """ Execution interrupt within a stage."""
        stage_breaks = {1: [33, 59]}
        return stage_breaks
    
    def static_stage0_forward(self, buffer_id):
        cmds = []
        cmds.append(ForwardPass(None, kwargs={'buffer_id': buffer_id, 'start': 0, 'end': split}))
        return cmds
    
    def static_stage0_backward(self, buffer_id):
        cmds = []
        cmds.append(BackwardPass(None, kwargs={'buffer_id': buffer_id, 'left': 0, 'right': split}))
        return cmds
        
    def static_stage1_forward(self, buffer_id):
        cmds = []
        cmds.append(ForwardPass(None, kwargs={'buffer_id': buffer_id, 'start': split_succ, 'end': 33}))
        cmds.append(ForwardPass(None, kwargs={'buffer_id': buffer_id, 'start': 34, 'end': 59}))
        cmds.append(ForwardPass(None, kwargs={'buffer_id': buffer_id, 'start': 60, 'end': 60}))
        return cmds
        
    def static_stage1_backward(self, buffer_id):
        cmds = []
        cmds.append(BackwardPass(None, kwargs={'buffer_id': buffer_id, 'left': 60, 'right': 60}))
        cmds.append(BackwardPass(None, kwargs={'buffer_id': buffer_id, 'left': 34, 'right': 59}))
        cmds.append(BackwardPass(None, kwargs={'buffer_id': buffer_id, 'left': split_succ, 'right': 33}))
        return cmds
    
    def steps(self):
        """"""
        prev_micro_batch_id = -1
        # each rank needs process micro_batch times forward+bacward (2*self.micro_batches) and 
        # warm-up/finish steps (2*(self.stages-1))
        total_steps = 2 * (self.micro_batches + self.stages - 1)
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            if self._valid_micro_batch(prev_micro_batch_id):
                prev_buffer = self._buffer_idx(prev_micro_batch_id)
            if self._valid_micro_batch(micro_batch_id):
                curr_buffer = self._buffer_idx(micro_batch_id)

            cmds = []

            # Exchange activations
            if is_forward:
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                        self.prev_stage):
                    # cmds.append(RecvActivation(curr_buffer))
                    cmds.append(RecvActivation(None, kwargs={'buffer_id': curr_buffer, 'send': split, 'recv': split_succ}))
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(
                        self.prev_stage):
                    # cmds.append(SendGrad(prev_buffer))
                    cmds.append(SendGrad(None, kwargs={'buffer_id': prev_buffer, 'send': split_succ, 'recv': split}))
            else:
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(
                        self.next_stage):
                    # cmds.append(SendActivation(prev_buffer))
                    cmds.append(SendActivation(None, kwargs={'buffer_id': prev_buffer, 'send': split, 'recv': split_succ}))
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                        self.next_stage):
                    # cmds.append(RecvGrad(curr_buffer))
                    cmds.append(RecvGrad(None, kwargs={'buffer_id': curr_buffer, 'send': split_succ, 'recv': split}))
            
            # First/last stage loads
            if len(self.stage_read_data_layers[self.stage]) > 0:
                if is_forward and self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatch(curr_buffer))

            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    if self.stage_id == 0:
                        cmds.extend(self.static_stage0_forward(curr_buffer))
                    else:
                        cmds.extend(self.static_stage1_forward(curr_buffer))
                    # cmds.append(ForwardPass(curr_buffer))
                else:
                    if self.stage_id == 0:
                        cmds.extend(self.static_stage0_backward(curr_buffer))
                    else:
                        cmds.extend(self.static_stage1_backward(curr_buffer))
                    # cmds.append(BackwardPass(curr_buffer))

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                cmds.append(ReduceTiedGrads())
                cmds.append(ReduceGrads())
                cmds.append(OptimizerStep())

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield cmds

    def num_pipe_buffers(self):
        """As many buffers as the distance from this stage to the last stage.
        """
        buffers = min(self.stages - self.stage_id + 1, self.micro_batches)
        return buffers

    def _step_to_micro_batch(self, step_id):
        if _is_even(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._even_step_forward_id(step_id)
            is_forward = True

        elif _is_odd(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._odd_step_forward_id(step_id)
            is_forward = True

        elif _is_even(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._even_step_backward_id(step_id)
            is_forward = False

        elif _is_odd(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._odd_step_backward_id(step_id)
            is_forward = False

        else:
            assert False

        return micro_batch_id, is_forward

    def _even_step_forward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _odd_step_forward_id(self, step_id):
        base = (step_id - 1) // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _even_step_backward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stages + (self.stage_id + 1) // 2)
        return micro_batch_id

    def _odd_step_backward_id(self, step_id):
        base = ((step_id - 1) // 2) - self.stages + 1
        micro_batch_id = int(base + self.stage_id // 2)
        return micro_batch_id
