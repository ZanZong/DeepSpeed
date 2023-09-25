import torch.nn as nn
import torch
from . import p2p

#### Departed ####


DEBUG = False
def _print(msg):
    if DEBUG:
        print(msg)

def _allocate_zeros(shape, device, **kwargs):
    """ Allocate a tensor of zeros on the engine's device.

    Arguments:
        shape: the shape of the tensor to allocate
        kwargs: passed to torch.zeros()

    Returns:
        A tensor from torch.zeros() allocated on self.device.
    """
    # TODO: Caution check! Wrong buffer dtype will recv error values
    # if "dtype" not in kwargs:
    #     if self.fp16_enabled():
    #         kwargs["dtype"] = torch.half
    #     if self.bfloat16_enabled():
    #         kwargs["dtype"] = torch.bfloat16
    return torch.zeros(shape, device=device, **kwargs)

def _allocate_buffer(shape, num_buffers, device, **kwargs):
    buffers = []
    for count in range(num_buffers):
        buffers.append(_allocate_zeros(shape, device, **kwargs))
    return buffers

def _allocate_buffers(shapes_and_dtypes, num_buffers, device, requires_grad=False):
    buffers = []
    for count in range(num_buffers):
        buffer = []
        for shape, dtype in shapes_and_dtypes:
            buffer.append(
                _allocate_zeros(shape,
                                device,
                                dtype=dtype,
                                requires_grad=requires_grad))
        buffers.append(buffer)
    return buffers

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
    
def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()

def _send_tensor_meta(local_device, buffer, dest_stage):
    """ Communicate metadata about upcoming p2p transfers.

    Metadata is communicated in this order:
        * type (0: tensor, 1: list)
        * num_tensors if type=list
        foreach tensor in buffer:
            * ndims
            * shape
    """

    # send_bytes = 0
    if isinstance(buffer, torch.Tensor):
        type_tensor = torch.LongTensor(data=[0]).to(local_device)
        p2p.send(type_tensor, dest_stage)
        send_shape = torch.LongTensor(data=buffer.size()).to(local_device)
        send_ndims = torch.LongTensor(data=[len(buffer.size())]).to(local_device)
        p2p.send(send_ndims, dest_stage)
        p2p.send(send_shape, dest_stage)
        # send_bytes += _tensor_bytes(buffer)
    elif isinstance(buffer, tuple):
        type_tensor = torch.LongTensor(data=[2]).to(local_device)
        p2p.send(type_tensor, dest_stage)
        count_tensor = torch.LongTensor(data=[len(buffer)]).to(local_device)
        p2p.send(count_tensor, dest_stage)
        for idx, tensor in enumerate(buffer):
            assert isinstance(tensor, torch.Tensor), f"Tensor not valid: {tensor}"
            send_shape = torch.LongTensor(data=tensor.size()).to(local_device)
            send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(local_device)
            send_dtype = torch.LongTensor(data=[DTYPE_TO_ID[tensor.dtype]]).to(
                local_device)
            p2p.send(send_dtype, dest_stage)
            p2p.send(send_ndims, dest_stage)
            p2p.send(send_shape, dest_stage)
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

def _recv_tensor_meta(source_stage, local_device):
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

    type_tensor = torch.LongTensor(data=[0]).to(local_device)
    p2p.recv(type_tensor, source_stage)
    recv_type = type_tensor.item()

    # A single tensor will be sent.
    if recv_type == 0:
        recv_ndims = torch.LongTensor(data=[0]).to(local_device)
        p2p.recv(recv_ndims, source_stage)
        recv_ndims = recv_ndims.item()
        recv_shape = torch.LongTensor([1] * recv_ndims).to(local_device)
        p2p.recv(recv_shape, source_stage)
        recv_shape = recv_shape.tolist()
        return _allocate_buffer(recv_shape, num_buffers=1, \
                                            device=local_device)[0]
        
    # List or tuple of tensors
    elif recv_type == 1 or recv_type == 2:
        count_tensor = torch.LongTensor(data=[0]).to(local_device)
        p2p.recv(count_tensor, source_stage)
        num_tensors = count_tensor.item()
        recv_shapes_and_dtypes = []
        for idx in range(num_tensors):
            recv_dtype = torch.LongTensor(data=[0]).to(local_device)
            p2p.recv(recv_dtype, source_stage)
            recv_dtype = ID_TO_DTYPE[recv_dtype.item()]
            recv_ndims = torch.LongTensor(data=[0]).to(local_device)
            p2p.recv(recv_ndims, source_stage)
            recv_ndims = recv_ndims.item()
            recv_shape = torch.LongTensor([1] * recv_ndims).to(local_device)
            p2p.recv(recv_shape, source_stage)
            recv_shapes_and_dtypes.append((recv_shape.tolist(), recv_dtype))
    
        buffers = _allocate_buffers(recv_shapes_and_dtypes, 
                                    num_buffers=1, device=local_device)[0]
        # Convert to tuples if requested.
        if recv_type == 2:
            buffers = tuple(buffers)
        return buffers

    else:
        raise NotImplementedError(f'Could not receive type {type(recv_type)}')


def gen_module_hook_activation_send(local_device, dest_stage):

    def module_hook_activation_send(module, inp, out):
        if module.first_output_send:
            module.first_output_send = False
            _send_tensor_meta(local_device, out, dest_stage)
        _print(f"Call send activation hook!")
        if isinstance(out, torch.Tensor):
            if not out.is_contiguous():
                out = out.contiguous()
            p2p.send(out, dest_stage)

        elif isinstance(out, tuple):
            for idx, buffer in enumerate(out):
                p2p.send(buffer, dest_stage)
        else:
            raise NotImplementedError('Could not send output of type '
                                      f'{type(out)}')
        _print(f"Send activation hook FINISH!")
    return module_hook_activation_send

def gen_module_hook_grad_recv_hook(succ_layer, source_stage, local_device):
    
    def module_hook_grad_recv_hook(module, inp_grad):
        # _print(f"Call recv grad hook! Get mock in_grad={inp_grad}")
        # if module.grad_layer is None:
        #     data = module.output_buffers[succ_layer][module.buffer_id]
        #     if isinstance(data, torch.Tensor):
        #             s = list(data.size())
        #             module.grad_layer = _allocate_buffer(s,
        #                                                 dtype=data.dtype,
        #                                                 num_buffers=1,
        #                                                 device=local_device)[0]
        #     else:
        #         sizes_and_dtypes = [(list(t.size()),
        #                                     t.dtype) for t in data
        #                                     if t.is_floating_point() and t.requires_grad]
        #         module.grad_layer = _allocate_buffers(sizes_and_dtypes,
        #                                             num_buffers=1,
        #                                             device=local_device)[0]
        if isinstance(module.grad_layer[succ_layer], torch.Tensor):
            p2p.recv(module.grad_layer[succ_layer], source_stage)
            _print(f"Recv real grad={module.grad_layer[succ_layer]}")
        else:
            # output is a tuple, recv grad one-by-one 
            # TODO: (check) is the order right? need a tag to distinguish peers
            for idx, buffer in enumerate(module.grad_layer[succ_layer]):
                p2p.recv(buffer, source_stage)
    
    return module_hook_grad_recv_hook

def gen_module_hook_activation_recv(pred_layer, source_stage, local_device):
    
    def module_hook_activation_recv(module, inp):
        _print("Call recv activation hook!")
        recvd = None
        # Allocate the buffer if necessary
        if module.pipe_recv_buf is None:
            module.pipe_recv_buf = _recv_tensor_meta(source_stage, local_device)
            
        if isinstance(module.pipe_recv_buf, torch.Tensor):
            p2p.recv(module.pipe_recv_buf, source_stage)
            recvd = module.pipe_recv_buf.clone().detach()
            recvd.requires_grad = recvd.is_floating_point()
            _print(f"Recv a tensor shape={recvd.shape}")
        else:
            assert isinstance(module.pipe_recv_buf, tuple)
            recvd = [None] * len(module.pipe_recv_buf)
            for idx, buffer in enumerate(module.pipe_recv_buf):
                assert torch.is_tensor(buffer)
                p2p.recv(buffer, source_stage)
                recvd[idx] = buffer.clone().detach()
            _print(f"Recv a tuple={len(recvd)}")
            recvd = tuple(recvd)
            
            # TODO a better way to identify non-grad tensor, e.g., mask
            for idx, buffer in enumerate(recvd):
                buffer.requires_grad = buffer.is_floating_point() and idx < 2

        if isinstance(recvd, torch.Tensor):
            recvd.register_hook(gen_tensor_hook_grad_send(dest_device=source_stage))
        else:
            # send multiple grad. 
            # TODO: (check) is the order right? need a tag to distinguish peers
            for t in recvd:
                t.register_hook(gen_tensor_hook_grad_send(dest_device=source_stage))
        module.input_buffers[pred_layer][module.buffer_id] = recvd
        return module.input_buffers[pred_layer][module.buffer_id]
        
    return module_hook_activation_recv

def gen_tensor_hook_grad_send(dest_device):
    def tensor_hook_grad_send(grad):
        _print(f"Call send grad hook! grad={grad.shape}, to device {dest_device}")
        if not grad.is_contiguous():
                _print(f"[grad.send] Tensor is not contiguous, convert!")
                grad = grad.contiguous()
        p2p.send(grad, dest_device)
    return tensor_hook_grad_send