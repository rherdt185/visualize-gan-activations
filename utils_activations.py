import torch

def replace_blocks_with_activation_vector(activation, activation_vector, block_size=3):
    replaced_activations = torch.clone(activation)
    #for h in range(0, activation.shape[2] - block_size, 2*block_size):
    #    for w in range(0, activation.shape[3] - block_size, 2*block_size):

    for h in range(0, activation.shape[2] - block_size, block_size+1):
        for w in range(0, activation.shape[3] - block_size, block_size+1):
            for y in range(block_size):
                for x in range(block_size):
                    replaced_activations[:, :, h+y, w+x] = activation_vector

    return replaced_activations#.permute(0, 1, 3, 2)


hook_output = None

def layer_hook(module, input_, output):
    global hook_output
    hook_output = output


def access_activations_forward_hook_gan(z, label, forward_function, forward_hook_point):
    handle = forward_hook_point.register_forward_hook(layer_hook)
    with torch.no_grad():
        img = forward_function(z, label, truncation_psi=1, noise_mode="const")#, force_fp32=True)
        #print(img.dtype)
    handle.remove()

    #if isinstance(hook_output, list) or isinstance(hook_output, tuple):
    #    return hook_output
    return hook_output.detach().cpu(), img.cpu()


def access_activations_forward_hook(x, forward_function, forward_hook_point):
    handle = forward_hook_point.register_forward_hook(layer_hook)
    with torch.no_grad():
        forward_function(*x)
    handle.remove()

    #if isinstance(hook_output, list) or isinstance(hook_output, tuple):
    #    return hook_output
    return hook_output.detach().cpu()


class ForwardHookSetChannelsToValue:
    def __init__(self, forward_hook_point, value_to_set_to, channels_to_set=[]):
        self.channels_to_set = channels_to_set
        self.value_to_set_to = value_to_set_to
        self.forward_hook_point = forward_hook_point
    def set_hook(self):
        return self.forward_hook_point.register_forward_hook(self.layer_hook)
    def layer_hook(self, module, input_, output):
        #print("value to set to: {}".format(self.value_to_set_to))
        output_clone = torch.clone(output)
        if len(self.channels_to_set) == 0:
            self.channels_to_set = [i for i in range(output.shape[1])]
        for channel in self.channels_to_set:
            if len(self.value_to_set_to.shape) <= 1:
                output_clone[:, channel] = self.value_to_set_to
            else:
                output_clone[:, channel] = self.value_to_set_to[:, channel]
        return output_clone


