
import torch
import pickle
import numpy as np
import random
import torchvision
from tqdm import tqdm

from torchvision.models import resnet50, resnet34

from pathlib import Path
import torch.nn.functional as F

# a feature extractor for the brecahad gan, since here a feature extractor trained on imagenet might not work good
# the model checkpoint can be downloaded from the monai model zoo (the 'Pathology tumor detection' model)
try:
    from monai.networks.nets import TorchVisionFCModel
except:
    pass

from utils_activations import replace_blocks_with_activation_vector, access_activations_forward_hook_gan, access_activations_forward_hook
from config import STYLEGAN2_REPO_PATH

import sys
sys.path.append(STYLEGAN2_REPO_PATH)


NORMALIZATION_IMAGENET = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))



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


def load_model(model_str):
    if model_str == "resnet50":
        return resnet50(pretrained=True).to("cuda").eval()
    elif model_str == "resnet34":
        return resnet34(pretrained=True).to("cuda").eval()
    elif model_str == "resnet50_robust":
        classifier_model = resnet50(pretrained=False)

        checkpoint = torch.load("models/ImageNet.pt")
        state_dict_path = 'model'
        if not ('model' in checkpoint):
            state_dict_path = 'state_dict'

        new_state_dict = {}
        for key in checkpoint['model'].keys():
            if not "attacker" in key:
                if 'model' in key:
                    new_key = key[13:]
                    #print(new_key)
                    new_state_dict[new_key] = checkpoint['model'][key]

        classifier_model.load_state_dict(new_state_dict)

        return classifier_model.eval().to("cuda")
    elif model_str == "resnet18_monai":
        model = TorchVisionFCModel(pretrained=False, use_conv=True)

        state_dict = torch.load("pathology_tumor_detection/models/model.pt")#.to("cuda").eval()

        model.load_state_dict(state_dict)
        model = model.to("cuda").eval()
        #print(model)
        #print(model_load)

        return model





def load_gan(gan_model="afhqwild.pkl"):
    device = torch.device('cuda')
    with open(gan_model, 'rb') as f:

        data = pickle.load(f)
        G = data['G_ema'].to(device).eval()
        #G = legacy.load_network_pkl("model_ckpts/stylegan2/brecahad.pkl")['G_ema'].to(device).eval()
    #self.network_gan = WrapperStyleGAN2(G)

    label = torch.zeros([1, G.c_dim], device="cuda")
    z = torch.from_numpy(np.random.RandomState(random.randint(0, 100000000)).randn(1, G.z_dim)).to("cuda")

    return G


def get_random_activation_vectors_from_batch(activations):
    batch_size = activations.shape[0]
    indices = np.random.randint(0, activations.shape[-1]-1, size=(batch_size, 2))
    batch_idx = torch.from_numpy(np.linspace(0, batch_size-1, batch_size)).to(torch.long)
    y = indices[:, 0]
    x = indices[:, 1]
    return activations[batch_idx, :, y, x]




def get_imgs_full_and_blocks_list(G, layer, num_samples_to_get=8, block_sizes=[1, 2, 3, 4, 5], batch_size=32,
                             debug_mode=False,
                             background='original'):
    #G = load_gan()
    #layer = G.synthesis.b16.conv1

    #num_samples = 8
    imgs_full = []
    imgs_block = [[] for _ in block_sizes]
    imgs_original = []
    pixel_masks = []
    masks_gan = []

    torch.cuda.empty_cache()

    for i in tqdm(range(0, num_samples_to_get, batch_size), ascii=True):
        with torch.no_grad():
            label = torch.zeros([batch_size, G.c_dim], device="cuda")
            z = torch.from_numpy(np.random.RandomState(random.randint(0, 100000000)).randn(batch_size, G.z_dim)).to("cuda")
            activation, img_original = access_activations_forward_hook_gan(z, label, G, layer)
            img_original = img_original*0.5+0.5

            #img_original = G(z, label, truncation_psi=1, noise_mode="const")*0.5+0.5
            #img_original = F.interpolate(img_original, size=256, mode="bilinear").cpu()
            imgs_original.append(img_original)

            indices = np.random.randint(0, activation.shape[-1]-1, size=(batch_size, 2))
            #y = random.randint(0, activation.shape[-1]-1)
            #x = random.randint(0, activation.shape[-2]-1)
            y = indices[:, 0]
            x = indices[:, 1]

            batch_idx = torch.from_numpy(np.linspace(0, batch_size-1, batch_size)).to(torch.long)

            pixel_mask = torch.zeros_like(activation)
            pixel_mask[batch_idx, :, y, x] = 1.0
            pixel_mask = pixel_mask[:, :3, :, :]
            #pixel_positions.append([y, x])

            indices_other = np.random.randint(0, activation.shape[-1]-1, size=(batch_size, 2))
            y_other = indices_other[:, 0]
            x_other = indices_other[:, 1]

            #y = random.randint(1, 3)
            #x = random.randint(1, 3)

            #y = random.randint(5, activation.shape[-1]-6)
            #x = random.randint(5, activation.shape[-2]-6)

            activation_vec = activation[batch_idx, :, y, x]

            activation_vec_other = activation[batch_idx, :, y_other, x_other]

            #activation = torch.ones_like(activation) * activation_vec_other.unsqueeze(dim=-1).unsqueeze(dim=-1)

            #print("acrtivatoin shape: {}".format(activation.shape))
            #print("activation vec shape: {}".format(activation_vec.shape))

            for idx_block, block_size in enumerate(block_sizes):
                #print(idx_block)
                if background == 'original':
                    replaced_activation = replace_blocks_with_activation_vector(activation, activation_vec, block_size=block_size)
                elif background == 'full_original':
                    replaced_activation = activation
                elif background == 'random_vec':
                    # use random activation vector from image as background
                    activation = torch.ones_like(activation) * activation_vec_other.unsqueeze(dim=-1).unsqueeze(dim=-1)
                    replaced_activation = replace_blocks_with_activation_vector(activation, activation_vec, block_size=block_size)

                    # use original activations completely
                    #replaced_activation = activation
                else:
                    raise RuntimeError("background input not understood")

                if block_size == 2:
                    mask_gan = replace_blocks_with_activation_vector(torch.zeros_like(activation), torch.ones_like(activation_vec), block_size=block_size)

                if debug_mode:
                    torchvision.utils.save_image(torch.mean(mask_gan, dim=1)[0].unsqueeze(dim=0).unsqueeze(dim=0), "mask_block_gan.png")

                hook = ForwardHookSetChannelsToValue(forward_hook_point=layer,
                                                value_to_set_to=replaced_activation).set_hook()
                img_blocks = G(z, label, truncation_psi=1, noise_mode="const")*0.5+0.5
                img_blocks = img_blocks.cpu()
                #img_blocks = F.interpolate(img_blocks, size=256, mode="bilinear").cpu()

                imgs_block[idx_block].append(img_blocks)
                hook.remove()

            replaced_activation = torch.ones_like(activation) * activation_vec.unsqueeze(dim=-1).unsqueeze(dim=-1)
            hook = ForwardHookSetChannelsToValue(forward_hook_point=layer,
                                            value_to_set_to=replaced_activation).set_hook()
            img_full = G(z, label, truncation_psi=1, noise_mode="const")*0.5+0.5
            img_full = img_full.cpu()
            #img_full = F.interpolate(img_full, size=256, mode="bilinear")

            imgs_full.append(img_full)
            hook.remove()

            pixel_mask = F.interpolate(pixel_mask, size=img_full.shape[-1], mode="nearest")
            pixel_masks.append(pixel_mask)

            mask_gan = F.interpolate(mask_gan[:, :3, :, :], size=img_full.shape[-1], mode="nearest")
            masks_gan.append(mask_gan)

            torch.cuda.empty_cache()

    for i, imgs in enumerate(imgs_block):
        imgs_block[i] = torch.cat(imgs, dim=0)
    #imgs_block = torch.cat(imgs_block, dim=0)
    #torchvision.utils.save_image(imgs_block, "sample_blocks.jpg")

    imgs_full = torch.cat(imgs_full, dim=0)
    #torchvision.utils.save_image(imgs_full, "sample_full.jpg")

    imgs_original = torch.cat(imgs_original, dim=0)
    pixel_masks = torch.cat(pixel_masks, dim=0)
    masks_gan = torch.cat(masks_gan, dim=0)

    torch.cuda.empty_cache()

    return imgs_block, imgs_full, imgs_original, pixel_masks, masks_gan


def get_imgs_full_and_blocks(G, layer, num_samples_to_get=8, block_size=2, batch_size=32,
                             debug_mode=False,
                             background='original'):
    #G = load_gan()
    #layer = G.synthesis.b16.conv1

    #num_samples = 8
    imgs_full = []
    imgs_block = []
    imgs_original = []
    pixel_masks = []
    masks_gan = []

    for i in tqdm(range(0, num_samples_to_get, batch_size), ascii=True):
        with torch.no_grad():
            label = torch.zeros([batch_size, G.c_dim], device="cuda")
            z = torch.from_numpy(np.random.RandomState(random.randint(0, 100000000)).randn(batch_size, G.z_dim)).to("cuda")
            activation, img_original = access_activations_forward_hook_gan(z, label, G, layer)
            img_original = img_original*0.5+0.5
            indices = np.random.randint(0, activation.shape[-1]-1, size=(batch_size, 2))
            #y = random.randint(0, activation.shape[-1]-1)
            #x = random.randint(0, activation.shape[-2]-1)
            y = indices[:, 0]
            x = indices[:, 1]

            batch_idx = torch.from_numpy(np.linspace(0, batch_size-1, batch_size)).to(torch.long)

            pixel_mask = torch.zeros_like(activation)
            pixel_mask[batch_idx, :, y, x] = 1.0
            pixel_mask = pixel_mask[:, :3, :, :]
            #pixel_positions.append([y, x])

            indices_other = np.random.randint(0, activation.shape[-1]-1, size=(batch_size, 2))
            y_other = indices_other[:, 0]
            x_other = indices_other[:, 1]

            #y = random.randint(1, 3)
            #x = random.randint(1, 3)

            #y = random.randint(5, activation.shape[-1]-6)
            #x = random.randint(5, activation.shape[-2]-6)

            activation_vec = activation[batch_idx, :, y, x]

            activation_vec_other = activation[batch_idx, :, y_other, x_other]

            #activation = torch.ones_like(activation) * activation_vec_other.unsqueeze(dim=-1).unsqueeze(dim=-1)

            #print("acrtivatoin shape: {}".format(activation.shape))
            #print("activation vec shape: {}".format(activation_vec.shape))

            if background == 'original':
                replaced_activation = replace_blocks_with_activation_vector(activation, activation_vec, block_size=block_size)
            elif background == 'full_original':
                replaced_activation = activation
            elif background == 'random_vec':
                # use random activation vector from image as background
                activation = torch.ones_like(activation) * activation_vec_other.unsqueeze(dim=-1).unsqueeze(dim=-1)
                replaced_activation = replace_blocks_with_activation_vector(activation, activation_vec, block_size=block_size)

                # use original activations completely
                #replaced_activation = activation
            else:
                raise RuntimeError("background input not understood")


            mask_gan = replace_blocks_with_activation_vector(torch.zeros_like(activation), torch.ones_like(activation_vec), block_size=block_size)

            if debug_mode:
                torchvision.utils.save_image(torch.mean(mask_gan, dim=1)[0].unsqueeze(dim=0).unsqueeze(dim=0), "mask_block_gan.png")

            #img_original = G(z, label, truncation_psi=1, noise_mode="const")*0.5+0.5
            imgs_original.append(img_original.cpu())

            hook = ForwardHookSetChannelsToValue(forward_hook_point=layer,
                                            value_to_set_to=replaced_activation).set_hook()
            img_blocks = G(z, label, truncation_psi=1, noise_mode="const")*0.5+0.5
            #img_blocks = F.interpolate(img_blocks, size=256, mode="bilinear")

            imgs_block.append(img_blocks.cpu())
            hook.remove()

            replaced_activation = torch.ones_like(activation) * activation_vec.unsqueeze(dim=-1).unsqueeze(dim=-1)
            hook = ForwardHookSetChannelsToValue(forward_hook_point=layer,
                                            value_to_set_to=replaced_activation).set_hook()
            img_full = G(z, label, truncation_psi=1, noise_mode="const")*0.5+0.5
            #img_full = F.interpolate(img_full, size=256, mode="bilinear")

            imgs_full.append(img_full.cpu())
            hook.remove()

            pixel_mask = F.interpolate(pixel_mask, size=img_full.shape[-1], mode="nearest")
            pixel_masks.append(pixel_mask)

            mask_gan = F.interpolate(mask_gan[:, :3, :, :], size=img_full.shape[-1], mode="nearest")
            masks_gan.append(mask_gan)

    imgs_block = torch.cat(imgs_block, dim=0)
    #torchvision.utils.save_image(imgs_block, "sample_blocks.jpg")

    imgs_full = torch.cat(imgs_full, dim=0)
    #torchvision.utils.save_image(imgs_full, "sample_full.jpg")

    imgs_original = torch.cat(imgs_original, dim=0)
    pixel_masks = torch.cat(pixel_masks, dim=0)
    masks_gan = torch.cat(masks_gan, dim=0)



    return imgs_block, imgs_full, imgs_original, pixel_masks, masks_gan


def get_model_variability(model, dataloader, layer, interpolation_size=256,
                          preprocessing=torch.nn.Identity()):
    activations = []
    #normalization = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    with torch.no_grad():
        for x in tqdm(dataloader, ascii=True):
            x = x[0].to("cuda")
            x = F.interpolate(x, size=interpolation_size)
            x = preprocessing(x)

            activation = access_activations_forward_hook([x], model, layer)
            activations.append(activation)

        activations = torch.cat(activations, dim=0)
        activations = torch.permute(activations, dims=[0, 2, 3, 1])
        #print("activations shape: {}".format(activations.shape))
        activation_vectors = torch.flatten(activations, start_dim=0, end_dim=2).to("cuda")
        #print("activation vgecshape: {}".format(activation_vectors.shape))

        #raise RuntimeError

        cosine_similarities = []
        L1_loss = []
        for probe_vec in tqdm(activation_vectors, ascii=True):
            probe_vec = probe_vec.unsqueeze(dim=0)
            cossim = F.cosine_similarity(probe_vec, activation_vectors, dim=1)
            l1_loss_sample = torch.mean(torch.abs(probe_vec - activation_vectors), dim=1)
            cosine_similarities.append(cossim.cpu())
            L1_loss.append(l1_loss_sample.cpu())
        cosine_similarities = torch.cat(cosine_similarities, dim=0)
        mean_cossim = torch.mean(cosine_similarities)
        std_cossim = torch.std(cosine_similarities)

        L1_loss = torch.cat(L1_loss, dim=0)
        mean_L1_loss = torch.mean(L1_loss)
        std_L1_loss = torch.std(L1_loss)

        print("mean cossim: {}".format(mean_cossim))
        print("std cossim: {}".format(std_cossim))

        print("mean L1_loss: {}".format(mean_L1_loss))
        print("std L1_loss: {}".format(std_L1_loss))


def get_imgs_as_dataloader(num_samples_to_get=256, batch_size=64, batch_size_loader=64, gan_model="afhqwild.pkl"):
    G = load_gan(gan_model)

    imgs = []
    for i in range(0, num_samples_to_get, batch_size):
        with torch.no_grad():
            label = torch.zeros([batch_size, G.c_dim], device="cuda")
            z = torch.from_numpy(np.random.RandomState(random.randint(0, 100000000)).randn(batch_size, G.z_dim)).to("cuda")

            gen_imgs = G(z, label, truncation_psi=1, noise_mode="const")*0.5+0.5
            imgs.append(gen_imgs.cpu())
    imgs = torch.cat(imgs, dim=0)

    dataset = torch.utils.data.TensorDataset(imgs)
    #print("len dataset: {}".format(len(dataset)))
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=8, batch_size=batch_size_loader)

    return dataloader


def run_metrics(classifier, layer_classifier, gan, layer_gan, interp_size=256,
                image_post_processing=torch.nn.Identity(), num_samples=256,
                classifier_block_size=3, classifier_max_pool_kernel_size=3,
                image_save_prefix="", gan_block_size=3, save_images=False,
                classifier_pixel_pool_size=1,
                batch_size=64, batch_size_img_generator=32, classifier_pixel_erosion=False,
                debug_mode=False, background='original', use_grid_mask=True):
    """ Generates num_samples of visualizations. Sorts them by tileability (similarity between setting all pixels to the activation vector
    and only in a grid) and saves the 8 most and least tileable visualizations. """
    #block_size = 2
    imgs_blocks, imgs_full, imgs_original, pixel_masks, masks_gan = get_imgs_full_and_blocks_list(gan, layer_gan, num_samples_to_get=num_samples, #block_size=gan_block_size,
                                                                                            batch_size=batch_size_img_generator, debug_mode=debug_mode,
                                                                                            background=background)

    #imgs_block, imgs_full, imgs_original, pixel_masks, masks_gan = get_imgs_full_and_blocks(gan, layer_gan, num_samples_to_get=num_samples, #block_size=gan_block_size,
    #                                                                                        batch_size=batch_size_img_generator, debug_mode=debug_mode,
    #                                                                                        background=background)

    imgs_block = imgs_blocks[1]  # block size of 2
    #print("imgs block shape: {}".format(imgs_block.shape))

    imgs_block_original = torch.clone(imgs_block)
    imgs_full_original = torch.clone(imgs_full)

    imgs = torch.cat([imgs_block, imgs_full, imgs_original, pixel_masks, masks_gan], dim=1)

    #normalization = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    dataset = torch.utils.data.TensorDataset(imgs)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=8, batch_size=batch_size)

    #classifier = load_model("resnet50")
    #layer_classifier = classifier.layer3#[2]

    cossine_similarity = []
    cossine_similarity_rolled = []
    cossine_similarity_full_replace = []
    cossine_similarity_mask_replace = []

    L1_loss = []
    L1_loss_rolled = []
    L1_loss_full_replace = []
    L1_loss_mask_replace = []

    with torch.no_grad():
        for data in dataloader:
            data = data[0]
            #print(data.shape)
            imgs_block = data[:, :3].to("cuda")
            imgs_full = data[:, 3:6].to("cuda")
            img_original = data[:, 6:9].to("cuda")
            imgs_pixel_mask = data[:, 9:12].to("cuda")
            mask_block = data[:, 12:].to("cuda")

            imgs_block = F.interpolate(imgs_block.to("cuda"), size=interp_size)
            imgs_full = F.interpolate(imgs_full.to("cuda"), size=interp_size)
            img_original = F.interpolate(img_original.to("cuda"), size=interp_size)
            #imgs_pixel_mask = F.interpolate(imgs_pixel_mask.to("cuda"), size=interp_size, mode="nearest")

            imgs_block = image_post_processing(imgs_block)
            imgs_full = image_post_processing(imgs_full)
            img_original = image_post_processing(img_original)

            if debug_mode:
                overlay = F.interpolate(imgs_pixel_mask[:, 0, :, :].unsqueeze(dim=1), mode="nearest", size=img_original.shape[-1]) * img_original
                torchvision.utils.save_image(overlay, "overlay.jpg")
                torchvision.utils.save_image(img_original, "original_image.jpg")

            activations_block = access_activations_forward_hook([imgs_block], classifier, layer_classifier)
            activations_full = access_activations_forward_hook([imgs_full], classifier, layer_classifier)
            activations_original = access_activations_forward_hook([img_original], classifier, layer_classifier)
            imgs_pixel_mask = F.interpolate(imgs_pixel_mask[:, 0, :, :].unsqueeze(dim=1), mode="nearest", size=activations_original.shape[-1]).cpu()

            mask_block = F.interpolate(mask_block[:, 0, :, :].unsqueeze(dim=1), mode="nearest", size=activations_original.shape[-1]).cpu()
            if classifier_pixel_pool_size > 1:
                to_pad = int(classifier_pixel_pool_size/2)
                imgs_pixel_mask = F.pad(imgs_pixel_mask, pad=(to_pad, to_pad, to_pad, to_pad), value=0.0)
                if not classifier_pixel_erosion:
                    imgs_pixel_mask = F.max_pool2d(imgs_pixel_mask, kernel_size=classifier_pixel_pool_size, padding=0, stride=1)
                else:
                    imgs_pixel_mask = -F.max_pool2d(-imgs_pixel_mask, kernel_size=classifier_pixel_pool_size, padding=0, stride=1)
            #print(torch.sum(imgs_pixel_mask))
            masked_activation_original = activations_original * imgs_pixel_mask
            masked_activation_vec = torch.sum(masked_activation_original, dim=[2, 3]) / torch.sum(imgs_pixel_mask, dim=[2, 3])

            #mask_block = replace_blocks_with_activation_vector(torch.zeros_like(activations_block), torch.ones(activations_block.shape[0], activations_block.shape[1]), block_size=classifier_block_size)
            to_pad = int(classifier_max_pool_kernel_size / 2)
            mask_block = F.pad(mask_block, pad=(to_pad, to_pad, to_pad, to_pad), value=0.0)
            mask_block = -F.max_pool2d(-mask_block, kernel_size=classifier_max_pool_kernel_size, padding=0, stride=1)

            if debug_mode:
                torchvision.utils.save_image(torch.mean(mask_block, dim=1)[0].unsqueeze(dim=0).unsqueeze(dim=0), "mask_block classifier.png")
                torchvision.utils.save_image(imgs_pixel_mask, "pixel_mask_processed.png")

            #print("mask block shape: {}".format(mask_block.shape))

            mask_block_rolled = torch.roll(mask_block, shifts=classifier_block_size, dims=-1)
            mask_block_rolled[:, :, :, :classifier_block_size] = 0.0

            activations_block_rolled = activations_block*mask_block_rolled
            #activations_block = activations_block*mask_block
            if use_grid_mask:
                activations_block = activations_block*mask_block
            #print(torch.sum(activations_block_rolled))

            activation_vector_full = torch.mean(activations_full, dim=[2, 3])
            if use_grid_mask:
                activation_vector_block = torch.sum(activations_block, dim=[2, 3]) / torch.sum(mask_block, dim=[2, 3])
            else:
                activation_vector_block = torch.mean(activations_block, dim=[2, 3])
            #activation_vector_block = torch.mean(activations_block, dim=[2, 3])
            activation_vector_block_rolled = torch.sum(activations_block_rolled, dim=[2, 3]) / torch.sum(mask_block_rolled, dim=[2, 3])



            if debug_mode:
                torchvision.utils.save_image(torch.mean(mask_block_rolled, dim=1)[0].unsqueeze(dim=0).unsqueeze(dim=0), "mask_block_rolled classifier.png")


            cossim = F.cosine_similarity(activation_vector_full, activation_vector_block, dim=1)
            cossim_rolled = F.cosine_similarity(activation_vector_full, activation_vector_block_rolled, dim=1)
            cossim_masked_pixel_full_replace = F.cosine_similarity(activation_vector_full, masked_activation_vec, dim=1)
            cossim_masked_pixel_mask_replace = F.cosine_similarity(activation_vector_block, masked_activation_vec, dim=1)

            cossine_similarity.append(cossim)
            cossine_similarity_rolled.append(cossim_rolled)
            cossine_similarity_full_replace.append(cossim_masked_pixel_full_replace)
            cossine_similarity_mask_replace.append(cossim_masked_pixel_mask_replace)

            l1_loss_sample = torch.mean(torch.abs(activation_vector_full - activation_vector_block), dim=[1])
            l1_loss_rolled_sample = torch.mean(torch.abs(activation_vector_full - activation_vector_block_rolled), dim=[1])
            l1_loss_masked_pixel_full_replace = torch.mean(torch.abs(activation_vector_full - masked_activation_vec), dim=[1])
            l1_loss_masked_pixel_mask_replace = torch.mean(torch.abs(activation_vector_block - masked_activation_vec), dim=[1])

            L1_loss.append(l1_loss_sample)
            L1_loss_rolled.append(l1_loss_rolled_sample)
            L1_loss_full_replace.append(l1_loss_masked_pixel_full_replace)
            L1_loss_mask_replace.append(l1_loss_masked_pixel_mask_replace)

    cossine_similarity = torch.cat(cossine_similarity, dim=0)
    cossine_similarity_rolled = torch.cat(cossine_similarity_rolled, dim=0)
    cossine_similarity_full_replace = torch.cat(cossine_similarity_full_replace, dim=0)
    cossine_similarity_mask_replace = torch.cat(cossine_similarity_mask_replace, dim=0)

    L1_loss = torch.cat(L1_loss, dim=0)
    L1_loss_rolled = torch.cat(L1_loss_rolled, dim=0)
    L1_loss_full_replace = torch.cat(L1_loss_full_replace, dim=0)
    L1_loss_mask_replace = torch.cat(L1_loss_mask_replace, dim=0)

    """
    print("mean cosine similarity: {}".format(torch.mean(cossine_similarity)))
    print("std cossim: {}".format(torch.std(cossine_similarity)))
    print("mean cosine similarity rolled: {}".format(torch.mean(cossine_similarity_rolled)))
    print("std cossim rolled: {}".format(torch.std(cossine_similarity_rolled)))
    print("mean cossine_similarity_full_replace: {}".format(torch.mean(cossine_similarity_full_replace)))
    print("std cossine_similarity_full_replace: {}".format(torch.std(cossine_similarity_full_replace)))
    print("mean cossine_similarity_mask_replace: {}".format(torch.mean(cossine_similarity_mask_replace)))
    print("std cossine_similarity_mask_replace: {}".format(torch.std(cossine_similarity_mask_replace)))

    print("mean l1 loss: {}".format(torch.mean(L1_loss)))
    print("std l1 loss: {}".format(torch.std(L1_loss)))
    print("mean l1 loss rolled: {}".format(torch.mean(L1_loss_rolled)))
    print("std l1 loss rolled: {}".format(torch.std(L1_loss_rolled)))
    print("mean L1_loss_full_replace: {}".format(torch.mean(L1_loss_full_replace)))
    print("std L1_loss_full_replace: {}".format(torch.std(L1_loss_full_replace)))
    print("mean L1_loss_mask_replace: {}".format(torch.mean(L1_loss_mask_replace)))
    print("std L1_loss_mask_replace: {}".format(torch.std(L1_loss_mask_replace)))
    """

    if save_images:
        Path("images/").mkdir(parents=True, exist_ok=True)

        sorted_indices = np.argsort(cossine_similarity.numpy())
        #torchvision.utils.save_image(imgs_block_original[sorted_indices], "images/" + "all_block_" + image_save_prefix + ".jpg")
        #torchvision.utils.save_image(imgs_full_original[sorted_indices], "images/" + "all_full_" + image_save_prefix + ".jpg")

        if debug_mode:
            torchvision.utils.save_image(imgs_block_original, "images/block.jpg")
            torchvision.utils.save_image(imgs_full_original, "images/full.jpg")


        for block_idx, imgs in enumerate(imgs_blocks):
            torchvision.utils.save_image(imgs[sorted_indices][:8], "images/" + image_save_prefix + "_lowest_block_"+str(block_idx+1)+".jpg")
            torchvision.utils.save_image(imgs[sorted_indices][-8:], "images/" + image_save_prefix + "_highest_block_"+str(block_idx+1)+".jpg")


        torchvision.utils.save_image(imgs_block_original[sorted_indices][:8], "images/" + image_save_prefix + "_lowest_block.jpg")
        torchvision.utils.save_image(imgs_full_original[sorted_indices][:8], "images/" + image_save_prefix + "_lowest_full.jpg")

        torchvision.utils.save_image(imgs_block_original[sorted_indices][-8:], "images/" + image_save_prefix + "_highest_block.jpg")
        torchvision.utils.save_image(imgs_full_original[sorted_indices][-8:], "images/" + image_save_prefix + "_highest_full.jpg")

    """
    largest_indices = np.argpartition(cossine_similarity.numpy(), -32)[-32:]

    img_block = imgs_block_original[largest_indices]
    img_full = imgs_full_original[largest_indices]

    torchvision.utils.save_image(img_block, "images_tmp/" + image_save_prefix + "_highest_block.jpg")
    torchvision.utils.save_image(img_full, "images_tmp/" + image_save_prefix + "_highest_full.jpg")


    smallest_indices = np.argpartition(cossine_similarity.numpy(), 32)[:32]

    img_block = imgs_block_original[smallest_indices]
    img_full = imgs_full_original[smallest_indices]

    torchvision.utils.save_image(img_block, "images_tmp/" + image_save_prefix + "_smallest_block.jpg")
    torchvision.utils.save_image(img_full, "images_tmp/" + image_save_prefix + "_smallest_full.jpg")
    """
    return torch.mean(cossine_similarity).numpy(), torch.mean(cossine_similarity_full_replace).numpy(), torch.mean(cossine_similarity_mask_replace).numpy(), torch.mean(L1_loss).numpy(), torch.mean(L1_loss_full_replace).numpy(), torch.mean(L1_loss_mask_replace).numpy()




def print_layers(model, prefix=""):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Module):
            print(prefix + name + ":")
            print_layers(module, prefix + "  ")
        else:
            print(prefix + name)


def run_eval():
    normalization = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    monai_transform = lambda x: 2*(x-0.5)

    gan_models = ["afhqwild.pkl", "afhqwild.pkl", "stylegan2-church-config-f.pkl", "stylegan2-church-config-f.pkl", "brecahad.pkl", "brecahad.pkl"]#*3
    gan_layers = ["gan.synthesis.b16.conv1", "gan.synthesis.b32.conv1", "gan.synthesis.b16.conv1", "gan.synthesis.b32.conv1", "gan.synthesis.b16.conv1", "gan.synthesis.b32.conv1"]*3
    classifiers = ["resnet50", "resnet50", "resnet50", "resnet50", "resnet18_monai", "resnet18_monai"]*3
    classifier_layers = ["classifier.layer2", "classifier.layer2", "classifier.layer2", "classifier.layer2", "classifier.features[5]", "classifier.features[5]"]*3
    preprocessing = [normalization, normalization, normalization, normalization, monai_transform, monai_transform]*3
    interp_sizes = [256 for i in range(18)]
    num_samples = [512 for i in range(18)]
    backgrounds = ["original" for i in range(6)]# + ["random_vec" for i in range(6)] + ["full_original" for i in range(6)]
    #classifier_block_sizes = [6, 3, 6, 3]
    #classifier_max_pool_sizes = [3, 1, 3, 1]
    classifier_block_sizes = [3 for i in range(18)] #unused
    classifier_max_pool_sizes = [3, 1, 3, 1, 3, 1]*3
    #classifier_pixel_pool_sizes = [1, 3, 1, 3, 1, 3]*3
    #classifier_max_pool_sizes = [1, 1, 1, 1, 1, 1]*3
    classifier_pixel_pool_sizes = [1, 1, 1, 1, 1, 1]*3

    gan_block_size = [2 for i in range(18)]
    #classifier_max_pool_sizes = [5, 3, 5, 3]

    gan_block_sizes = [2]

    to_use = [i+1 for i in range(18)]
    #to_use = [1, 2, 4]

    for i, gan_str in enumerate(gan_models):
        if not i in to_use:
            continue
        print("-----------------------------------------------------------------------")
        print(gan_str)
        print("background: " + backgrounds[i])
        print("-----------------------------------------------------------------------")
        classifier = load_model(classifiers[i])
        gan = load_gan(gan_str)

        gan_layer_str = gan_layers[i]
        gan_layer = eval(gan_layers[i])
        classifier_layer = eval(classifier_layers[i])

        #dataloader = get_imgs_as_dataloader(num_samples_to_get=256, gan_model=gan_str)
        #print("variability:")
        #get_model_variability(classifier, dataloader, classifier_layer, preprocessing=preprocessing[i])

        cossim_mean_values = []
        cossim_std_values = []
        cossim_full_mean_values = []
        cossim_full_std_values = []
        cossim_masked_mean_values = []
        cossim_masked_std_values = []


        #cossim_original_img_ = []
        #cossim_mask_original_img_ = []


        for gan_block_size in gan_block_sizes:

            cossim_ = []
            cossim_full_ = []
            cossim_mask_ = []
            L1_loss_ = []
            L1_loss_full_ = []
            L1_loss_mask_ = []

            n = 1

            torch.cuda.empty_cache()
            print("metrics use original image:")
            for _ in range(n):
                cossim, cossim_full, cossim_mask, L1_loss, L1_loss_full, L1_loss_mask = run_metrics(classifier, classifier_layer, gan, gan_layer, interp_size=interp_sizes[i],
                        image_post_processing=preprocessing[i], num_samples=num_samples[i], classifier_block_size=classifier_block_sizes[i],
                        classifier_max_pool_kernel_size=classifier_max_pool_sizes[i], image_save_prefix=gan_str + "_" + gan_layers[i] + "_" + classifier_layers[i] + "_" + str(interp_sizes[i]),
                        gan_block_size=gan_block_size, classifier_pixel_pool_size=classifier_pixel_pool_sizes[i],
                        save_images=True, background=backgrounds[i], use_grid_mask=True, classifier_pixel_erosion=False)
                        #batch_size=32, batch_size_img_generator=16)

                # Append the results to corresponding lists
                cossim_.append(cossim)
                cossim_full_.append(cossim_full)
                cossim_mask_.append(cossim_mask)
                L1_loss_.append(L1_loss)
                L1_loss_full_.append(L1_loss_full)
                L1_loss_mask_.append(L1_loss_mask)


            # Calculate mean and standard deviation for each list
            cossim_mean = np.mean(cossim_)
            cossim_full_mean = np.mean(cossim_full_)
            cossim_mask_mean = np.mean(cossim_mask_)
            L1_loss_mean = np.mean(L1_loss_)
            L1_loss_full_mean = np.mean(L1_loss_full_)
            L1_loss_mask_mean = np.mean(L1_loss_mask_)

            cossim_std = np.std(cossim_)
            cossim_full_std = np.std(cossim_full_)
            cossim_mask_std = np.std(cossim_mask_)
            L1_loss_std = np.std(L1_loss_)
            L1_loss_full_std = np.std(L1_loss_full_)
            L1_loss_mask_std = np.std(L1_loss_mask_)

            # Print mean and standard deviation of each list
            print("cossim mean:", cossim_mean, "std:", cossim_std)
            print("cossim_full mean:", cossim_full_mean, "std:", cossim_full_std)
            print("cossim_mask mean:", cossim_mask_mean, "std:", cossim_mask_std)
            print("L1_loss mean:", L1_loss_mean, "std:", L1_loss_std)
            print("L1_loss_full mean:", L1_loss_full_mean, "std:", L1_loss_full_std)
            print("L1_loss_mask mean:", L1_loss_mask_mean, "std:", L1_loss_mask_std)

            cossim_mean_values.append(cossim_mean)
            cossim_std_values.append(cossim_std)
            cossim_full_mean_values.append(cossim_full_mean)
            cossim_full_std_values.append(cossim_full_std)
            cossim_masked_mean_values.append(cossim_mask_mean)
            cossim_masked_std_values.append(cossim_mask_std)




if __name__ == "__main__":
    run_eval()
