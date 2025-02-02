from typing import Callable

import numpy as np
from PIL import Image

from mask import Mask, ColorMask
from model.model import InpaintNN
from util import apply_variant, image_to_array, expand_bounding, find_regions


def test():
    color_file_path = "../decensor_input/mermaid_censored.png"
    colored_img = Image.open(color_file_path)
    model = InpaintNN("")
    is_mosaic = False
    save_image = lambda i, img: print(i, img)
    mask_gen = lambda ori, colored: ColorMask(np.squeeze(colored if is_mosaic else ori, axis=0),
                                              rgb=(0, 1, 0)).display()
    decensor_image_variations(model, colored_img, colored_img, mask_gen, 1, False, save_image)


def decensor_image_variations(model: InpaintNN, ori: Image, colored: Image, mask_gen: Callable[[Image, Image], Mask],
                              variations: int, is_mosaic: bool, callback: Callable[[int, Image], None]):
    for i in range(variations):
        ori = apply_variant(ori, i)
        colored = apply_variant(colored, i)
        mask = mask_gen(ori, colored)
        out = decensor_image(model, mask, ori, colored, is_mosaic)
        out = apply_variant(out, i)
        callback(i, out)


def save_alpha(img):
    if img.mode == "RGBA":
        alpha_channel = np.asarray(img)[:, :, 3]
        alpha_channel = np.expand_dims(alpha_channel, axis=-1)
        return img.convert('RGB'), True, alpha_channel
    else:
        return img, False, None


# TODO: decensor all cropped parts of the same image in a batch (then i need input for colored an array of those images and make additional changes)
def decensor_image(model: InpaintNN, mask: Mask, ori: Image, colored: Image, is_mosaic: bool) -> Image.Image:
    ori, has_alpha, alpha_channel = save_alpha(ori)

    ori_array = image_to_array(ori)
    ori_array = np.expand_dims(ori_array, axis=0)
    mask_arr = mask.find_mask_simple()
    mask_img = mask.display()

    # colored image is only used for finding the regions
    regions = find_regions(colored.convert('RGB'), mask_arr)

    if len(regions) == 0 and not is_mosaic:
        raise Exception("No regions found")

    output_img_array = ori_array[0].copy()

    for region_counter, region in enumerate(regions, 1):
        bounding_box = expand_bounding(ori, region, expand_factor=1.5)
        crop_img = ori.crop(bounding_box)

        # resize the cropped images
        crop_img = crop_img.resize((256, 256))
        crop_img_array = image_to_array(crop_img)

        # resize the mask images
        mask_img = mask_img.crop(bounding_box)
        mask_img = mask_img.resize((256, 256))

        # convert mask_img back to array
        mask_array = image_to_array(mask_img)
        # the mask has been upscaled so there will be values not equal to 0 or 1

        # mask_array[mask_array > 0] = 1
        # crop_img_array[..., :-1][mask_array==0] = (0,0,0)

        if not is_mosaic:
            a, b = np.where(np.all(mask_array == 0, axis=-1))
            crop_img_array[a, b, :] = 0.

        crop_img_array = np.expand_dims(crop_img_array, axis=0)
        mask_array = np.expand_dims(mask_array, axis=0)

        crop_img_array = crop_img_array * 2.0 - 1

        # Run predictions for this batch of images
        pred_img_array = model.predict_image(crop_img_array, crop_img_array, mask_array)

        pred_img_array = np.squeeze(pred_img_array, axis=0)
        pred_img_array = (255.0 * ((pred_img_array + 1.0) / 2.0)).astype(np.uint8)

        # scale prediction image back to original size
        bounding_width = bounding_box[2] - bounding_box[0]
        bounding_height = bounding_box[3] - bounding_box[1]
        # convert np array to image

        pred_img = Image.fromarray(pred_img_array.astype('uint8'))
        pred_img = pred_img.resize((bounding_width, bounding_height), resample=Image.Resampling.BICUBIC)

        pred_img_array = image_to_array(pred_img)

        pred_img_array = np.expand_dims(pred_img_array, axis=0)

        # copy the decensored regions into the output image
        for i in range(len(ori_array)):
            for col in range(bounding_width):
                for row in range(bounding_height):
                    bounding_width_index = col + bounding_box[0]
                    bounding_height_index = row + bounding_box[1]
                    if (bounding_width_index, bounding_height_index) in region:
                        output_img_array[bounding_height_index][bounding_width_index] = pred_img_array[i, :, :, :][row][
                            col]

    output_img_array = output_img_array * 255.0

    # restore the alpha channel if the image had one
    if has_alpha:
        output_img_array = np.concatenate((output_img_array, alpha_channel), axis=2)

    return Image.fromarray(output_img_array.astype('uint8'))


if __name__ == "__main__":
    test()
