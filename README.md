# Stable Diffusion WebUI Smart Pre-Processing Extension

## What is this??

As the name would imply, this is an extension for
the [Stable-Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) by @Automatic1111

## What does this do?

It does a few things, actually.

For starters, it utilizes a combination of BLIP/CLIP and YOLOv5 to provide "smart cropping" for images. The primary
subject of each image is identified, the center of that subject is determined, and then the application tries it's best
to crop the image so as to keep as much of the subject as possible within the dimensions specified.

Second, it allows storing the determined image caption directly to the image filename, versus having to create a txt
file along side every image. You can still create a txt file, use existing captions, or not do any captioning at all.

Third, I've provided face restoration and upscaling options for input images. You can select from GFPGAN and Codeformer
for face restoration, and any of the provided upscalers from the "extras' tab to refine/smooth/add detail to your final
output images.

Last, but not least, it offers a rudimentary way to swap the "class" of a captioned image with the specific keyword in
the image. So, if you're trying to train a subject called "xyz" and "xyz" is a dog, you can easily swap "dog" (and "a
dog") wth "xyz" in your captions. Neato!

## Smart Cropping

As I said above, smart cropping utilizes a combination of YOLOV5 object recognition and BLIP/CLIP (and DeepDanBooru)
captioning to automatically determine the most prominent subject in a photo, and automatically crop the subject as
completely as possible. You can also specify a specific subject (dog/cat/woman/house) for the software to find, and skip
the YOLOV5 detection entirely.

<img src="https://user-images.githubusercontent.com/1633844/198178259-e1ade3d6-386e-41b8-9c93-0eca19c82d3d.png" width="550" height="741" />

If a subject is not found, the image will be downscaled and cropped from the center.

## Smart Captioning

This uses all the same features as set in user preferences, with the additional options to save to txt or append to the
image file name.

Additionally, you can swap the generic "class" of the image with a specific subject keyword. This feature may not be
perfect in all cases, but it should still go a long way in speeding up the captioning process.

You can also specify a maximum caption length, which will split the caption by spaces and append words until the maximum
length is reached.

## Post Processing

It's basically a simplified version of the "extras" tab. The idea is that you can do facial restoration and/or use a
model like swinIR or LDSR to smooth or add details to an image. If an image is "actually" upscaled beyond the target
crop size, it will be downscaled again back to the original size.

