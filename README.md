## Introduction

  Image blending is a pivotal technique in image processing, integral
to creating visually compelling compositions. Its primary objective is to
seamlessly combine multiple images or regions within a single image,
ensuring natural transitions and harmonious integration.
  The essence of successful image blending lies in the creation of
cohesive and realistic compositions without apparent boundaries or
visible artifacts. Achieving this seamless integration involves merging
distinct images or regions, and managing their colors, tones, and
textures to produce a unified, visually appealing output.
  Methods like Laplacian pyramids provide a structured approach to
achieving this seamless blending by decomposing images into multi-
resolution representations and enabling controlled merging at different
levels of detail.

## Understanding Laplacian Pyramids

  Laplacian pyramids are a multi-scale representation technique used
in image processing and computer vision. They provide a hierarchical
decomposition of an image into a series of levels or layers, each
capturing different levels of detail. The pyramid structure resembles a
stack of images, with each level representing a different scale of the
original image.
  The construction of a Laplacian pyramid involves two main steps:
image Gaussian smoothing and subsequent downsampling, followed by
the computation of the difference between the original image and the
upsampled version of its smoothed counterpart. This process generates a
series of images, each emphasizing different levels of detail, from
coarse features in the lower levels to fine details in the higher levels.

<p align="center"> 
<img src=https://github.com/akifozgur/using-image-pyramid-for-image-blending/blob/main/img/gaussian.png>
</p>

### Key Components of Laplacian Pyramids
  Gaussian Pyramid: Initially, the original image is repeatedly
smoothed and downsampled to create a series of images at different
scales, forming the Gaussian pyramid. Each level of the Gaussian
pyramid represents a progressively downsampled and blurred version of
the original image.
  Laplacian Representation: The Laplacian pyramid is derived by
taking the difference between each level of the Gaussian pyramid and
an upsampled version of its smoothed counterpart from the previous
level. This process results in a set of images capturing the residual
details or high-frequency components at each scale.

## Algorithm Overview

### Masking and Alignment
Accurate selection and alignment of regions ensure precise
blending, minimizing discontinuities and artifacts between different
portions of the image.
### Pyramid Generation
Creating Gaussian and Laplacian pyramids allows for multi-scale
representation, preserving details at different levels. This preservation
of detail facilitates controlled blending at various scales, contributing to
smooth transitions and natural-looking merges.
### Pyramid Blending with Mask
Utilizing the mask during pyramid blending ensures that the
blending process is guided by the defined regions, enabling a controlled
and seamless merge between images or regions based on the mask's
influence.
### Reconstruction for Final Image
The reconstruction step reintegrates the blended pyramid levels
into a single image, ensuring a cohesive and visually consistent output
by combining details from different scales in a seamless manner.

<p align="center"> 
<img src=https://github.com/akifozgur/using-image-pyramid-for-image-blending/blob/main/img/gaussian.png>
</p>

## Some Result Images

First Input                |  Second Input             |   Output Image
:-------------------------:|:-------------------------:|:-------------------------:
![]([https://...Dark.png](https://github.com/akifozgur/using-image-pyramid-for-image-blending/blob/main/input%20images/even/sky1.jpg)https://github.com/akifozgur/using-image-pyramid-for-image-blending/blob/main/input%20images/even/sky1.jpg)  |  ![]([https://...Ocean.png](https://github.com/akifozgur/using-image-pyramid-for-image-blending/blob/main/input%20images/even/sky2.jpg)https://github.com/akifozgur/using-image-pyramid-for-image-blending/blob/main/input%20images/even/sky2.jpg)  |  ![]([https://...Ocean.png](https://github.com/akifozgur/using-image-pyramid-for-image-blending/blob/main/output_images/sky.jpg)https://github.com/akifozgur/using-image-pyramid-for-image-blending/blob/main/output_images/sky.jpg) 
