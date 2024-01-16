import numpy as np
from scipy import ndimage
import cv2
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


class ImageBlending:
  
    def __init__(self):
        oneD_kernel = cv2.getGaussianKernel(5, sigma=0)
        self.kernel = oneD_kernel*oneD_kernel.T

    def saving_final_image(self, img_name, R_reconstructed, G_reconstructed, B_reconstructed):
        if not os.path.exists(os.path.join(Path(__file__).resolve().parents[1], "output_images")):
            os.makedirs(os.path.join(Path(__file__).resolve().parents[1], "output_images"))

        final_image = cv2.merge((R_reconstructed, G_reconstructed, B_reconstructed))

        cv2.imwrite(os.path.join(Path(__file__).resolve().parents[1], "output_images", img_name + ".jpg"), final_image)

    def saving_reconstruct_levels(self, img_name, R_reconstructed_list, G_reconstructed_list, B_reconstructed_list):
        if not os.path.exists(os.path.join(Path(__file__).resolve().parents[1], "reconstruct_level")):
            os.makedirs(os.path.join(Path(__file__).resolve().parents[1], "reconstruct_level"))
        if not os.path.exists(os.path.join(Path(__file__).resolve().parents[1], "reconstruct_level", img_name)):
            os.makedirs(os.path.join(Path(__file__).resolve().parents[1], "reconstruct_level", img_name))
        for i, r, g, b in zip(range(len(B_reconstructed_list)), R_reconstructed_list, G_reconstructed_list, B_reconstructed_list):
            img_dir = os.path.join(os.path.join(Path(__file__).resolve().parents[1], "reconstruct_level", img_name, str(i)+"thLevel.jpg"))
            cv2.imwrite(img_dir, cv2.merge((r, g, b)))

    def saving_pyramids_levels(self, img_name, R_pyramids,G_pyramids,B_pyramids):
        if not os.path.exists(os.path.join(Path(__file__).resolve().parents[1], "pyramids_level")):
            os.makedirs(os.path.join(Path(__file__).resolve().parents[1], "pyramids_level"))

        if not os.path.exists(os.path.join(Path(__file__).resolve().parents[1], "pyramids_level", img_name)):
            os.makedirs(os.path.join(Path(__file__).resolve().parents[1], "pyramids_level", img_name))

        pyramid_types = ["first_image_laplacian", "second_image_laplacian", "mask_gaussian", "blended"]

        for pyramid_type in pyramid_types:
            if not os.path.exists(os.path.join(Path(__file__).resolve().parents[1], "pyramids_level", img_name, pyramid_type)):
                os.makedirs(os.path.join(Path(__file__).resolve().parents[1], "pyramids_level", img_name, pyramid_type))

        imagePyramids = list(zip(*[R_pyramids,G_pyramids,B_pyramids]))

        firstImageLaplacianPyramids = list(imagePyramids[0])
        secondImageLaplacianPyramids = list(imagePyramids[1])
        maskGaussianPyramids = list(imagePyramids[2])
        blendedPyramids = list(imagePyramids[3])

        for i in range(len(firstImageLaplacianPyramids[0])):

            img_dir = os.path.join(os.path.join(Path(__file__).resolve().parents[1], "pyramids_level", img_name, "first_image_laplacian", str(i)+"thLevel.jpg"))
            img = cv2.merge((firstImageLaplacianPyramids[0][i], firstImageLaplacianPyramids[1][i], firstImageLaplacianPyramids[2][i]))
            norm_image = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)
            cv2.imwrite(img_dir, norm_image) 
            
            img_dir = os.path.join(os.path.join(Path(__file__).resolve().parents[1], "pyramids_level", img_name, "second_image_laplacian", str(i)+"thLevel.jpg"))
            img = cv2.merge((secondImageLaplacianPyramids[0][i], secondImageLaplacianPyramids[1][i], secondImageLaplacianPyramids[2][i]))
            norm_image = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)
            cv2.imwrite(img_dir, norm_image)  
    
            img_dir = os.path.join(os.path.join(Path(__file__).resolve().parents[1], "pyramids_level", img_name, "mask_gaussian", str(i)+"thLevel.jpg"))
            img = cv2.merge((maskGaussianPyramids[0][i], maskGaussianPyramids[1][i], maskGaussianPyramids[2][i]))
            norm_image = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)
            cv2.imwrite(img_dir, norm_image)

            img_dir = os.path.join(os.path.join(Path(__file__).resolve().parents[1], "pyramids_level", img_name, "blended", str(i)+"thLevel.jpg"))
            img = cv2.merge((blendedPyramids[0][i], blendedPyramids[1][i], blendedPyramids[2][i]))
            norm_image = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)
            cv2.imwrite(img_dir, norm_image)


    def masking(self, image1, image2):

        roi1 = cv2.selectROI('Select an Object', image1)
        roi2 = cv2.selectROI('Select a region', image2)
        cv2.destroyAllWindows()

        mask = np.zeros_like(image1[:, :, 0])

        mask[int(roi2[1]):int(roi2[1] + roi2[3]), int(roi2[0]):int(roi2[0] + roi2[2])] = 255

        x_dir = int(roi2[0]-roi1[0])
        y_dir = int(roi2[1]-roi1[1])

        translation_matrix = np.float32([ [1, 0, x_dir], [0, 1, y_dir] ])

        return mask, translation_matrix

    def upsampling(self, old_image, image):

        image_up = np.zeros((old_image.shape[0], old_image.shape[1]))
        image_up[::2, ::2] = image

        return ndimage.filters.convolve(image_up, 4*self.kernel, mode='constant')
                                
    def downsampling(self, image):

        image_blur = ndimage.filters.convolve(image, self.kernel, mode='constant')
        return image_blur[::2, ::2]                                
               
                                      
    def gaussian_and_laplacian_pyramids(self, image):

        Gaussians = [image]
        Laplacians = []

        while image.shape[0] >= 2 and image.shape[1] >= 2:
            image = self.downsampling(image)
            Gaussians.append(image)

        for i in range(len(Gaussians) - 1):
            Laplacians.append(Gaussians[i] - self.upsampling(Gaussians[i], Gaussians[i + 1]))

        return Gaussians[:-1], Laplacians

    def pyramid_blending(self, image1_color, image2_color, mask):
        [GA, LA] = self.gaussian_and_laplacian_pyramids(image1_color)
        [GB ,LB] = self.gaussian_and_laplacian_pyramids(image2_color)

        [Gmask, LMask] = self.gaussian_and_laplacian_pyramids(mask)

        pyramids = [LA, LB, Gmask]

        blend = []
        for i in range(len(LA)):

            LS = Gmask[i]/255*LA[i] + (1-Gmask[i]/255)*LB[i]
            blend.append(LS)

        pyramids.append(blend)

        return (blend, pyramids)

    def reconstruct(self, belended_pyramid, pyramids):
        revPyramid = belended_pyramid[::-1]
        reconstructed = revPyramid[0]
        reconstructed_list = [reconstructed]

        for i in range(1, len(revPyramid)):
            reconstructed = self.upsampling(revPyramid[i], reconstructed) + revPyramid[i]
            reconstructed_list.append(reconstructed)

        return reconstructed, reconstructed_list, pyramids

    def color_blending(self, img_name, img1, img2, mask):
        img1R,img1G,img1B = cv2.split(img1)
        img2R,img2G,img2B = cv2.split(img2)
        R_reconstructed, R_reconstructed_list, R_pyramids = self.reconstruct(*self.pyramid_blending(img1R, img2R, mask))
        G_reconstructed, G_reconstructed_list, G_pyramids = self.reconstruct(*self.pyramid_blending(img1G, img2G, mask))
        B_reconstructed, B_reconstructed_list, B_pyramids = self.reconstruct(*self.pyramid_blending(img1B, img2B, mask))

        self.saving_pyramids_levels(img_name, R_pyramids, G_pyramids, B_pyramids)
        self.saving_reconstruct_levels(img_name, R_reconstructed_list, G_reconstructed_list, B_reconstructed_list)
        self.saving_final_image(img_name, R_reconstructed, G_reconstructed, B_reconstructed)


img_name = "balloon"  #monolisa, bird, fish, mirror, phone, sky, fruit, turtle, balloon

if os.path.exists(os.path.join(Path(__file__).resolve().parents[1], "even", img_name + "1.jpg")):
    img1 = cv2.imread(os.path.join(Path(__file__).resolve().parents[1], "even", img_name + '1.jpg'))
    img2 = cv2.imread(os.path.join(Path(__file__).resolve().parents[1], "even", img_name + '2.jpg'))

else:
    img1 = cv2.imread(os.path.join(Path(__file__).resolve().parents[1], "odd", img_name + '.jpg'))
    img2 = cv2.imread(os.path.join(Path(__file__).resolve().parents[1], "odd", img_name + '.jpg'))

imageBlending = ImageBlending()

mask, translation_matrix = imageBlending.masking(img1,img2)

img1 = cv2.warpAffine(img1, translation_matrix, (img1.shape[1],img1.shape[0]))

imageBlending.color_blending(img_name, img1, img2, mask)

     