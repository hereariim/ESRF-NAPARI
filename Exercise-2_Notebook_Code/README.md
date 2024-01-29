# Notebook Code

## Code

The code is an image inference to detect nuclei from a U-NET model. This U-NET model is saved in tensorflow .h5 format.

A user writes as input:

- `model_path_` : Absolute path of the training model
- `input_`: Absolute path to an RGB image

As output, a user obtains

- `output_` : A binary mask

Algorithm : 

An RGB image is reduced to the input size of the learning model. This model segments this image to obtain a probability mask. To detect kernels, thresholding is applied to obtain a binary mask. Finally, this binary mask is resized to its original size.


## Exercise

### Import data

Upload `esrf.ipynb`, images and model into a repository in your PC.

### Run Notebook

Go to the `esrf.ipynb` spreadsheet.
At the top right, connect the kernel to the napari environment.

⚠️ **If you receive a message from Windows Security Alert, click CANCEL**

Please, write as input the absolute path of `model_path_` and `input_`.

View the result of the segmentation.