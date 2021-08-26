# Train Custom Semantic Segmentation Models with TensorFlow

Semantic segmentation is a computer vision task where we classify each and every individual pixel in an image into different classes. The classes that can be classified are strictly dependent on what we train our model with.

Demo of semantic segmentation (GIF from [PyImageSearch](https://www.pyimagesearch.com/2018/09/03/semantic-segmentation-with-opencv-and-deep-learning/))

![segmentation example](images/opencv_semantic_segmentation_animation.gif)

# Notebooks
There are 3 notebooks used in this GitHub Repo.

1. `1. Image collection.ipynb`

    shows how to label your data with Label Studio to prepare for semantic segmentation.

1. `2. Training - Oxford-IIIT Pet Dataset.ipynb`

    This notebook shows how to preprocess your data, and build a U-net model from scratch in Keras for semantic segmentation. The training was done in Google Colab and you can open it here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1f4xl0Oy_3pQNdVRbyDNC-DNx8M21QT6Q?usp=sharing)

2. `3. Training with Pre-built Model - Brain MRI Segmentation.ipynb`

    The third Jupyter notebook shows how to use the `segmentation_models` library ([GitHub link](https://github.com/qubvel/segmentation_models)) to easily use pre-built architectures such as U-net, LinkNet etc. The notebook was also used in Google Colab for faster training. It can be accessed directly in Google Colab here too: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dcQ_p3CVytqnUrvl3Xie4JbDqS5Q031M?usp=sharing)

# Installation

You only need to install the requirements specified in the `requirements.txt` file using the command below:
```
pip install -r requirements.txt
```

And you also need to install COCO API to use it to create masks. Refer to the instructions below.

## COCO API Installation
COCO API is used to load the COCO JSON annotation format and create masks if necessary. If you are using Label Studio to label the masks, then you will need to install this library for generating the mask images to use for training.

For Windows:
<details><summary> <b>Expand</b> </summary>

1. Download Visual C++ 2015 Build Tools from this [Microsoft Link](https://go.microsoft.com/fwlink/?LinkId=691126) and install it with default selection
2. Also install the full Visual C++ 2015 Build Tools from [here](https://go.microsoft.com/fwlink/?LinkId=691126) to make sure everything works
3. Go to `C:\Program Files (x86)\Microsoft Visual C++ Build Tools` and run the `vcbuildtools_msbuild.bat` file
4. In Anaconda prompt, run
```
pip install cython
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```
</details>

<br>
For Linux:
<details><summary> <b>Expand</b> </summary>

```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools <PATH_TO_TF>/TensorFlow/models/research/
```
</details>
