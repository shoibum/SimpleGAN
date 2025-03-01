## Brain MRI GAN

This repository contains a simple Generative Adversarial Network (GAN) implementation for generating synthetic brain MRI images.

**Dataset:**

The model is trained on the Brain MRI dataset from Kaggle. This dataset contains two folders: `Yes` (containing images of patients with brain tumors) and `No` (containing images of patients without brain tumors).

**Dependencies:**

* PyTorch
* Torchvision

**Usage:**

1.  **Download the dataset:**
    * Download the Brain MRI dataset from Kaggle.
    * Extract the dataset into a folder named `data/brain_mri`.
2.  **Install the dependencies:**
    ```bash
    pip install torch torchvision
    ```
3.  **Run the code:**
    ```bash
    python gan_brain_mri.py
    ```

**Output:**

The generated images will be saved in the `output` folder.

**Note:**

* The code is designed to run on a GPU if available.
* The training process may take some time depending on your hardware.
* The generated images may not be perfect, but they should resemble brain MRI scans.

**Future Work:**

* Improve the quality of the generated images.
* Train the model on a larger dataset.
* Experiment with different GAN architectures.

**Credits:**

* The code is based on the PyTorch GAN tutorial.
* The dataset is from Kaggle.
