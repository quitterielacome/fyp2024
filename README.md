# Automated Prediction of MGMT Promoter Methylation in Gliomas

This project develops and evaluates three deep-learning models of different complexities to predict the methylation status of the MGMT promoter from MRI scans of glioma patients.

## Aims and Motivations

This project is dedicated to enhancing glioma diagnosis by using deep learning to non-invasively predict the MGMT promoter methylation status from MRI scans. We evaluate three models of varying complexities—ranging from simple to advanced architectures—to determine if more complex models consistently yield better predictive accuracy and efficiency. This comparative analysis aims to identify the most effective model configuration for guiding personalised chemotherapy treatments, ultimately improving patient care by circumventing invasive biopsy procedures.


## Models Evaluated
- **Baseline Model**: A simple convolutional neural network (CNN).
- **Enhanced Baseline Model**: Baseline CNN with data augmentation.
- **Transfer Learning Model**: Utilises the VGG16 architecture pretrained on ImageNet.


## Dataset

The project utilises the RSNA-MICCAI Brain Tumor Radiogenomic Classification dataset, which is substantial in size (over 130GB). Due to this large size, the dataset is not included in the repository and must be downloaded separately. The dataset consists of 1006 patients' brain MRI images in FLAIR, T1w, T1wCE and T2w modalities, stored in DICOM format.

## Downloading the Dataset from Kaggle

### Using the Provided Notebook Link

The easiest way to access the data is through the link provided in the first cell of the Colab notebook included in the project submission. This link is configured to automatically download and set up the dataset, but please note it is valid for a limited time (typically 6 days).

If the link in the notebook has expired, either:

1. Create a new notebook within the [Kaggle dataset page](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/data).
2. Use the 'Add Data' option within Kaggle to attach the dataset to you notebook.
3. Open the notebook in Google Colab by selecting the 'Open in Colab' option provided by Kaggle.
4. Kaggle will generate a new cell with code to import the dataset, which you can copy into the MGMT methylation project notebook to replace the expired code.
 
or 
1. Request access to the original Kaggle notebook from the project owner, which can then be used to regenerate a fresh link.
2. Once access is granted, open the notebook on Kaggle and select the 'Open in Colab' option. This will generate a new Colab notebook with an updated dataset import link.

### Downloading the Dataset from Kaggle (Alternative Option)

As a last resort, if the above methods do not work, or if making the project public is not feasible, the dataset can be manually downloaded from Kaggle:

1. Visit the [Kaggle dataset page](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/data).
2. Create a Kaggle account or log in if you already have one.
3. Agree to the competition rules to gain access to the dataset.
4. Use the Kaggle API or the direct download option to download the dataset onto your local machine.

Please be aware that downloading and storing this large dataset requires considerable storage space and bandwidth.


## Prerequisites

Before running this project, make sure you have Python 3.8+, and pip installed on your machine. Familiarity with deep learning concepts and experience with Google Colab or Jupyter Notebooks is recommended.

## Installation

Before running the application, follow these steps to set up your environment:

### Clone the repository to your local machine:

```bash
git clone https://github.com/quitterielacome/fyp2024.git
cd fyp2024
```

### Installing Dependencies

This project requires Python and several packages which are listed in requirements.txt. Ensure you have Python installed, then run the following command in your terminal to install the necessary packages

```bash
pip install -r requirements.txt
```
## Training the Models

For those interested in training the models themselves or understanding the entire workflow from data exploration to evaluation, the process is documented comprehensively in the Jupyter notebook `pred_MGMT.ipynb`.

### Steps to Train the Models Using the Notebook

1. **Open the Notebook**: The notebook `pred_MGMT.ipynb` is located in the main directory of the project. You can open this notebook using Jupyter Notebook or JupyterLab, or through Google Colab if you prefer an online environment.

```bash
   cd pred_MGMT/
   jupyter notebook pred_MGMT.ipynb
```

Alternatively, if you are using Google Colab:

Navigate to Google Colab
Click on 'File' > 'Open notebook' > 'GitHub' tab
Enter the GitHub URL or search for the repository and select the notebook.

2. **Install Required Libraries:** Before running the notebook, ensure that all required libraries are installed. You can do this within the notebook by running cells containing pip install commands, or manually in your environment as described in the Installation section.

3. **Run the Notebook:** Execute the cells sequentially in the notebook:
- Data Loading and Preprocessing: The notebook will include cells to load and preprocess the data.
- Exploratory Data Analysis (EDA): Cells that visually and statistically explore the data.
- Model Training: Cells that set up and train the different models.
- Evaluation: Cells that evaluate the model performance using appropriate metrics.
4. **View Results:** Results from EDA, model training, and evaluations are displayed directly in the notebook. Look for cells containing visualizations and output tables.
5. **Save and Export Models:** The notebook should also contain instructions or code cells for saving the trained models, which are necessary for deploying or using the models in the application.

### Troubleshooting
If you encounter errors related to missing libraries or dependencies, refer back to the Installation section to ensure all necessary software is correctly installed.
If cells fail to execute in order, it might be necessary to restart the kernel and clear output before re-running the notebook.

## Web App

This section provides step-by-step instructions to run the web application and analyse MRI scans for MGMT promoter methylation status prediction.

### Running the Web Application

1. **Open Terminal**: Open your command line interface (CLI).

2. **Navigate to the Project Directory**: Change to the directory containing the project files, specifically where `app.py` is located.
```bash
cd project-site
```

3. Start the Application: Launch the web application by running:
```bash
python3 app.py
```

Access the Web Application: Open a web browser and go to http://localhost:5000 (or whatever URL the application prints to the terminal).

### Using the Application
Once the application is running, follow these steps to use it:

1. **Enter Patient Name:** On the home page, enter the name of the patient whose MRI scans you want to analyse.
2. **Upload MRI Scan:** Upload the DICOM folder for the patient. Ensure that the folder structure is correct and that the files are in the supported DICOM format.
3. **Submit for Analysis:** Click the 'Analyse' button to submit the data. The application will process the MRI scans and use the pretrained models to predict the methylation status.
4. **View Predictions:** The results, including the prediction and confidence rate for each model, will be displayed on the results page. Review the predictions and the associated confidence levels.

5. **Navigate History:** If enabled, you can view past analyses by navigating to the 'History' section via the application's menu. This feature lets you compare current results with past results for the same or different patients.

### Tips for Troubleshooting
Ensure that all dependencies are properly installed as described in the 'Installation' section.
Check that the DICOM files are not corrupted and conform to the expected format.
If the application does not start, check the terminal for error messages that may indicate what went wrong.

## Results
The models achieved the following performance metrics:

1. Baseline Model: Accuracy of 80%, AUC of 0.874.
2. Enhanced Baseline Model: Accuracy of 83%, AUC of 0.889.
3. Transfer Learning Model: Accuracy of 88%, AUC of 0.939.

These results indicate that the Transfer Learning Model outperformed the other models in predicting MGMT methylation status.


## Demonstrations
- Screenshots and outputs are included within the Colab notebook.
- For a more detailed walkthrough, refer to the presentation linked [here](presentation).

## Contact
For any queries or further discussion, please contact [Quitterie Lacome d'Estalenx](quitterie@estalenx.com).

## License

This project is licensed under the MIT License.

## Acknowledgments

I would like to express my deepest appreciation to my supervisor, Sareh Rowlands, for her guidance and support throughout this year.
