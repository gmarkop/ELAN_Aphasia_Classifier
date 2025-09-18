# ELAN_Aphasia_Classifier
Machine Learning Classification of aphasic patients vs controls based on ELAN annotation data

## Overview
The ELAN Classifier Tool is an application that allows you to:
1. Train machine learning models to detect aphasia based on speech patterns
2. Upload ELAN (.eaf) files and predict whether they belong to an aphasic or control patient

## Features
- Upload and process ELAN annotation files
- Extract linguistic features automatically (words per minute, grammaticality ratio, etc.)
- Train Random Forest or SVM classifiers with regularization to prevent overfitting
- Visualize model performance with classification reports, confusion matrices, and ROC curves
- Make predictions on new ELAN files

## Installation Requirements
- Python 3.6+
- Required packages: streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn, pympi-ling

## How to Use

### Training a Model
1. Navigate to the "Train Model" tab
2. Upload your training data CSV files:
   - Combined Statistics CSV: Contains participant statistics (words per minute, grammatical utterances, etc.)
   - Combined Pause Analysis CSV: Contains pause duration information
3. Configure model options:
   - Select model type (Random Forest or SVM)
   - Choose feature selection method
   - Adjust model hyperparameters
4. Click "Train Model" to train the classifier
5. Review model performance metrics and visualizations

### Predicting from ELAN Files
1. Navigate to the "Predict from ELAN File" tab
2. Upload an ELAN (.eaf) file
3. The application will:
   - Extract linguistic features from the ELAN file
   - Display the extracted features
   - Make a prediction (Aphasic or Control)
   - Show prediction probabilities

## ELAN File Requirements
For accurate feature extraction, your ELAN files should include:
- Speech/utterance tiers with annotations
- Pause tiers with annotations
- Grammaticality markers (e.g., "*" for ungrammatical utterances)
- Filled pause markers (e.g., "um", "uh", "er")

## Tips for Best Results
- Ensure consistent annotation practices across all ELAN files
- Mark ungrammatical utterances clearly (e.g., with "*" at the beginning)
- Annotate all pauses, including filled pauses
- For training data, include a balanced set of aphasic and control samples

## Troubleshooting
- If feature extraction fails, check that your ELAN file contains the necessary tiers
- If prediction results seem incorrect, try retraining the model with more data
- For any issues with file uploads, ensure your files are in the correct format

## Contact
For questions or support, please contact the development team.
