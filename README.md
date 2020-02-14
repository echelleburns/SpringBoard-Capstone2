# Identifying Humans in Drone Footage from Local Beaches
This folder houses code and documents from my second capstone project with SpringBoard. 
<br>
Completed Feb 2020 by Echelle Burns

# Table of Contents: General Information 
<br>
> Documents == contains reports and powerpoint presentations about the progress and results
    of this project
<br>
<br>
> requirements.txt == the packages and versions of packages that were installed in the virtual
    environment to run this model
<br>
<br>
> Data > raw  == contains raw drone images
<br>
> Data > raw > resized == contains raw drone images resized to 960x540 pixels
<br>
> Data > raw > resized > with_people == contains resized drone images that include humans
<br>
> Data > raw > resized > with_people > splits == contains 25 images split from original 
    resized images and converted to HSV
<br>
> Data > processed == contains labeled images that have ellipses around humans
<br>
> Data > processed > dots == contains labeled images with dots at human locations 
<br>
> Data > processed > dots > with_people == contains labeled images from raw images with humans
<br>
> Data > processed > dots > with_people > splits == contains 25 images split from original labeled
    images

# Table of Contents: Order of Execution
<br>
> Data_Processing == contains jupyter notebooks that were used to pre-process data
<br>
> Data_Processing > Labeling_Images == notebook that allows user to label humans in images, either
   by using ellipses to circle an entire human or by using dots to indicate the center of a human, 
   also generates resized images
<br>
> Data_Processing > DataWrangling_PhotoContrasts == notebook that goes through different color scales 
   for the   drone images to see which might be best for model performance
<br>
> Data_Processing > DataWrangling_ImageSlicing == notebook that splits resized images into 25 smaller
   images for model ingestion, also converts images to HSV/grayscale
<br>
<br>
> Exploratory_Data_Analysis == contains jupyter ntoebooks that were used for EDA
<br>
> Exploratory_Data_Analysis > Descriptive_Statistics == notebook that sees how many images (split and 
    full images) contain humans
<br>
<br>
> Neural_Network == contains jupyter notebooks used for exploration of different convolutional neural
    networks via brute force
<br>
> Neural_Network > Building_a_Model == the first notebook used to build a baseline generator and model
<br>
> Neural_Network > Model_Attempts_Log == the notebook that was used to go through a variety of different
    models with changes in the size and number of kernels, different colors of images, different color 
    channels, and split vs full size images
<br>
> Neural_Network > Choosing_Models-Activator_Functions == notebook used to try different activator 
    functions on the best performing model from Model_Attempts_Log with plots to assess convergence
<br>
<br>
> generator.py (do not need to run) == contains the data generator, is called in the model.py file
<br>
> model.py == runs the best fitting model to the dataset and has an early stopping in case the model
    reaches convergence before 50 epochs, saves the resulting model to a hd5 file and presents the
    convergence plot for the training and testing datasets
<br>
> model.hd5 == the resulting model generated from my local drive from the entire dataset
<br>
> load_model.py == example for how to load the hd5 model into your own drive
<br> 
<br>
> Model_Evaluation == contains a notebook that plots the residuals of the models
