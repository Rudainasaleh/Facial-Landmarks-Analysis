## Facial-Landmarks-Analysis
This project involves analyzing 3D facial landmarks using three classic machine learning algorithms: Random Forest, Support Vector Machine (SVM), and Decision Tree. The goal is to classify facial landmarks and evaluate the performance of these classifiers. A report with experimental results and discussions will be submitted.

# Instructions
Follow the instructions below to complete the project:

Programming Language: Use Python for implementing the project. You can import any necessary modules. Create a Python file named Project1.py, and feel free to split your program into multiple files if required.

Run Machine Learning Classifiers: Run the following machine learning classifiers:

Random Forest (RF)
Support Vector Machine (SVM)
Decision Tree (TREE)
10-Fold Cross Validation: Perform 10-fold cross-validation for each of the classifiers mentioned above. Implement subject-independent cross-validation, where the total number of subjects is divided into 10 folds. In each iteration, use 9 folds for training and 1 fold for testing. Create the cross-validation code by yourself. You can refer to the example "classify.py" provided on Canvas for guidance.

Data Availability: The data required to train and evaluate the classifiers using 10-fold subject-independent cross-validation is available in the FacialLandmarks.zip file. The zip file size is approximately 105 MB, and once uncompressed, it is around 200 MB. The directory structure within the zip file is as follows:

/BU4DFE_BND_V1.1
/Subject directories (e.g., F001)
/Expressions (e.g., Angry)
.BND files (text data files)
Data Format: The .bnd files contain an index and one facial landmark per row. Each row represents the x, y, and z coordinates of a landmark. Ignore the index and consider only the coordinates. There are a total of 83 (or 84) rows, corresponding to 83 landmarks. Some files may have an additional empty row.

Three Experiments: Perform the following experiments, each with 10-fold cross-validation using RF, SVM, and TREE classifiers:
a. Use raw 3D landmarks: The feature vector will have a size of 249, considering the 83 (x, y, z) coordinates.
b. Translate raw 3D landmarks: Calculate the average 3D landmark (x, y, and z) and subtract it from each of the 3D landmarks to center the face approximately at the origin.
c. Rotate raw 3D landmarks 180 degrees: Rotate each 3D point over the x-axis, y-axis, and z-axis by 180 degrees. Refer to the provided math formula to perform the rotation.

Performance Evaluation: For each experiment and classifier, calculate the confusion matrix, accuracy, precision, and recall. Average the values for each cell across the 10 folds. Save the results from each classifier run, such as saving them to a text file or another appropriate format.

Command Parameters: Your program must accept the following command parameters:

Classifier to run: RF, SVM, TREE
Data type to use: Original, Translated, RotatedX, RotatedY, RotatedZ
Data directory
Example python run: python Project1.py RF RotatedX ./BU4DFE_BND_V1.1
This example will use the Random Forest classifier on rotated (over the x-axis) data.
