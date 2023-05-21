import os
import sklearn
from sklearn import metrics
from sklearn import svm, datasets
import matplotlib.pyplot as plt
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
import argparse
from math import acos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




# a function that load the data from the provided data folder 
def getData(tracefile):
    X = [] #result array 
    y = [] #expression array
    # a for loop that goes inside the directory for each folder 
    for data_dir in sorted(os.listdir(tracefile)):
        get_expression = sorted(os.listdir(os.path.join(tracefile, data_dir)))
        # another for loop to get inside each exprestion for each data folder
        for exprs in get_expression:
            files = sorted(os.listdir(os.path.join(tracefile, data_dir, exprs)))
            get_bnd = [f for f in files if f.endswith('.bnd')]
            # another for loop that goes throgh all the .bnd data inside that expression 
            for bnd_file in get_bnd:
                with open(os.path.join(tracefile, data_dir, exprs, bnd_file)) as f:
                    #It reads in the data and extracts the threeD in columns( 1, 2, and 3) using np.loadtxt
                    threeD = np.loadtxt(f, skiprows=1, usecols=[1,2,3]).ravel()
                    # 
                    X.append(threeD)
                    y.append(exprs)
    #return the 2 array (x,y)
    return np.array(X), np.array(y)

# translated function 
def translated(X):
    X = X.astype(float) #convert array to a float
    # get the average of x, y, and z data using mean
    avg_x = np.mean(X[:, 0::3], axis=1)
    avg_y = np.mean(X[:, 1::3], axis=1)
    avg_z = np.mean(X[:, 2::3], axis=1)
    # a for loop that goes through X
    for i in range(X.shape[0]):
        #subtract the avg from x,y, and z and ubdate x with the new values
        X[i, 0::3] -= avg_x[i]
        X[i, 1::3] -= avg_y[i]
        X[i, 2::3] -= avg_z[i]
    
    return X

# function to rotate landmark data around the x, y, or z axis by 180 degrees
def rotated_xyz(X, axis):
    # define pi value 
    pi = round(2*acos(0.0), 3)
    # check what rotated axis was input
    # we calculte the rotation depending in what kind of rotation 
    if axis == 'x':
        Rotation = np.array([[1, 0, 0],
                      [0, np.cos(pi), np.sin(pi)],
                      [0, -np.sin(pi), np.cos(pi)]])
        
    elif axis == 'y':
        Rotation = np.array([[np.cos(pi), 0, -np.sin(pi)],
                      [0, 1, 0],
                      [np.sin(pi), 0, np.cos(pi)]])
    elif axis == 'z':
        Rotation = np.array([[np.cos(pi), np.sin(pi), 0],
                      [-np.sin(pi), np.cos(pi), 0],
                      [0, 0, 1]])
    
    X = X.astype(float) 
    #make a new array with same size as X
    rotated = np.zeros_like(X)

    # a foor loop that goes through each column in X
    for i in range(X.shape[1]):
        col = X[:, i] # col is i column in X
        #if statment to check col is in shpe (3, )
        if col.shape == (3,):
            # if it is we multiply it with the rotation 
            rotated[:, i] = np.matmul(Rotation, col)
        else:
            rotated[:, i] = col
            

    return rotated

# function to run the classifier with k-fold cross validation on the specified data with 10 folds
def run_classifier(data, classifier_type, num_folds=10):
    #kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    kf = KFold(n_splits=num_folds)
    metrics = {'precision': [], 'recall': [], 'accuracy': []}
    for train_idx, test_idx in kf.split(data['X']):
        X_train, X_test = data['X'][train_idx], data['X'][test_idx]
        y_train, y_test = data['y'][train_idx], data['y'][test_idx]

        # preprocess data, check what data type is been using to implimint it
        if data['type'] == 'Original':
            pass
        elif data['type'] == 'Translated':
            X_train = translated(X_train)
            X_test = translated(X_test)
        else:
            X_train = rotated_xyz(X_train, data['type'][6:])
            X_test = rotated_xyz(X_test, data['type'][6:])

        # Then, we train theh classifier
        if classifier_type == 'RF':
            #default RF
            clf = RandomForestClassifier()
        elif classifier_type == 'SVM':
            #default SVM 
            clf = svm.SVC()
        elif classifier_type == 'TREE':
            #default TREE
            clf = DecisionTreeClassifier()
        else:
            print("wrong classifier")

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        


        # evaluate the model using precision, recall, and accuracy metrics
        plot_3d_scatter(X_test, 'original')
        precision, recall, accuracy, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['accuracy'].append(accuracy)
        cm = confusion_matrix(y_test, y_pred)


    # print results, the confusion matrix and the average precision, recall, and accuracy metrics.
    print(cm)
    print(f"Average precision: {np.mean(metrics['precision'])}")
    print(f"Average recall: {np.mean(metrics['recall'])}")
    print(f"Average accuracy: {np.mean(metrics['accuracy'])}")
    #return {'report': report, 'confusion_matrix': cm}

#function to plot the samples
def plot_3d_scatter(X, data_type):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='b', label='original')
    ax.set_title('Translated RF Plot')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.savefig('RF_Translated.1.jpg')

    #ax.legend()

    #plt.show()



def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='project 1.')
    parser.add_argument('classifier_type', type=str, help='Type of classifier to use.') 
    parser.add_argument('type', type=str, help='Type of data preprocessing to apply.')
    parser.add_argument('tracefile', type=str, help='Directory path.')
    
    
    args = parser.parse_args()

    X, y = getData(args.tracefile) # we start by gitting and loading the data from the directory file

    
    #if (args.type == 'Original'):
   
    #elif args.type == 'Translated':
        #X = translated(X)
        #ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='b', label='Translated')
        #plot_3d_scatter(X, args.type)
    #elif args.type.startswith('Rotated'):
        #X = rotated_xyz(X, args.type[7:])
        #ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='g', label='Rotated')


    
    # Run classifier
    best_cm = None
    
    run_classifier({'X': X, 'y': y, 'type': args.type}, args.classifier_type) # we run the classifier from the data we had get 
    
if __name__ == '__main__':
    main()