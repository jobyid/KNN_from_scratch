# KNN_from_scratch

A from scratch implementation of the k-NN classification algorithm.

## Model

The model calls takes 1 parameter and offers 2 methods. To work with KNN you need to have a set of data which has been classified already. This act as training for the model, you can then input new row of the same data you need to classify, the model will then find the most likely classification based on how close the new data points are to the other points already classified. 
The model can only take in numeric data and will not work with any NaN values.  

### Class Parameters:
- k :- This is the number of neighbours to consider when making the classification. 

### Methods:

- predict: 
	- This is your core method for prediction
	- Parameters:
		- t: the test data as an np.array which you wish to classify.
		- X: the feature data to train the model on
		- y: the classifications of the feature training data  
- evaluate:
	- Use this method to check how accurate the model is. It can only be run after predict has been run. 
	- Parameters:
		- true: the true values of the data you have asked the model to classify 

A few things to note, all the data must be in the form of np.array there cannot be any NaN values. It will raise an exception when NaN are present.  

## Command Line
The repo can also be used from the command line. To do so, navigate to the the folder where you have the repo using the 'cd' command. 

`python knn_main.py `

runs the program you will be prompted for file paths required. The files must be csv files. 
Here are all the command line options: 

```
Options:
  -k, --set_k INTEGER       Set k which represents the number of neighbours to
                            consider when making predictions  [default: 5]

  -td, --train_data TEXT    path to training data features without target as np
                            array

  -tt, --train_target TEXT  The correct classifications of the training data as
                            an np array

  -t, --test_data TEXT      The data you want to be classified based on the
                            training data as np array with same dimensions as
                            training data

  --save / --no_save        Save the results to a csv
  --help                    Show this message and exit.
  ```
  