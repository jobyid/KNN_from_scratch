import numpy as np
import click
from knn import Knn
from data_prep import process_data_command_line as dp

@click.command()
@click.option("--set_k",'-k',default=5, type=int,show_default=True, help="Set k which represents the number of neighbours to consider when makeing predictions")
@click.option('--train_data','-td', prompt="Please provide a file path to training features. must be a csv",help='path to traning data features without target as np array')
@click.option('--train_target','-tt', prompt="Please provide a file path to training target. must be a csv",help="The correct classifactions of the training data as an np array")
@click.option('--test_data', '-t', prompt="Please provide a file path to data to test. must be a csv",help="The data you want to be classfied bassed on the trainign data as np array with same dimensions as training data")
@click.option('--save/--no_save',  help="Save the results to a csv")

def classify(set_k,train_data, train_target,test_data, save):
    print("Lets get classifying")
    knn = Knn(set_k)
    X_train, X_test, y_train = dp(train_data,train_target, test_data)
    pred = knn.predict(X_test,X_test,y_train)
    click.echo(pred)
    click.echo(save)
    if save:
        np.savetxt('predictions.csv',pred)


if __name__ == '__main__':
    classify()
