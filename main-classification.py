import argparse
import pandas as pd
from Classifiers import Classifier

# parses the arguments
parser = argparse.ArgumentParser(description='file to check')
parser.add_argument('--clf', help='Specify Classifier. RF|ENN|DNN', type=str)
parser.add_argument('--method', help='Specify Method: ho|cv|cl.', type=str)
parser.add_argument('--load', default=False, const=True, nargs='?', help='Loads an already Initialized Classifier.')
parser.add_argument('--train', help='Specify Training Set.', type=str)
parser.add_argument('--test', help='Specify Testing Set.', type=str)

args = parser.parse_args()


method = 'ho'
if args.method == 'cv' or args.method == 'custom' or args.method == 'cl':
    method = args.method
cl = 'RF'
if args.clf == 'ENN':
    cl = 'ENN'
elif args.clf == 'DNN':
    cl = 'DNN'

clf = Classifier(components=25)

# load classifier
if args.load:
    clf = clf.loadObject()
else:
    # initialize classifier and Train Set
    clf.Init_Classifier(choice=cl, estimators=75, threshold=0)

    if args.train:
        trainSet = pd.read_csv(args.train, sep='\t')
        print("ERROR: No Training set was given.")
        exit(0)
    if method == 'cl':
        clf.fit()
        clf.StoreObject()


# initializes the Testing Set and predict
if method == 'cl':
    if args.test:
        clf.TestSet_Construction(pd.read_csv(args.test, sep='\t'), construct=False)
        print("'Classification' mode requires a Testing set.")
        print("ERROR: No Testing set was given.")
        exit(0)
    predictions = clf.predict()


# perform validation
elif method == 'custom':
    clf.Validate_Classification(pd.read_csv('test.csv', sep='\t'))
else:
    clf.Validation(method=method)
