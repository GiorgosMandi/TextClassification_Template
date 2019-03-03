import matplotlib.pyplot as plt
import os


from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc

from modules.ArtificialNeuralNetwork import *
from modules.TextPreprocessor import TextPreProcessor
from sklearn.ensemble import RandomForestClassifier


from sklearn import metrics
from sklearn.model_selection import StratifiedKFold


class Classifier:
    def __init__(self, components=100 ):
        self.tpp = TextPreProcessor(components=components)
        self.threshold = 0
        self.choice = None
        self.testSet = None
        self.path = 'Classification'

        if not os.path.exists(self.path):
            os.makedirs(self.path)
            os.makedirs(self.path + '/EncapsulatedFiles')
            os.makedirs(self.path + '/Plots')

        self.le = preprocessing.LabelEncoder()
        self.clf = None
        self.X = None
        self.y = None


    # Calls classifiers Constructors -- Due to the fact that ENN needs train set
    # in order to be constructed, it is initialized in fit/Validation
    def Init_Classifier(self, choice="RF", label_num=6, estimators=100, threshold=0):
        self.threshold = threshold
        self.choice = choice
        if choice == "RF":
            self.clf = RandomForestClassifier(n_estimators=estimators)
        elif choice == "DNN":
            self.clf = DNN(components=self.tpp.components, labels=label_num, path=self.path )
            self.clf.compile()

        elif choice == "ENN":
            # initialization occuring when we receive text ~~~> fit/Validation
            return
        else:
            print('\nError: Wrong choice for classier, the accepted values are:'
                ,"\n\tRF\t->\tRandom Forest\n\tDNN\t->\tDense Neural Network"
                ,"\n\tENN\t->\tEmbedded Neural Network")



    # Fit the chosen classifier into the training set(if the classifier is ENN then it only initialized it)
    # Apply the required pre-process to the training set
    def fit(self):
        if self.X is None:
            print("ERROR: The Training Set has not been initialised")
            return
        if self.choice == "ENN":
            self.clf = ENN(self.X, labels=len(set(self.y)), path=self.path)
            self.clf.compile()
            self.X = self.clf.train
        else:
            self.X = self.tpp.vect_lsi.fit_transform(self.X)

        self.clf.fit(self.X, self.y)




    # Apply the required pre-process to the testing set and then perform the predictions
    # The results are stored in the produced csv
    # The NN_score column contains the scores that the NN gave to the tweets
    # The higher the score the more certain we are that the classification is correct.
    def predict(self):
        if self.testSet is None :
            print("\nError: You must Construct your test set first.\nCall TestSet_Construction")
            return

        self.test = self.tpp.vect_lsi.transform(self.test) if self.choice != 'ENN' else self.clf.TextPreprocess_transform(self.test)
        predictions = self.clf.predict(self.test)
        # for each tweet NN returns an array that contains the probabilities for each class.
        # If a threshold is defined, a tweet is classified as the class with the maximum probability,
        # for which the difference of its probability with the max is smaller than the threshold.
        if self.choice != "RF" and  self.threshold > 0:
            results = []
            for prediction in predictions:
                max_index = np.argmax(prediction)
                max_value = prediction[max_index]
                p_i = [self.le.inverse_transform([max_index])]
                for index,p_val in enumerate(prediction):
                    if index == max_index: continue
                    if max_value - p_val <= self.threshold :
                        p_i.append(self.le.inverse_transform([index]))
                results.append(p_i)
            predictions = results
        else:
            if self.choice != "RF":
                predictions = [ np.argmax(p) for p in predictions]

            predictions = self.le.inverse_transform(predictions)
            predictions_set = set(predictions)
            print("\n\nRESULTS\n")
            for label in predictions_set:
                print(label , ":\t\t", predictions.tolist().count(label))

        if 'Category' in self.testSet.columns:
            self.testSet['Category'] =  pd.Series(predictions)
        else:
            self.testSet.insert(self.testSet.shape[1], 'Category', pd.Series(predictions))
        self.testSet.to_csv(self.path+'/CLASSIFIED_'+self.choice+'_Test.csv',
            sep='\t', index=False, columns=self.testSet.columns)
        return predictions



    # Perform a Validation method -- either Cross Validation or Holdout
    def Validation(self, method='cv', splits=10, ratio=0.3):

        # the metrics that will evaluate the model
        precision = 0
        accuracy = 0
        recall = 0
        f1 = 0
        mse = 0
        cntr = 0

        if method == 'cv':
            # ENN is initialised in fit
            if self.choice == "ENN":
                self.clf = ENN(self.X, labels=len(set(self.y)), path=self.path)
                self.clf.compile()
                self.X = self.clf.train
            else:
                self.X = self.tpp.vect_lsi.fit_transform(self.X)

            skf = StratifiedKFold(n_splits=splits)
            skf.get_n_splits(self.X, self.y)
            for train_index, test_index in skf.split(self.X, self.y):
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]

                self.clf.fit(X_train, y_train)
                predictions = self.clf.predict(X_test)
                if self.choice != "RF":
                    predictions = [ np.argmax(p) for p in predictions]

                # Evaluate Classification
                precision += metrics.precision_score(y_test, predictions, average='micro')
                recall += metrics.recall_score(y_test, predictions, average='micro')
                f1 += metrics.f1_score(y_test, predictions, average='micro')
                accuracy += metrics.accuracy_score(y_test, predictions)
                mse += metrics.mean_squared_error(y_test, predictions)

                #Confusion Matrix
                y_actu = pd.Series(y_test)
                y_pred = pd.Series(predictions)
                df_confusion = pd.crosstab(y_actu, y_pred,rownames=['Actual'], colnames=['Predicted'])
                print("--------\t",cntr,"\t--------")
                print(df_confusion)
                print("\n")
                cntr += 1

            precision = precision / splits
            recall = recall / splits
            f1 = f1 / splits
            accuracy = accuracy / splits
            mse = mse / splits

            # print the overall results  of validation
            print("\n")
            for i in set(self.y):
                print(i, "-->", self.le.inverse_transform([i]))
            print("\n\n\nPrecision:\t",precision,
                  "\nRecall:\t\t",recall,
                  "\nF1-score:\t",f1,
                  "\nAccuracy:\t",accuracy,
                  "\nMSE:\t\t",mse,"\n\n")

        elif method == 'ho':
            X_test = self.X[:int(len(self.X) * ratio)]
            X = self.X[int(len(self.X) * ratio):]
            y_test = self.y[:int(len(self.y) * ratio)]
            y = self.y[int(len(self.y)* ratio):]

            # ENN is initialised in fit
            if self.choice == "ENN":
                self.clf = ENN(X, labels=len(set(y)), path=self.path)
                self.clf.compile()
                X = self.clf.train
                test = self.clf.TextPreprocess_transform(X_test)
            else:
                X = self.tpp.vect_lsi.fit_transform(X)
                test = self.tpp.vect_lsi.transform(X_test)

            self.clf.fit(X, y)

            # the results are used to plot the ROC Curves
            if self.choice != 'RF':
                results = self.clf.predict(test)
            else:
                results = self.clf.predict_proba(test)
            # prediction is set to the index that contains the maximum probability
            predictions = [np.argmax(p) for p in results]

            precision = metrics.precision_score(y_test, predictions, average='micro')
            recall = metrics.recall_score(y_test, predictions, average='micro')
            f1 = metrics.f1_score(y_test, predictions, average='micro')
            accuracy = metrics.accuracy_score(y_test, predictions)
            mse = metrics.mean_squared_error(y_test, predictions)

            # Confusion Matrix
            y_actu = pd.Series(y_test)
            y_pred = pd.Series(predictions)
            df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'])

            # print the miss-classified tweets
            '''for i,text in enumerate(X_test):
                if predictions[i] != y_test[i]:
                    print(text, "\n", self.le.inverse_transform([predictions[i]])," // ", self.le.inverse_transform(y_test[i]) ,"\n\n----------------------------------\n"    )
            '''
            print("\n\n----------------Confusion Matirix----------------")
            print(df_confusion)

            # print the overall results  of validation
            print("\n----------Datasets----------")
            print("Training Set Size:\t", X.shape[0])
            print("Testing Set Size:\t", X_test.shape[0])
            print("\n-------Labels-------")
            for i in set(y):
                print(i, "-->", self.le.inverse_transform([i]))
            print("\n\n--------------Metrics--------------\nPrecision:\t",precision,"\nRecall:\t\t",recall,"\nF1-score:\t",f1,"\nAccuracy:\t",accuracy,
             	"\nMSE:\t\t",mse,"\n\n")

            # Compute AUC and plot the ROC Curve
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            n_classes = len(set(y))

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve([ 1 if i == val else 0 for val in y_test ], [ prediction[i] for prediction  in results ])
                roc_auc[i] = auc(fpr[i], tpr[i])

            class_to_plot = 2
            class_label = self.le.inverse_transform([class_to_plot])
            plt.figure()
            lw = 2
            plt.plot(fpr[class_to_plot], tpr[class_to_plot], color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve class: '+class_label)
            plt.legend(loc="lower right")
            plt.savefig(self.path+'/Plots/'+self.choice+'_'+str(class_label)+'_ROC.png')
            plt.show()

            # Compute the ROC curve for every class against all the others
            #( ROC curve is applied to binary problems so it calculated ROC for each class against all the others)
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            # interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

            # average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(fpr["macro"], tpr["macro"],
                     label='macro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["macro"]),
                     color='navy', linestyle=':', linewidth=4)

            colors = ['aqua', 'darkorange', 'cornflowerblue','r','g','b']
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                         ''.format(self.le.inverse_transform([i]), roc_auc[i]))

            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Some extension of Receiver operating characteristic to multi-class')
            plt.legend(loc="lower right")
            plt.savefig(self.path + '/Plots/' + self.choice + '_ROC_Curves.png')
            plt.show()


    # Serialize the Object
    def StoreObject(self):
            pickle.dump(self, open(self.path + '/EncapsulatedFiles/Classifier', 'wb'))


    def loadObject(self):
            with open(self.path + '/EncapsulatedFiles/Classifier', 'rb') as f:
                return pickle.load(f)


    def Analyze(self):
            self.tpp.Analyze(self.X)


    def Validate_Classification(self, testSet):
        X = self.X
        y = self.y
        self.tpp.test_texts = testSet['tweetText'].values
        test = self.tpp.clean(opt='test')
        y_test = testSet['Result'].values
        label_set = set(y_test)
        print("\n\nTesting Set:\t", y_test.shape[0])
        for label in label_set:
            print("\n", label, ":\t\t", y_test.tolist().count(label))

        # ENN is initialised in fit
        if self.choice == "ENN":
            self.clf = ENN(X, labels=len(set(y)), path=self.path)
            self.clf.compile()
            X = self.clf.train
            test = self.clf.TextPreprocess_transform(test)
        else:
            X = self.tpp.vect_lsi.fit_transform(X)
            test = self.tpp.vect_lsi.transform(test)

        self.clf.fit(X, y)
        if self.choice != 'RF':
            results = self.clf.predict(test)
        else:
            results = self.clf.predict_proba(test)
        predictions = [np.argmax(p) for p in results]

        y_test = self.le.transform(y_test)
        precision = metrics.precision_score(y_test, predictions, average='micro')
        recall = metrics.recall_score(y_test, predictions, average='micro')
        f1 = metrics.f1_score(y_test, predictions, average='micro')
        accuracy = metrics.accuracy_score(y_test, predictions)
        mse = metrics.mean_squared_error(y_test, predictions)

        # Confusion Matrix
        y_actu = pd.Series(y_test)
        y_pred = pd.Series(predictions)
        df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'])
        print("\n\n----------------Confusion Matirix----------------")
        print(df_confusion)
        # print the overall results of the validation
        print("\n-------Labels-------")
        for i in set(y):
            print(i, "-->", self.le.inverse_transform([i]))
        print("\n\n--------------Metrics--------------\nPrecision:\t",precision,"\nRecall:\t\t",recall,"\nF1-score:\t",f1,"\nAccuracy:\t",accuracy,
            "\nMSE:\t\t",mse,"\n\n")


        # Compute AUC and plot the ROC Curve
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = len(set(y))

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve([ 1 if i == val else 0 for val in y_test ], [ prediction[i] for prediction  in results ])
            roc_auc[i] = auc(fpr[i], tpr[i])

        class_to_plot = 2
        class_label = self.le.inverse_transform([class_to_plot])
        plt.figure()
        lw = 2
        plt.plot(fpr[class_to_plot], tpr[class_to_plot], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve class: '+class_label)
        plt.legend(loc="lower right")
        plt.savefig('results/Test/T_'+self.choice+'_'+class_label+'_ROC.png')
        plt.show()

        # Compute the ROC curve for every class against all the others
        # ( ROC curve is applied to binary problems so it calculated ROC for each class against all the others)
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = ['aqua', 'darkorange', 'cornflowerblue','r','g','b']
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'.format(self.le.inverse_transform([i]),
                     roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.savefig('results/Test/T_'+self.choice+'_ROC_Curves.png')
        plt.show()



    # plot the ROC Curve of every classifier for a specified class (in the same figure)
    def roc_plot(self, test_set):
        # list of classifiers
        classifiers = ['RF', 'DNN', 'RF']

        # sets initializations
        X = self.X
        y = self.y
        self.tpp.test_texts = test_set['tweetText'].values
        test = self.tpp.clean(opt='test')
        y_test = test_set['Result'].values
        label_set = set(y_test)
        print("\n\nTesting Set:\t", y_test.shape[0])
        for label in label_set:
            print("\n", label, ":\t\t", y_test.tolist().count(label))

        # variables for the plot
        plt.figure()
        class_to_plot = 2  # <-- this var must contain the number that corresponds to the class that you want to plot.
        class_label = self.le.inverse_transform([class_to_plot])
        n_classes = len(set(y))
        lw = 2

        # The colors of the curves
        colors = ['aqua', 'darkorange', 'cornflowerblue']

        # iterate over the classifiers, implement the classification (predictions)
        # and calculate the ROC curve of each clf
        for clf_iter in classifiers:

            # Training
            if clf_iter == "ENN":
                self.clf = ENN(X, labels=len(set(y)), path=self.path)
                self.clf.compile()
                X = self.clf.train
                clean_test = self.clf.TextPreprocess_transform(test)
            else:
                X = self.tpp.vect_lsi.fit_transform(X)
                clean_test = self.tpp.vect_lsi.transform(test)
            self.clf.fit(X, y)

            # Predictions
            if clf_iter != 'RF':
                results = self.clf.predict(clean_test)
            else:
                results = self.clf.predict_proba(clean_test)

            # Calculate the ROC Curve
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve([1 if i == val else 0 for val in y_test],
                                              [prediction[i] for prediction in results])
                roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[class_to_plot], tpr[class_to_plot], color=colors[classifiers.index(clf_iter)],
                     lw=lw, label=clf_iter + ' ROC curve (area = %0.2f)' % roc_auc[2])

        # plot the ROC Curve in one figure
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve class: ' + class_label)
        plt.legend(loc="lower right")
        plt.savefig('results/Test/T_' + self.choice + '_' + class_label + '_ROC.png')
        plt.show()

