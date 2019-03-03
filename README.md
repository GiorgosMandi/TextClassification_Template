<h2>Text Classification Template</h2>

This project is a template for Text Classification. It provides NLP techniques for text pre-processing, 
several Classifiers, evaluation methods and many more functionalities.
<br/>
In more detail

1. In text pre-process offers the following functionalities <br/>
    * Removal of the Stop Words.
    * Expansion of the Contractions.
    * Stemming or lemmatization.
    * Vectorization of the input text using the Tf-Idf vectorizer.
    * Dimension Reduction using the LSI method.
    * It is also capable of oversampling a class using the SMOTE technique.
    
2. The Provided Classifiers are:
    * Random Forest Classifier
    * A simple Neural Network that consists of several Dense Layers.
    * An Embedding Neural Network which its first layer is an Embedding Layer. An embedding is a mapping from discrete objects, such as words, to vectors of real numbers.
    <br/>
    (The Serialization of the Classifiers is also provided.)

3. As Evaluation methods it offers:
    * The Cross Validation method.
    * The Hold Out method.
    * A Custom method which perform classification to an already correct classified dataset in order to compare its results.
    * ROC Curves Analysis.
    * Analysis on the components produced by the LSI method. This method reveals which components have greater impact to the result of LSI.
     <br/>
    (The three first techniques also calculate the confusion matrix and other metrics such as Accuracy, Recall, etc.)
  <br/>  

**The Libraries required for the execution are**:  _pandas_, _numpy_, _sklearn_, _tensorflow(keras)_, _nltk_, _imblearn_, _pickle and matplotlib._

**Execution:** python main-classification --clf <*CLF*> --method <*the selected method*> --train <*path to the training set*> --test <*path to the training set(OPTIONAL)*>
 
