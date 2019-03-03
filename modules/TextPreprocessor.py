import numpy as np
import pandas as pd
import re

import matplotlib as plt
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import Normalizer


class TextPreProcessor:
    clean_train = []
    clean_test = []
    components = 100
    STOPWORDs = ENGLISH_STOP_WORDS.union(['i', '-', 'said', 'did', 'say', 'says', 'year', 'day'
            ,'going', 'having', 'like', 'wasn','given', 'got', 'the', 'didnt','didn','t', 'we'
            ,'km', 'wa', 'don','wasn','thi','tha', 'indonesia','indonesian','alaska','bali', 'japan','hokkaido'
            ,'fiji', 'lombok', 'jakarta','gempa', 'kodiak', 'ca', 'california', 'earthquak', 'earthquake'])

    dict_contractions = {'werent': 'were not', 'weren\'t': 'were not', 'weren\' t':'were not'
            ,'wasnt': 'was not', 'wasn\'t': 'was not', 'wasn\' t':'was not'
            ,'didnt': 'did not', 'didn\'t': 'did not', 'didn\' t':'did not'
            ,'cant':'can not', 'can\'t':'can not', 'cant\' t': 'can not'
            ,'couldnt':'could not', 'couldn\'t':'could not', 'couldnt\' t': 'could not'
            ,'havent':'have not', 'haven\'t':'have not', 'havent\' t': 'have not'
            ,'hasnt':'has not', 'hasn\'t':'has not', 'hasnt\' t': 'has not'
            ,'hadnt':'had not', 'hadn\'t':'had not', 'hadnt\' t': 'had not'
            }

    def __init__(self, train_texts="", test_texts=[], opt = 'train', components = 100):
        self.train_texts  = train_texts
        self.test_texts  = test_texts
        self.components = components
        self.vectorizer = TfidfVectorizer(stop_words=self.STOPWORDs, min_df=2, use_idf=True)
        self.lmtzr = WordNetLemmatizer()
        self.ps = PorterStemmer()
        self.lsi_model = TruncatedSVD(n_components=components)
        self.normalizer = Normalizer(copy=False)
        self.vect_lsi = Pipeline([('vectorize', self.vectorizer) , ('lsi', self.lsi_model), ('normalizer', self.normalizer)])
        self.sm = SMOTE(kind='regular')

    # pre-process data -- clean text from capitals and special characters
    # and use lemitizer or stemmer -- if the opt is train stores the results
    # in clean_train else in clean_test
    def clean(self, opt = 'train'):
        if opt == 'train':
            texts  = self.train_texts
        else:
            texts  = self.test_texts
        clean_texts = []
        for i,text in enumerate(texts):
            new_text = []
            try:
                for word in text.split(" "):
                    if not 'http' in word and not '@' in word and not word in self.STOPWORDs :
                        new_word = re.sub('[^a-z]|\r|\n',' ', word.lower())
                        new_word = re.sub(' +',' ',new_word)
                        if len(new_word) - new_word.count(' ') < 2: continue
                        if ' ' in new_word:
                            for w in new_word.split(' '):
                                w = self.ps.stem(w)
                                (new_text.append(w) if w != "" and not w in self.STOPWORDs and len(w) > 1   else None)
                        else:
                            new_word = self.ps.stem(new_word)
                            (new_text.append(new_word) if new_word != "" and not new_word in self.STOPWORDs and len(new_word) > 1 else None)
            except AttributeError:
                print("Error: Wrong formation of csv close to line: ",i, opt)
                exit()
            new_text=' '.join(new_text)
            new_text = re.sub(' +',' ',new_text)
            clean_texts.append(new_text)
        clean_texts =  pd.Series(clean_texts)
        if opt == 'train':
            self.clean_train = clean_texts
        else:
            self.clean_test  = clean_texts
        return clean_texts


    # get a string and return a list containing the produced clean text
    def clean_text(self, text, wanted_sw=[], sw_flag=False):
        STOPWORDs = {}
        if not sw_flag:
            STOPWORDs = set(self.STOPWORDs)
            for sw in wanted_sw:
                STOPWORDs.remove(sw)
        clean_text = []
        for word in text.split(" "):
            if not 'http' in word and not '@' in word and not word in STOPWORDs :
                new_word = re.sub('[^a-z]|\r|\n',' ', word.lower())
                new_word = re.sub(' +',' ',new_word)
                if ' ' in new_word:
                    for w in new_word.split(' '):
                        w = self.ps.stem(w)
                        (clean_text.append(w) if w != "" and not w in STOPWORDs and len(w) > 1   else None)
                else:
                    new_word = self.ps.stem(new_word)
                    (clean_text.append(new_word) if new_word != "" and not new_word in STOPWORDs and len(new_word) > 1 else None)
        return clean_text


    # expand the contractions that are included in dict_contractions
    def Contractions_Expand(self, text):
        text = re.sub('’','\'', text.lower())
        for contraction in self.dict_contractions.keys():
            if contraction in text:
                text = re.sub(contraction, self.dict_contractions[contraction], text)
        return text


    # Oversampling -- Generalize minority Class and undersample majority
    def oversample(self,X, y):
        print("Before Oversampling:\t", X.shape[0])
        X, y = self.sm.fit_sample(X, y)
        print("After Oversampling:\t",X.shape[0])
        return X, y


    # plot the words that are affecting most the
    # components after dimension reduction
    def Analyze(self, X):

        print("\nPerforming dimensionality reduction using LSA")
        # Project the tfidf vectors onto the first N principal components.
        # Though this is significantly fewer features than the original tfidf vector,
        # they are stronger features, and the accuracy is higher..
        feat_names = self.vectorizer.get_feature_names()

        # The SVD matrix will have one row per component, and one column per feature
        # of the original data.
        # for compNum in range(0, 100, 10):
        for compNum in range(0, self.components):

            comp = self.lsi_model.components_[compNum]
            # Sort the weights in the first component, and get the indeces
            indeces = np.argsort(comp).tolist()

            # Reverse the indeces, so we have the largest weights first.
            indeces.reverse()

            # Grab the top 10 terms which have the highest weight in this component.
            terms = [feat_names[weightIndex] for weightIndex in indeces[0:15]]
            weights = [comp[weightIndex] for weightIndex in indeces[0:15]]

            # Display these terms and their weights as a horizontal bar graph.
            # The horizontal bar graph displays the first item on the bottom; reverse
            # the order of the terms so the biggest one is on top.
            terms.reverse()
            weights.reverse()
            positions = np.arange(15) + .5    # the bar centers on the y axis

            plt.figure(compNum)
            plt.barh(positions, weights, align='center')
            plt.yticks(positions, terms)
            plt.xlabel('Weight')
            plt.title('Strongest terms for component %d' % (compNum))
            plt.grid(True)
            plt.show()
