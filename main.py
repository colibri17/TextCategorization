# Main file which is executed by the user

# It allows to match the names
import fnmatch

# It allows to interact with the operative system
import os

# It carries out the preprocessing operations
import nltk 

# Module to retrieve the elements
# from the SGML files
import SGMLReutersParser

# Module to preprocess the text of the documents
import TextPreProcessing

# Module to compute the selecting phase
import Select

# Module to set up the topic models
import LdaTopicModels

# Itertools
import itertools

# cPickle, to store elements in a file
import cPickle

# NLTK frequency distribution
from nltk import FreqDist

# Scikit learn
from sklearn.feature_extraction import DictVectorizer

# Naive Bayes
from sklearn.naive_bayes import MultinomialNB, GaussianNB
# Random Forest
from sklearn.ensemble import RandomForestClassifier
# SVM
from sklearn.svm import LinearSVC
# Perceptron
from sklearn.linear_model import Perceptron
# K-means
from sklearn.cluster import KMeans
# Hierarchical
from sklearn.cluster import Ward
# DBSCAN
from sklearn.cluster import DBSCAN

# Metrics evaluation
from sklearn import metrics

# To create store the confusion matrix plots
from confusion_matrix import ConfMatrix

# Sequence of variables that allows to completly rule and control the program
data_path = 'reuters21578'

# When there is more than one topic within the topic list, this variable chooses 
# if keeping the most/less recurrent
keep_first = True

# Min frequency the word has to have in the training set not to be deleted. Not effective 
# in the definitve version of the program
min_frequency = 5

# first words to keep in the training set. Effective in the definite version of the program
first_words = 150

# I specify the class
class_attribute = 'topics'

# I decide what type of text representation I want
text_bag = 'binary_weight'

# I decide if I want the topic model
topic_model = True

# Topics on which perform the classification
selected_topics = ['earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn']

# Create the selected_keys, namely the keys I want to save from the dictionary I
# initially create with the SGML parsing
selected_keys = ['text', 'split', class_attribute]

# Define the number of topics which has to be generated
number_topics = 9





# Start the reading phase

# List of the root folder, the subdirectories and the files contaised
# is a top-down order
for root, _dirnames, filenames in os.walk(data_path):
    
    # Documents will be formed by dictionaries where the keys are tag/attributes
    # of the document and the values are the corresponding values 
    documents = []
        
    print 'Start analysing the SGML files..'
        
    # Retrieve the files which end with sgm
    for filename in fnmatch.filter(filenames, '*.sgm'):

        # print 'Filename', filename
        # Creatisg the path of the file which has to be opened and parsed
        file_path = os.path.join(root, filename)
        print file_path
        
        # Creatisg the parser
        parser = SGMLReutersParser.SGMLReutersParser()
        
        # Document dictionary: each element refer to a single article
        documents += parser.parse(file_path)
        
        # break
        
    print 'Finished to read the SGML files!'


# I filter on the number of documents and the 
# parameters of each text.
# In this way the number of the documents in the database and the number of elements per test is decreased.

# I choose the selected keys within the documents.
# Therefore I remove some of the keys that will not be utilized but that have been
# however initially stored (to increment the portability of the program)
documents = [dict([(k, doc[k]) for k in selected_keys]) for doc in documents]

# Choose the instances having at least one element of the selected topics
documents = [doc for doc in documents if not set(doc['topics']).isdisjoint(selected_topics)]

# Save only the documents which are test or training instances
documents = [doc for doc in documents if doc['split'] == 'TRAIN' or doc['split'] == 'TEST']


# I choose either the most common topic or the less common one when I have a document
# having more than one topic.
# So I derive the distribution of the topics by utilizing the nltk library
all_topics_words = list(itertools.chain.from_iterable([doc[class_attribute] for doc in documents]))
topics_frequency_dist = FreqDist(all_topics_words)
print 'frequencies topics:', topics_frequency_dist
# Replace the topics that have more than one values with
# the most common in the list or the less common according to the variable keep_first
for el in documents:
    # print 'before',  el[class_attribute], 
    if len(el[class_attribute]) > 1:
        # print el
        # Compute the frequencies related to the elements in the list and the related names
        frequencies = [(topics_frequency_dist[x], x) for x in el[class_attribute] if x in selected_topics]
        # Now sort this list from the biggest value to the smallest one
        frequencies.sort()
        frequencies.reverse()
        # Now retrieve the first element. If two elements are equal
        if keep_first:
            el[class_attribute] = frequencies[0][1]
        else:
            el[class_attribute] = frequencies[-1][1]
        # print 'maintained', el[class_attribute],
    else:
        # Remove the list. No more necessary
        assert len(el[class_attribute]) == 1
        el[class_attribute] = el[class_attribute][0]
    # print 'after', el[class_attribute]
    # print el






# Starting the preprocessing phase over the text

# It takes track of the current document number
i = 0

# Because the following operations are very computational expensive,
# but always the same, I can check if they are already performed. If this is the
# case then the file containing them is already created
if not os.path.isfile('picked_documents_%s.p' % keep_first):
    
    for doc in documents:
        
        i += 1
        
        # Initializing the textpreproccesor
        text_preproc = TextPreProcessing.TextPreProcessing(doc['text'])
    
        # Transforming some english abbreviation in complete text
        text_preproc.spell_check()
        
        # Divide into words using regex
        text_preproc.tokenize()
        
        # The POS_tagging has to be performed in this position.
        text_preproc.POS_tagging()
        
        # Retrieving the entities names
        # text_preproc.entity_names()
        
        # Lowering all the characters
        text_preproc.lowercase()
        
        # Performing the lemmatization process
        text_preproc.lemmatize()
        
        # Removing the punctuation
        text_preproc.remove_punctuation()
        
        # Removing the numbers
        text_preproc.remove_numbers()
        
        # I remove the stop words. This process is done here
        # because I cannot remove words as 'to' before doing the POS-tagging since it might
        # use them to proper recognize the POS.
        text_preproc.remove_stopwords()
        
        # Returning a list from the POS tag
        text_preproc.return_list()
        
        # I update the corresponding value of the element in the dictionary
        doc['text'] = text_preproc.updated_element
        
        
        print 'Document number', i, 'processed'
    
        # print 'Document', doc
        
    # Construct the file
    cPickle.dump(documents, open('picked_documents_%s.p' % keep_first, 'wb'))

# I the previous operations were already performed and therefore the pickled 
# exist
else:
    # Load the files
    print 'All the documented already processed. Retrieve the list of dictionaries..'
    documents = cPickle.load(open('picked_documents_%s.p' % keep_first, 'rb'))
    print 'Done.'
    





# Beginning the feature creation phase

# From a list to list I pass to a single list containing all the words in the training
# set. This is very important: the list of words must be created starting from the training set
training_texts = [doc['text'] for doc in documents if doc['split'] == 'TRAIN']
all_words = list(itertools.chain.from_iterable(training_texts))

# Initialize the object that allows to perform the selection
sel = Select.Select(all_words)

# Creates the frequency distribution related to all the words
# A frequency distribution contains the count of the words
# Not necessary.
# sel.frequency_distribution_all()

# Removes all words having lowest frequencies in all the set of words. They are considered as noise.
# It is important to update the all words inside the called routine, because then 
# they'll be utilized to create the bag of words
# all_words = sel.select_upto_n_frequent_words_all(min_frequency)

# Select only the first first_words from the distribution
all_words = sel.select_n_frequent_words_all(first_words)

print 'Selected words to keep', sel.selected_words
    
# I erase from each text the words that are not in the modified 
# list of all words (I erase the less frequent words from each of the texts).
# This cycle is not fundamental, because the words that are not within the selected 
# ones are not considered because the measures are computed cycling over the selected 
# overall words. However, it is left so that some further selections can be performed if wanted
for doc in documents:
    # print 'before', doc['text']
    # doc['text'] = sel.select_words_in_doc(doc['text'])
    # Other selection within the single files 
    # doc['text'] = sel.select_n_frequent_words_doc(first_words, doc['text'])
    # doc['text'] = sel.select_upto_n_frequent_words_doc(min_frequency, doc['text']) 
    # print 'Formatted selected text', doc['text']
    # print 'after', doc['text']
    pass



all_texts = [doc['text'] for doc in documents if doc['split'] == 'TRAIN']

print
print 'Begin the feature selection..'

# Beginning the topic models phase if selected
if topic_model:
    print 'Begin topic model phase..'
    
    # Create the LdaTopicModels object
    lda = LdaTopicModels.LdaTopicModels()
    # Create the basic dictionary from all the retrieved texts
    lda.create_dictionary(all_texts)
    
    # Create the corpus from all the texts
    corpus = lda.create_vector_corpus(all_texts)
    
    # Create the model providing the number of topics desired
    model = lda.create_model(corpus, num_topics = number_topics)
    
    #
    for doc in documents:
        # Evaluate the model over the given corpus. In our situation,
        # the corpus is the same as the training set used to generate the model
        corpus_lda = lda.evaluate_model(lda.convert_into_vector(doc['text']))
        
        # I sort the corpus_lda, containing the topic with the corresponding probability
        # according to the second value so I can retrieve the max
        corpus_lda.sort(key=lambda x: x[1])
        # Store the found topic model. I store it as a string
        # so I know it is a categorical value
        # print corpus_lda
        doc['topic_model'] = str(corpus_lda[-1][0])
        
        # print 'evaluation', [x for x in corpus_lda], 'selected', doc['topic_model']
        

if text_bag == 'none':
    print 'Delete all the texts without doing anything'
    # Do not do anything except that deleting the text
    for doc in documents:
        # Delete the text
        del doc['text']
        
elif text_bag == 'binary_weight':
    print 'Begin binary weight.'
    # I construct the binary weighting bag of word. To do that, I utilize the 
    # nltk library. The output is a dictionary that will have to be jointed with
    # the original documents
    for doc in documents:
        # Erasing the repetition of the words in the document
        words = set(doc['text'])       
        # Dictionary which will contain the attributes
        features = {}    
        # For each word in the word_features created
        # I determine if the document contain the word or not
        for word in all_words:
            features['contains(%s)' % word] = (word in words)   
            
        # print features
        # Update the dictionary
        doc.update(features)
        # Delete the text. Nomore necessary
        del doc['text']
    
    
elif text_bag == 'TF':
    print 'begin term frequency TF.'
    for doc in documents:
        # Dictionary which will contain the attributes
        features = {}       
        
        # I create the frequency distribution of the text.
        # For the words that are not in the document the frequency which is returned is 0
        text_features = sel.frequency_distribution_doc(doc['text'])
        
        for word in all_words:
            features['frequency(%s)' % word] = text_features[word]
            
        # Update the dictionary
        doc.update(features)
        
        # Delete the text. Nomore necessary
        del doc['text']
        
        
elif text_bag == 'tf':
    print 'begin weighted term frequency tf.'
    for doc in documents:
        # Dictionary which will contain the attributes
        features = {}       
        
        # I create the frequency distribution of the text.
        # For the words that are not in the document the frequency which is returned is 0
        text_features = sel.frequency_distribution_doc(doc['text'])
        
        
        for word in all_words:
            # Now I have to check the maximum frequency of the word
            # in any document of the training set. I use the function count
            max_frequency = max([x.count(word) for x in all_texts])
            print max_frequency
            features['frequency(%s)' % word] = text_features[word] / float(max_frequency)
            
        # Update the dictionary
        doc.update(features)
        
        # Delete the text. Nomore necessary
        del doc['text']
      
elif text_bag == 'tfidf':
    print 'Begin tfidf.'
    # It will contain the features
    features = {}
    
    # Generate all the texts as list of lists
    all_texts = [doc['text'] for doc in documents if doc['split'] == 'TRAIN']
    # print 'All the selected and formatted texts', all_texts
    
    # Create the LdaTopicModels object
    tfidf = LdaTopicModels.LdaTopicModels()
    # Create the basic dictionary from all the retrieved texts
    tfidf.create_dictionary(all_texts)
    # Original dictionary
    dictionary_all_words = tfidf.token2id_dictionary()
    # print dictionary_all_words
    # Create the reverse dictionary from id to token. Useful to the check phase
    # reversed = dict((v,k) for k,v in tfidf.token2id_dictionary().iteritems())
    # print 'reversed', reversed
    
    # Create the corpus from all the texts
    corpus = tfidf.create_vector_corpus(all_texts)
    
    # Create the tfidf model
    tfidf_model = tfidf.create_tfidf(corpus)
    
    for doc in documents:
        
        # Sort the text so to retrieve the element
        tf_idf_doc = tfidf.evaluate_model(tfidf.convert_into_vector(doc['text']))
        # print tf_idf_doc
        # Reconvert in the original dictionary format
        for word in set(all_words):
            # If the id associated to the text word is in the ids of the tf_idf_doc
            # the value of the tfidf must not be null
            if dictionary_all_words[word] in [id for (id, tfidf_value) in tf_idf_doc]:
                features['tfidf(%s)' % word] = [tfidf_value for (id, tfidf_value) in tf_idf_doc if dictionary_all_words[word] == id][0]
            else:
                features['tfidf(%s)' % word] = 0
        # print 'dictionary features', features
        
        # Now that I have the values I can update the dictionary and delete the text
        doc.update(features)
        # Delete the text. Nomore necessary
        del doc['text']


else:
    print text_bag

print 'Done!'
print 
print






# Beginning classification part
# Create the object to perform the classification. It needs the documents
# and the name of the class

print 'Begin the classification..'

# cl = Classification.Classifier(documents, class_attribute)

# Divide training set from test set and return for each of them
# the target values
training_doc = [x for x in documents if x['split'] == 'TRAIN']
test_doc = [x for x in documents if x['split'] == 'TEST']

# Construct the class vectors
target_training = [x[class_attribute] for x in training_doc]
target_test = [x[class_attribute] for x in test_doc]

# Now remove the unuseful attributes 'split' and 'topics'
# from the training and the test
for el in training_doc + test_doc:
    del el['split']
    del el[class_attribute]
    
    
# Now translate the elements in a proper numpy matrix
# This it is done by utilizing the sklearn.feature_extraction.DictVectorizer function.
# For the continuous values nothing it is touch, for the discrete ones different columns are utilized.
# For a more exaustive explanation, please see http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html
vec = DictVectorizer(sparse=False)
training = vec.fit_transform(training_doc)
test = vec.fit_transform(test_doc)
print 'Numpy arrays created..'
print


# Naive Bayes
# Create the model and train it over the training instances
if text_bag == 'binary_weight' or text_bag == 'none':
    nb = MultinomialNB()
else:
    nb = GaussianNB()
nb.fit(training, target_training)
print 'Naive bayes created..'
# Test over the training data
training_predictions_nb = nb.predict(training)
# Test over the test data
test_predictions_nb = nb.predict(test)

# Random Forest
# Create the model and train it over the training instances
rf = RandomForestClassifier()
rf.fit(training, target_training)
print 'Random forest created..'
# Test over the training data
training_predictions_rf = rf.predict(training)
# Test over the test data
test_predictions_rf = rf.predict(test)


# SVM
# Create the model and train it over the training instances
sv = LinearSVC(C = 0.5)
sv.fit(training, target_training)
print 'Support vector machine created..'
# Test over the training data
training_predictions_sv = sv.predict(training)
# Test over the test data
test_predictions_sv = sv.predict(test)


# Perceptron
# Create the model and train it over the training instances
pc = Perceptron()
pc.fit(training, target_training)
print 'Support vector machine created..'
# Test over the training data
training_predictions_pc = pc.predict(training)
# Test over the test data
test_predictions_pc = pc.predict(test)

# Compute the errors of the three methods on the training
print 'accuracy Naive Bayes training:', metrics.accuracy_score(target_training, training_predictions_nb)
print 'precision with Naive Bayes training', metrics.precision_score(target_training, training_predictions_nb, average = None)
print 'recall with Naive Bayes training', metrics.recall_score(target_training, training_predictions_nb, average = None)
print 'micro precision with Naive Bayes training', metrics.precision_score(target_training, training_predictions_nb, average = 'micro')
print 'macro precision with Naive Bayes training', metrics.precision_score(target_training, training_predictions_nb, average = 'macro')
print 'micro recall with Naive Bayes training', metrics.recall_score(target_training, training_predictions_nb, average = 'micro')
print 'macro recall with Naive Bayes training', metrics.recall_score(target_training, training_predictions_nb, average = 'macro')
print
print 'accuracy Random Forest training:', metrics.accuracy_score(target_training, training_predictions_rf)
print 'precision with Random Forest training', metrics.precision_score(target_training, training_predictions_rf, average = None)
print 'recall with Random Forest training', metrics.recall_score(target_training, training_predictions_rf, average = None)
print 'micro precision with Random Forest training', metrics.precision_score(target_training, training_predictions_rf, average = 'micro')
print 'macro precision with Random Forest training', metrics.precision_score(target_training, training_predictions_rf, average = 'macro')
print 'micro recall with Random Forest training', metrics.recall_score(target_training, training_predictions_rf, average = 'micro')
print 'macro recall with Random Forest training', metrics.recall_score(target_training, training_predictions_rf, average = 'macro')
print
print 'accuracy SVM training:', metrics.accuracy_score(target_training, training_predictions_sv)
print 'precision with SVM training', metrics.precision_score(target_training, training_predictions_sv, average = None)
print 'recall with SVM training', metrics.recall_score(target_training, training_predictions_sv, average = None)
print 'micro precision with SVM training', metrics.precision_score(target_training, training_predictions_sv, average = 'micro')
print 'macro precision with SVM training', metrics.precision_score(target_training, training_predictions_sv, average = 'macro')
print 'micro recall with SVM training', metrics.recall_score(target_training, training_predictions_sv, average = 'micro')
print 'macro recall with SVM training', metrics.recall_score(target_training, training_predictions_sv, average = 'macro')

# Create a very good graph representing the confusion matrix
labels = list(set(target_training))
labels.sort()

cm = ConfMatrix(metrics.confusion_matrix(target_training, training_predictions_sv), labels)
cm.save_matrix('confusion_matrix_SVM.p')
cm.get_classification()
cm.gen_conf_matrix('confusion matrix SVM training')
cm.gen_highlights('conf_matrix_highlights')

cm = ConfMatrix(metrics.confusion_matrix(target_training, training_predictions_rf), labels)
cm.save_matrix('confusion_matrix_RandomForest.p')
cm.get_classification()
cm.gen_conf_matrix('confusion matrix Random Forest training')
cm.gen_highlights('conf_matrix_highlights')

print
print 'accuracy Perceptron training:', metrics.accuracy_score(target_training, training_predictions_pc)
print 'precision with Perceptron training', metrics.precision_score(target_training, training_predictions_pc, average = None)
print 'recall with Perceptron training', metrics.recall_score(target_training, training_predictions_pc, average = None)
print 'micro precision with Perceptron training', metrics.precision_score(target_training, training_predictions_pc, average = 'micro')
print 'macro precision with Perceptron training', metrics.precision_score(target_training, training_predictions_pc, average = 'macro')
print 'micro recall with Perceptron training', metrics.recall_score(target_training, training_predictions_pc, average = 'micro')
print 'macro recall with Perceptron training', metrics.recall_score(target_training, training_predictions_pc, average = 'macro')
print
print

# Write the measure on a file
f = open('training_measures_%s_%s_%s_%s.txt' % (first_words, text_bag, topic_model, keep_first), 'w')
f.write('accuracy Naive Bayes training: %s\n' % metrics.accuracy_score(target_training, training_predictions_nb))
# f.write('precision with Naive Bayes training %s\n' % metrics.precision_score(target_training, training_predictions_nb, average = None))
# f.write('recall with Naive Bayes training %s\n' % metrics.recall_score(target_training, training_predictions_nb, average = None))
f.write('micro precision with Naive Bayes training %s\n' % metrics.precision_score(target_training, training_predictions_nb, average = 'micro'))
f.write('macro precision with Naive Bayes training %s\n' % metrics.precision_score(target_training, training_predictions_nb, average = 'macro'))
f.write('micro recall with Naive Bayes training %s\n' % metrics.recall_score(target_training, training_predictions_nb, average = 'micro'))
f.write('macro recall with Naive Bayes training %s\n\n' % metrics.recall_score(target_training, training_predictions_nb, average = 'macro'))

f.write('accuracy Random Forest training: %s\n' % metrics.accuracy_score(target_training, training_predictions_rf))
# f.write('precision with Random Forest training %s\n' % metrics.precision_score(target_training, training_predictions_rf, average = None))
# f.write('recall with Random Forest training %s\n' % metrics.recall_score(target_training, training_predictions_rf, average = None))
f.write('micro precision with Random Forest training %s\n' % metrics.precision_score(target_training, training_predictions_rf, average = 'micro'))
f.write('macro precision with Random Forest training %s\n' % metrics.precision_score(target_training, training_predictions_rf, average = 'macro'))
f.write('micro recall with Random Forest training %s\n' % metrics.recall_score(target_training, training_predictions_rf, average = 'micro'))
f.write('macro recall with Random Forest training %s\n\n' % metrics.recall_score(target_training, training_predictions_rf, average = 'macro'))

f.write('accuracy SVM training %s\n' % metrics.accuracy_score(target_training, training_predictions_sv))
# f.write('precision with SVM training %s\n' % metrics.precision_score(target_training, training_predictions_sv, average = None))
# f.write('recall with SVM training %s\n' % metrics.recall_score(target_training, training_predictions_sv, average = None))
f.write('micro precision with SVM training %s\n' % metrics.precision_score(target_training, training_predictions_sv, average = 'micro'))
f.write('macro precision with SVM training %s\n' % metrics.precision_score(target_training, training_predictions_sv, average = 'macro'))
f.write('micro recall with SVM training %s\n' % metrics.recall_score(target_training, training_predictions_sv, average = 'micro'))
f.write('macro recall with SVM training %s\n\n' % metrics.recall_score(target_training, training_predictions_sv, average = 'macro'))

f.write('accuracy Perceptron training: %s\n' % metrics.accuracy_score(target_training, training_predictions_pc))
# f.write('precision with Perceptron training %s\n' % metrics.precision_score(target_training, training_predictions_pc, average = None))
# f.write('recall with Perceptron training %s\n' % metrics.recall_score(target_training, training_predictions_pc, average = None))
f.write('micro precision with Perceptron training %s\n' % metrics.precision_score(target_training, training_predictions_pc, average = 'micro'))
f.write('macro precision with Perceptron training %s\n' % metrics.precision_score(target_training, training_predictions_pc, average = 'macro'))
f.write('micro recall with Perceptron training %s\n' % metrics.recall_score(target_training, training_predictions_pc, average = 'micro'))
f.write('macro recall with Perceptron training %s\n' % metrics.recall_score(target_training, training_predictions_pc, average = 'macro'))

f.close()

# Compute the errors of the three methods on the tests
print 'accuracy with Naive Bayes test', metrics.accuracy_score(target_test, test_predictions_nb)
print 'precision with Naive Bayes test', metrics.precision_score(target_test, test_predictions_nb, average = None)
print 'recall with Naive Bayes test', metrics.recall_score(target_test, test_predictions_nb, average = None)
print 'micro precision with Naive Bayes test', metrics.precision_score(target_test, test_predictions_nb, average = 'micro')
print 'macro precision with Naive Bayes test', metrics.precision_score(target_test, test_predictions_nb, average = 'macro')
print 'micro recall with Naive Bayes test', metrics.recall_score(target_test, test_predictions_nb, average = 'micro')
print 'macro recall with Naive Bayes test', metrics.recall_score(target_test, test_predictions_nb, average = 'macro')
print
print 'accuracy with Random Forest test', metrics.accuracy_score(target_test, test_predictions_rf)
print 'precision with Random Forest test', metrics.precision_score(target_test, test_predictions_rf, average = None)
print 'recall with Random Forest test', metrics.recall_score(target_test, test_predictions_rf, average = None)
print 'micro precision with Random Forest test', metrics.precision_score(target_test, test_predictions_rf, average = 'micro')
print 'macro precision with Random Forest test', metrics.precision_score(target_test, test_predictions_rf, average = 'macro')
print 'micro recall with Random Forest test', metrics.recall_score(target_test, test_predictions_rf, average = 'micro')
print 'macro recall with Random Forest test', metrics.recall_score(target_test, test_predictions_rf, average = 'macro')
print
print 'accuracy with SVM test', metrics.accuracy_score(target_test, test_predictions_sv)
print 'precision with SVM test', metrics.precision_score(target_test, test_predictions_sv, average = None)
print 'recall with SVM test', metrics.recall_score(target_test, test_predictions_sv, average = None)
print 'micro precision with SVM test', metrics.precision_score(target_test, test_predictions_sv, average = 'micro')
print 'macro precision with SVM test', metrics.precision_score(target_test, test_predictions_sv, average = 'macro')
print 'micro recall with SVM test', metrics.recall_score(target_test, test_predictions_sv, average = 'micro')
print 'macro recall with SVM test', metrics.recall_score(target_test, test_predictions_sv, average = 'macro')

# Create a very good graph representing the confusion matrix for the random forest and the SVM
labels = list(set(target_test))
labels.sort()

cm = ConfMatrix(metrics.confusion_matrix(target_test, test_predictions_sv), labels)
cm.save_matrix('confusion_matrix_SVM_test.p')
cm.get_classification()
cm.gen_conf_matrix('confusion matrix SVM test')
cm.gen_highlights('conf_matrix_highlights')

cm = ConfMatrix(metrics.confusion_matrix(target_test, test_predictions_rf), labels)
cm.save_matrix('confusion_matrix_RandomForest_test.p')
#cm.get_classification()
cm.gen_conf_matrix('confusion matrix Random Forest test')
cm.gen_highlights('conf_matrix_highlights')

print 'accuracy with Perceptron test', metrics.accuracy_score(target_test, test_predictions_pc)
print 'precision with Perceptron test', metrics.precision_score(target_test, test_predictions_pc, average = None)
print 'recall with Perceptron test', metrics.recall_score(target_test, test_predictions_pc, average = None)
print 'micro precision with Perceptron test', metrics.precision_score(target_test, test_predictions_pc, average = 'micro')
print 'macro precision with Perceptron test', metrics.precision_score(target_test, test_predictions_pc, average = 'macro')
print 'micro recall with Perceptron test', metrics.recall_score(target_test, test_predictions_pc, average = 'micro')
print 'macro recall with Perceptron test', metrics.recall_score(target_test, test_predictions_pc, average = 'macro')
print

g = open('test_measures_%s_%s_%s_%s.txt' % (first_words, text_bag, topic_model, keep_first), 'w')
# Compute the errors of the three methods on the tests
g.write('accuracy with Naive Bayes test %s\n' % metrics.accuracy_score(target_test, test_predictions_nb))
g.write('precision with Naive Bayes test %s\n' % metrics.precision_score(target_test, test_predictions_nb, average = None))
g.write('recall with Naive Bayes test %s\n' % metrics.recall_score(target_test, test_predictions_nb, average = None))
g.write('micro precision with Naive Bayes test %s\n' % metrics.precision_score(target_test, test_predictions_nb, average = 'micro'))
g.write('macro precision with Naive Bayes test %s\n' % metrics.precision_score(target_test, test_predictions_nb, average = 'macro'))
g.write('micro recall with Naive Bayes test %s\n' % metrics.recall_score(target_test, test_predictions_nb, average = 'micro'))
g.write('macro recall with Naive Bayes test %s\n\n' % metrics.recall_score(target_test, test_predictions_nb, average = 'macro'))

g.write('accuracy with Random Forest test %s\n' % metrics.accuracy_score(target_test, test_predictions_rf))
g.write('precision with Random Forest test %s\n' % metrics.precision_score(target_test, test_predictions_rf, average = None))
g.write('recall with Random Forest test %s\n' % metrics.recall_score(target_test, test_predictions_rf, average = None))
g.write('micro precision with Random Forest test %s\n' % metrics.precision_score(target_test, test_predictions_rf, average = 'micro'))
g.write('macro precision with Random Forest test %s\n' % metrics.precision_score(target_test, test_predictions_rf, average = 'macro'))
g.write('micro recall with Random Forest test %s\n' % metrics.recall_score(target_test, test_predictions_rf, average = 'micro'))
g.write('macro recall with Random Forest test %s\n\n' % metrics.recall_score(target_test, test_predictions_rf, average = 'macro'))

g.write('accuracy with SVM test %s\n' % metrics.accuracy_score(target_test, test_predictions_sv))
g.write('precision with SVM test %s\n' % metrics.precision_score(target_test, test_predictions_sv, average = None))
g.write('recall with SVM test %s\n' % metrics.recall_score(target_test, test_predictions_sv, average = None))
g.write('micro precision with SVM test %s\n' % metrics.precision_score(target_test, test_predictions_sv, average = 'micro'))
g.write('macro precision with SVM test %s\n' % metrics.precision_score(target_test, test_predictions_sv, average = 'macro'))
g.write('micro recall with SVM test %s\n' % metrics.recall_score(target_test, test_predictions_sv, average = 'micro'))
g.write('macro recall with SVM test %s\n\n' % metrics.recall_score(target_test, test_predictions_sv, average = 'macro'))

g.write('accuracy with Perceptron test %s\n' % metrics.accuracy_score(target_test, test_predictions_pc))
g.write('precision with Perceptron test %s\n' % metrics.precision_score(target_test, test_predictions_pc, average = None))
g.write('recall with Perceptron test %s\n' % metrics.recall_score(target_test, test_predictions_pc, average = None))
g.write('micro precision with Perceptron test %s\n' % metrics.precision_score(target_test, test_predictions_pc, average = 'micro'))
g.write('macro precision with Perceptron test %s\n' % metrics.precision_score(target_test, test_predictions_pc, average = 'macro'))
g.write('micro recall with Perceptron test %s\n' % metrics.recall_score(target_test, test_predictions_pc, average = 'micro'))
g.write('macro recall with Perceptron test %s\n\n' % metrics.recall_score(target_test, test_predictions_pc, average = 'macro'))

g.close()





# Beginning of the clustering part

# Create the elements on which doing clustering
clustering = training_doc +  test_doc
# Labels
labels = target_training + target_test

# Create the numpy array
vec = DictVectorizer(sparse=False)
clustering = vec.fit_transform(clustering)

# K-means
km = KMeans(n_clusters = 9)
km.fit(clustering)
print 'Kmeans created..'
# Test over the training data
cluster_predictions_km = km.predict(clustering)

print("Homogeneity k-means: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness k-means: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure k-means: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Silhouette Coefficient k-means: %0.3f" % metrics.silhouette_score(clustering, km.labels_, sample_size = 8000))

# DBSCAN
# Structured hierarchical clustering
db = DBSCAN()
db.fit(clustering)
print 'DBSCAN clusters created..'

print("Homogeneity DBSCAN: %0.3f" % metrics.homogeneity_score(labels, db.labels_))
print("Completeness DBSCAN: %0.3f" % metrics.completeness_score(labels, db.labels_))
print("V-measure DBSCAN: %0.3f" % metrics.v_measure_score(labels, db.labels_))
print("Silhouette Coefficient DBSCAN: %0.3f" % metrics.silhouette_score(clustering, db.labels_, sample_size = 5000))

# Structured hierarchical clustering
ward = Ward(n_clusters = 9)
ward.fit(clustering)
print 'Hierarchical clusters created..'

print("Homogeneity hierarchical: %0.3f" % metrics.homogeneity_score(labels, ward.labels_))
print("Completeness hierarchical: %0.3f" % metrics.completeness_score(labels, ward.labels_))
print("V-measure hierarchical: %0.3f" % metrics.v_measure_score(labels, ward.labels_))
print("Silhouette Coefficient hierarchical: %0.3f" % metrics.silhouette_score(clustering, ward.labels_, sample_size = 5000))




