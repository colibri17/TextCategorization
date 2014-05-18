# Retrieve the model of the topic
# Also it allows to perform some selection on the text
from gensim import corpora, models

class LdaTopicModels:
    
    # Init function
    def __init__(self):
        
        # It will contain the dictionary related to the corpora
        self.dictionary = {}
        self.dictionary_token2id = {}
        
    
    # Creating the dictionary
    def create_dictionary(self, documents):
        self.dictionary = corpora.Dictionary(documents)
        return self.dictionary
    
        
    # Visualize the dictionary as couple of token and id
    def token2id_dictionary(self):
        self.dictionary_token2id = self.dictionary.token2id
        return self.dictionary_token2id
    
    # Convert the document provided into vector according to the created dictionary
    def convert_into_vector(self, doc):
        return self.dictionary.doc2bow(doc)
    
    # Convert the list of documents into a list of corpus
    def create_vector_corpus(self, documents):
        self.corpus = [self.dictionary.doc2bow(doc) for doc in documents]
        return self.corpus
    
    # Create the model by using the dictionaries elements
    # and the training as training set
    def create_model(self, training, num_topics = 100):
        self.model = models.ldamodel.LdaModel(training, id2word = self.dictionary, num_topics = num_topics)
        return self.model
    
    # Create the tfidf model
    def create_tfidf(self, training):
        self.model = models.TfidfModel(training)
        return self.model
    
    # Evaluate the model on a given corpus
    def evaluate_model(self, corpus):
        return self.model[corpus]
    
    # Get the topic names
    def get_topics(self, n):
        return self.model.show_topics(topics=n)
        
        
    