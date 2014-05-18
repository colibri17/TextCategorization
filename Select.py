# Allows to perform the selection on the desired feature of 
# the documents. It allows to select only some words 
# in the text of the document of cull only determined features

# Import nltk
import nltk


class Select:
    
    def __init__(self, all_words):
        
        # List containing all the words which will be utilized
        # up to the time the class is initialized
        self.all_words = all_words
        
        # It will contain the list of the selected words
        # Initially all the words are selected
        self.selected_words = all_words
        
        # Distribution related to all the words
        self.freq_distribution_all = [] 
        
        
    # It creates the frequency_distribution starting from all the words list
    # Starting from this frequency distribution one can filter with respect to certain conditions
    def frequency_distribution_doc(self, word_list):     
        # Creating the frequencies of all the provided words
        return nltk.FreqDist(word_list)
    
    
    # Creates the frequency distribution of all the words in all the documents
    def frequency_distribution_all(self):
        # Creating the frequencies of all the words
        self.freq_distribution_all = self.frequency_distribution_doc(self.all_words)
        return self.freq_distribution_all

    
    # It selects the first n elements in the frequency distribution, namely
    # the n most frequent rows
    def select_n_frequent_words_all(self, n):
        self.selected_words = self.frequency_distribution_all().keys()[:n]
        # Update the all words list after this selection
        self.all_words = self.selected_words
        return self.selected_words
        
    
    # It selects the words which are more frequent than the index selected by the user
    # erasing those which does not match this threshold value
    def select_upto_n_frequent_words_all(self, n):
        self.selected_words = [word for (word,freq) in self.frequency_distribution_all().items() if freq >= n]
        # Update the all words list after this selection
        self.all_words = self.selected_words
        return self.selected_words
    

    # It selects the first n elements in the frequency distribution, namely
    # the n most frequent words in the doc. For example it selects the most 10 frequent rows if n = 10
    def select_n_frequent_words_doc(self, n, word_list):
        return self.frequency_distribution_doc(word_list).keys()[:n]
        
    
    # It selects the words which are more frequent in the doc than the index selected by the user
    # erasing those which does not match this threshold value. For example it selects the words more frequent than or equal to 3 if n = 3
    def select_upto_n_frequent_words_doc(self, n, word_list):
        return [word for (word,freq) in self.frequency_distribution_doc(word_list).items() if freq >= n]
    
    
    # It only returns the words in the text_list that are in the
    # selected list given by self.selected_words
    def select_words_in_doc(self, text_list):       
        # Return the words
        return [word for word in text_list if word in self.selected_words]
        
    
    
    
    