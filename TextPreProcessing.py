# In this class the pre processing operations are carried out.
# In order to do that, the nltk Python tool was utilized
# If an element is provided to the method, then it works on it.
# Otherwise a default element representing the text/list constructed up to now 
# is used. In all the cases the element after the function application is returned


# nltk module
import nltk

# Regex module
import re


class TextPreProcessing:
    
    # Init file. It accepts the document which has to be processed
    def __init__(self, text):
        # self.text = '&#2; ******CCR VIDEO SAYST RECEIVED OFFER TO NEGOTIATE A TAKEOVER BY INTERCEP INVESTMENT CORP Blah blah blah. &#3;'
        # self.text = "That U.S.A.'friend poster-print don't won't jbjhb!! you which used the car AHAHAHA 'O01'2sk21' ok I'm and you i am costs He's $12.40..."
        # text = "I went to New York to meet John Smith!"
        # print text
        
        # Updated element on which to work if nothing is provided
        self.updated_element = text
        
        # Initializing the elements
        self._reset()
    
    def _reset(self):
        # Items produced by the corresponding methods
        self.spell_checked = []
        self.tokenized = []
        self.lowercased = []
        self.removed_punctuation = []
        self.removed_numbers = []
        self.pos_tagged = []
        self.removed_stopwords = []
        self.lemmatized = []
        self.entity_named = []
        self.final_list = []

        
    # It adjust some word spells in English and works only
    # on text
    def spell_check(self, text = None):
        
        # If nothing is provided I use the default 
        # text given during the creation
        if text is None:
            text = self.updated_element
        
        # A problem to deal with in the tokenization regards the character "'". 
        # In English this character is used both as English possessive and contraction.
        # Before starting the tokenization, I modify the text string by expliciting 
        # SOME of the contraction forms (which can be visualized at http://en.wikipedia.org/wiki/Contraction_%28grammar%29#English)       
        self.spell_checked = re.sub(r"n't|N'T", " not", text)
        self.spell_checked = re.sub(r"I'm|I'M", "I am", self.spell_checked)
        self.spell_checked = re.sub(r"Let's|let's|LET'S", "let us", self.spell_checked)
        self.spell_checked = re.sub(r"'re|'RE", " are", self.spell_checked)
        # This following three are not always true
        # but many times they are
        self.spell_checked = re.sub(r"it's|It's|IT'S", "it is", self.spell_checked)
        self.spell_checked = re.sub(r"he's|He's|HE'S", "he is", self.spell_checked)
        self.spell_checked = re.sub(r"she's|She's|SHE'S", "she is", self.spell_checked)
        self.spell_checked = re.sub(r"'ve|'VE", " have", self.spell_checked)
        self.spell_checked = re.sub(r"'ll|'LL", " will", self.spell_checked)
        self.spell_checked = re.sub(r"'em|'EM", " them", self.spell_checked)
        
        # print 'Spell checked', self.spell_checked
        
        # I update the element
        self.updated_element = text
        
        return self.updated_element
        
    
    # It performs the tokenization task 
    # on the selected document
    def tokenize(self, text = None):
        
        if text is None:
            text = self.updated_element

        # Create the pattern using the regex
        # The words as 21893hjk are allowed, namely they are not splitted
        pattern = r'''(?x) ([A-Z]\.)+ | [^\w\s]+ | \d+[\.,]\d+ |\w+ '''
        
        # I could use the default tokenize function of nltk but it does not allow to separate
        # properly some strings and puntuaction characters
        # self.tokens = nltk.word_tokenize(self.text)
        # With this command I have more control instead
        self.tokenized = nltk.regexp_tokenize(text, pattern)
        # print 'Tokenized', self.tokenized
        
        # I update the element
        self.updated_element = self.tokenized
        
        return self.updated_element
        
        
    # It performs the normalization of the text:
    # all the world are put as lowercases. It accepts a list of tuples
    # or a simple list.
    def lowercase(self, element = None):
        
        if element is None:
            element = self.updated_element
        
        if type(element[0]) == tuple:
            self.lowercased = [(x.lower(),y) for (x,y) in element]
        else:
            self.lowercased = [x.lower() for x in element]
        # print 'Lowercase', self.lowercased
        
        # I update the element
        self.updated_element = self.lowercased
        
        return self.updated_element
        
        
    # I remove the punctuation on the obtained list
    # element can be either a list of tuples (after the POS tagging)
    # or a list (before the post tagging)
    def remove_punctuation(self, element = None):
        
        if element is None:
            element = self.updated_element
        
        # Using the regex to determine all those words that
        # are not formed by external characters
        if type(element[0]) == tuple:
            self.removed_punctuation = [(word,tag) for (word,tag) in element if re.search('\w+', word)]
        else:
            self.removed_punctuation = [word for word in element if re.search('\w+', word)]
            
        # print 'Remove punctuation', self.removed_punctuation

        # I update the element
        self.updated_element = self.removed_punctuation
        
        return self.updated_element
    
    
    # I remove the numbers on the obtained list.
    # The code remove all the elements containing one or more digits
    def remove_numbers(self, element = None):
        
        if element is None:
            element = self.updated_element
        
        # Using the regex to determine all those words that
        # are not formed by external characters
        if type(element[0]) == tuple:
            self.removed_numbers = [(word,tag) for (word,tag) in element if not re.search('\d+', word)]
        else:
            self.removed_numbers = [word for word in element if not re.search('\d+', word)]
            
        # print 'Remove numbers', self.removed_numbers

        # I update the element
        self.updated_element = self.removed_numbers
        
        return self.updated_element
            
        
    # POS tagging
    def POS_tagging(self, list = None):
        
        if list is None:
            list = self.updated_element
        
        # Performing the POS tagging with the nltk library
        self.pos_tagged = nltk.pos_tag(list)
        # print 'POS tagging', self.pos_tagged

        # I update the element
        self.updated_element = self.pos_tagged
        
        return self.updated_element
           
        
    # I remove the stop words
    # element can be either a list of tuples (after the POS tagging)
    # or a list (before the post tagging)
    def remove_stopwords(self, element = None):
        
        if element is None:
            element = self.updated_element
            
        forbidden = ['Reuter', 'reuter']
        
        stop_words = nltk.corpus.stopwords.words('english') + forbidden
        
        if type(element[0]) == tuple:
            self.removed_stopwords = [(word, tag) for (word, tag) in element if not word in stop_words]
        else:
            self.removed_stopwords = [word for word in element if not word in stop_words]
            
        # print 'Remove stopwords', self.removed_stopwords

        # I update the element
        self.updated_element = self.removed_stopwords
        
        return self.updated_element
            
    
    # I perform the lemmatization of the text
    # Having the pos_tagging I can perform this process
    # by using the new tag information
    # It is important in fact to observe that the lemmatization
    # accept a parameter as input to specify the tag
    def lemmatize(self, tuple_list = None):
        
        if tuple_list is None:
            tuple_list = self.updated_element
        
        # Creating the lemmatizer
        wnl = nltk.WordNetLemmatizer()
        
        # Extract the original lemma
        for word, tag in tuple_list:         
            if self.get_wordnet_pos(tag) == '':
                self.lemmatized.append((wnl.lemmatize(word), tag))
            else:
                self.lemmatized.append((wnl.lemmatize(word, self.get_wordnet_pos(tag)), tag))
        
        # Print lemmatized elements
        # print 'Lemmatized', self.lemmatized

        # I update the element
        self.updated_element = self.lemmatized
        
        return self.updated_element
            
        
    # Retrieve the name entities
    def entity_names(self, tuple_list = None):

        if tuple_list is None:
            tuple_list = self.updated_element        
        
        # Recognize the names of the entities contained in the string
        tree = nltk.ne_chunk(tuple_list, binary=False)
        
        # At this point instead of the entity names I substitute the 
        # name of the entity. In order to recognize the names having the 
        # entity name I simply make a check on the final name of the element
        # if it is different from 'S', then it was a valid node
        for el in tree:
            
            if type(el) == nltk.tree.Tree:
                # If it was an entity then the tag can be 
                # stored as a name
                self.entity_named.append((el.node, 'NNP'))
            else:
                self.entity_named.append(el)
        
        # Print
        # print 'Named entities', self.entity_named
        
        # I update the element
        self.updated_element = self.entity_named
        
        return self.updated_element
        
        
    # Return the translation of the original tag
    # in a single character which is accepted
    # by the lemmatizer
    def get_wordnet_pos(self, treebank_tag):

        # I can have different cases
        if treebank_tag.startswith('J'):
            return nltk.corpus.reader.wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return nltk.corpus.reader.wordnet.VERB
        elif treebank_tag.startswith('N'):
            return nltk.corpus.reader.wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return nltk.corpus.reader.wordnet.ADV
        else:
            return ''
        
        
    # Return a list from a tuple_list
    def return_list(self, element = None):
        
        if element is None:
            element = self.updated_element   
    
        if type(element[0]) == tuple:
            self.final_list = [x for (x,tag) in element]
        else:
            self.final_list = element
        
        # print 'Final list', self.final_list
        
        self.updated_element = self.final_list
            
        return self.updated_element
        