# This file allows to parse the documents which are contaised isto
# the Reuters-21578 database, analyzisg the tags


# I import the Python library which is used
# to parse the file
import sgmllib
# I import the regex file, used to carry out
# quick research and selection operations over the strings
import re


# I defise a class which is a subclass of the base
# class of the Python SGMLParser. For more documentation
# about this class, see https://docs.python.org/2/library/sgmllib.html
class SGMLReutersParser(sgmllib.SGMLParser):
    """Class to parse a sisgle document of the Reuters-21578 database"""
    
    # Constructor class
    def __init__(self, verbose = 0):
        
        # I call the constructor of the superclass
        sgmllib.SGMLParser.__init__(self, verbose)
        # I isitialize some basic variables
        self._reset()


    # This function is used both for isitializisg and 
    # reset the attributes of the class. It is reset
    # each time an article change
    def _reset(self):
        # Variables to recognize if we are isside the title
        # or not
        self.is_reuters = 0
        self.is_title = 0
        self.is_author = 0
        self.is_dateline = 0
        self.is_date = 0
        self.is_text = 0
        self.is_body = 0
        self.is_topics = 0
        self.is_topic_d = 0
        
        # Variables to memorize the strings I want to keep
        self.had_topic = ""
        self.split = ""
        self.title = ""
        self.author = ""
        self.dateline = ""
        self.place = ""
        self.date = ""
        self.day = ""
        self.month = ""
        self.hour = ""
        self.body = ""
        self.text = ""
        self.text_type = ""
        self.topics = []
        self.topic_d = ""  
        
        # Variable which keeps track of the information
        # related to a single document. It is a dictionary
        # that will be formed by the all information
        self.document = {}


    def parse(self, file_path):
        
        # I open the file
        file = open(file_path, 'r')
        
        # List which will contain a dictionary as element that
        # in turn will be formed by the information related
        # to a single article
        self.documents = []
        
        for row in file:
            # print 'Row', row
            # Gives to the parser the row
            # which has to be parsed
            self.feed(row)
            
        # Close the file
        file.close()
        
        # I return the list of dictionaries
        return self.documents

    # This is called when we are within any tag. Therefore,
    # we have to recognize for the different tags
    def handle_data(self, data):
        if self.is_body:
            self.body += data
        elif self.is_title:
            self.title += data
        elif self.is_topic_d:
            self.topic_d += data
        elif self.is_author:
            self.author += data
        elif self.is_dateline:
            self.dateline += data
        elif self.is_date:
            self.date += data
        # In order to retrieve the complete text
        # I have to write inside the self.text variable
        # when I am within the author, the dateline, the title, 
        # the body for the processed documents. For the unprocessed documents
        # these inner tags does not exist
        if self.is_text:
            self.text += data

    def start_reuters(self, attributes):
        self.is_reuters = 1
        # I store two of the attributes of the reuters tag,
        # namely the topics and lewisplit. These, according to the 
        # README file are always the first two tags
        self.had_topic = attributes[0][1]
        self.split = attributes[1][1]


    # Function which is called when the 
    # document is parsed 
    def end_reuters(self):
        self.is_reuters = 0
        # Using the following expressions to retrieve some information
        self.author = self.author.strip()
        self.author = re.sub(r"by |By ", "", self.author)
        self.author = re.sub(r", Reuters", "", self.author)
        # In self.dateline I am interested only in the city the story comes from
        # This is always the first element of the dateline
        self.place = self.dateline.strip().split(',')[0]
        
        self.day = self.date.split()[0].split('-')[0]
        self.month = self.date.split()[0].split('-')[1]
        self.hour = self.date.split()[1].split(':')[0]
        
        # I create the dictionary related to the document
        document = {'had_topic': self.had_topic,
                    'split': self.split,
                    'title': self.title,
                    'author': self.author,
                    'text_type': self.text_type,
                    'place': self.place,
                    'day': self.day,
                    'month': self.month,
                    'hour': self.hour,
                    'body': self.body,
                    'topics': self.topics,
                    'text': self.text}
        
        # Increment the documents list, which will be formed
        # by the elements related to one single article
        self.documents.append(document)
        
        # Reset the elements related to one article
        self._reset()


    def start_title(self, attributes):
        self.is_title = 1


    def end_title(self):
        self.is_title = 0
        
        
    def start_author(self, attributes):
        self.is_author = 1


    def end_author(self):
        self.is_author = 0
        
        
    def start_dateline(self, attributes):
        self.is_dateline = 1


    def end_dateline(self):
        self.is_dateline = 0
        
        
    def start_date(self, attributes):
        self.is_date = 1


    def end_dateline(self):
        self.is_date = 0
        
        
    def start_text(self, attributes):
        self.is_text = 1
        # I memorize the type of text when there is it.
        # It is the single attribute the text can have
        if len(attributes) == 1:
            self.text_type = attributes[0][1]
        else:
            self.text_type = ""

    def end_text(self):
        self.is_text = 0


    def start_body(self, attributes):
        self.is_body = 1


    def end_body(self):
        self.is_body = 0


    def start_topics(self, attributes):
        self.is_topics = 1


    def end_topics(self):
        self.is_topics = 0


    def start_d(self, attributes):
        # I have to consider the d tag only when it is inside
        # the topic
        if self.is_topics == 1 : 
            self.is_topic_d = 1


    def end_d(self):
        # I write the topic_d element only when we are inside the topics
        # tag, namely when self.is_topics = 1
        if self.is_topic_d == 1:
            self.topics.append(self.topic_d)
            self.topic_d = ""
            self.is_topic_d = 0