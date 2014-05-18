''' Confusion matrix util class:
    - simple visualization
    - to precision and recall transformation
    - to classification erreur
    - generate confusion hightlights  
    - '''

import numpy as np
# Import the matplotlib library in order to print the histogram
# and also make comparison with the normal distribution
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.font_manager

# These three lines are needed in order to save the plot in a file such that it can be
# later easily included in a latex file
font = {'size'   : 6}
matplotlib.rc('font', **font)
fig = plt.figure(figsize = (6,5))

import pickle

class ConfMatrix:
    def __init__(self, matrix=None, labels=None):
        self.m = None
        if matrix!=None:
            self.m = matrix
        if labels:
            self.labels = labels

    @staticmethod
    def load_matrix(fname):
        f = open(fname, "r")
        self.m = pickle.load(f)
        self.labels = pickle.load(f)
        f.close()

    def save_matrix(self, fname, labels=None):
        if '.' not in fname:
            fname+=".pickle"
        f = open(fname, "w")
        pickle.dump(self.m , f)
        pickle.dump(self.labels , f)
        print("saving %s" %fname)
        f.close()

    def get_classification(self, labels=None, 
                           rm_labels=['tbd', '?','TBD']):
        labels = labels or self.labels
        idxs = [idx for idx, el in enumerate(labels) if el not in rm_labels]
        rm_idxs = [idx for idx, el in enumerate(labels) if el in rm_labels]
        if len(rm_idxs):
            print("don't consider %s" %(','.join(["%s->%s" %(label, idx) for label, idx in zip(rm_labels, rm_idxs)])))
        total = self.m[idxs,:].sum()
        target = 0.0
        for i in idxs:
            target+=self.m[i][i]
        print("global precision: %i/%i=%2.2f" %(target, total, target/total))
        return target/total

    def to_precision(self):
        precision = np.array(self.m, dtype=float)
        precision/=self.m.sum(axis=0)    
        precision*=100
        return precision

    def to_recall(self):
        recall = np.array(self.m, dtype=float)
        recall/=self.m.sum(axis=1)[:,np.newaxis]
        recall*=100
        return recall

    def _gen_conf_matrix(self, fname, labels=None, title=None, threshold=.0, factor=1, normalize=""):
        fname = fname or "conf_matrix"
        labels = labels or self.labels
        matrix = self.m
        if normalize!='':
            fname+= "_%s" %normalize
        title = title or fname
        #title += " %s" %normalize
        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(2)
        matrix = np.array(matrix, dtype=float)
        res = ax.imshow(matrix, cmap='Greys', 
                        interpolation='nearest')
        
        size = matrix.shape[0]

        for x in xrange(size):
            for y in xrange(size):
                if abs(matrix[x][y])>=threshold:
                    value = "%2.0f" %(matrix[x][y]*factor)
                    ax.annotate(value, xy=(y, x), 
                                horizontalalignment='center',
                                verticalalignment='center', color='blue')
                
        cb = fig.colorbar(res)
    
        plt.xticks(range(size), labels, rotation='vertical')
        plt.yticks(range(size), labels)
        plt.title(title)
        plt.xlabel('Predicted class')
        plt.ylabel('Actual class')
        fname+=".png"
        fname = fname.replace(' ','')
        plt.savefig(fname, format='png')
        print("generated : %s" %fname)

    def gen_conf_matrix(self, fname, labels=None, title=None, threshold=.0, factor=1):
        fname = fname or "conf_matrix"
        labels = labels or self.labels
        self._gen_conf_matrix(fname, labels, title=title, threshold=threshold, factor=factor)

    def gen_highlights(self, fname, labels=None, val_threshold=0):
        from mlboost.core.pphisto import SortHistogram
        confusion_matrix = self.m
        labels = labels or self.labels
        fname+=".highlight.txt"
        with open(fname, 'w') as fout:
            for label in labels:
                i = 0
                dist = {}
                labels2 = list(labels)
                labels2.remove(label)
                for label_2 in labels2:
                    if isinstance(confusion_matrix, list):
                        dist[label_2] = confusion_matrix[label_2][label]
                    else: # assume numpy matrix
                        dist[label_2] = confusion_matrix[labels.index(label_2)][labels.index(label)]
                        
                        
                sdist = SortHistogram(dist, False, True)
            
                confusions = ["%s %2.0f" %(key, value) for key, value in sdist if value >val_threshold 
                              and key not in('?','tbd')]
                
                if isinstance(confusion_matrix, list):
                    fout.write('\n%s (%2.0f%%) -> ' %(label,confusion_matrix[label][label]*100))
                else: # assume numpy matrix
                    idx = labels.index(label)
                    fout.write('\n%s (%2.0f) -> ' %(label,confusion_matrix[idx][idx]))
                fout.write(' | '.join(confusions))

                       
        print("generated : %s" %fname)
    

if __name__ == "__main__":
    from optparse import OptionParser
    import argparse
    import pickle
    import sys
    parser = OptionParser(description=__doc__)
    
    parser.add_option("-m", dest="matrix_fname", default = None, 
                      help="pickle matrix fname")
    parser.add_option("-2", dest="matrix_fname2", default = None, 
                      help="second matrix fname (show diff)")
    parser.add_option("-t", dest="title", default = None, 
                      help="title")
    parser.add_option("-M", dest="threshold", default = .05, type=float, 
                        help="min threshold")
    parser.add_option("-o", dest='output', default=None,
                        help="outputfile")
    options, args = parser.parse_args()

    if options.matrix_fname:
        matrix = ConfMatrix.load_matrix(options.matrix_fname)
    else:
        sys.exit(1)
    
    output = options.output or options.matrix_fname

    if options.matrix_fname2:
            matrix2 = Matrix.load_matrix(options.matrix_fname2)
            matrix.d-= matrix2.d
            
    matrix.gen_conf_matrix(matrix.labels, output, title=options.title, threshold=options.threshold)
    matrix.gen_highlights(output, matrix.labels)