#add this imports
from neat_gp import *
from crosspoints import *

#on the beginning of the class just add
#the parameter 'neat'

#class PrimitiveTree(list, neat):

#modify the init function
def __init__(self, content):
        list.__init__(self, content)
        self.tspecie=None
        self.descendent=None
        self.fitness_h=None
        self.nspecie=None

#@property
#         def params(self):
#         l=len(self)+4
#         p=np.ones(l)
#         return p