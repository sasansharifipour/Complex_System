import numpy as np
import random
import networkx as nx
from IPython.display import Image
import matplotlib.pyplot as plt
from data_set_loader import *
from CommonCodes import *
from calculating_features import *
from Feature_Calculator_Enum import *
import csv

CC = Common_Codes()
graph = CC.create_paper_graph()

CC.Convert_graph_to_adj_list(graph, "paper_graph_adj_list.txt")
