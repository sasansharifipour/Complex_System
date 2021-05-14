import numpy as np
import random
import networkx as nx
from IPython.display import Image
import matplotlib.pyplot as plt
from enum import Enum
from scipy.io import mmread

class Datasets(Enum):
    CA_GRQC = 1
    CA_HepPh = 2
    CA_HepTh = 3
    Celegansneural = 4
    Football = 5
    Polblogs = 6
    Power = 7
    Macaque = 8
    USAir97 = 9
    Yeast = 10
    Amazon = 11

    def get_path(self):
        if (self.value == 1):
            return "..\\Datasets\\CA-GrQc.txt"
        elif (self.value == 2):
            return "..\\Datasets\\CA-HepPh.txt";
        elif(self.value == 3):
            return "..\\Datasets\\hep-th.gml";
        elif(self.value == 4):
            return "..\\Datasets\\Celegansneural.gml";
        elif(self.value == 5):
            return "..\\Datasets\\Football.gml";
        elif(self.value == 6):
            return "..\\Datasets\\Polblogs.gml";
        elif(self.value == 7):
            return "..\\Datasets\\Power.gml";
        elif(self.value == 8):
            return "..\\Datasets\\Rhesus_cerebral.cortex_1.graphml";
        elif(self.value == 9):
            return "..\\Datasets\\USair97.net";
        elif(self.value == 10):
            return "..\\Datasets\\YeastS.net";
        elif(self.value == 11):
            return "..\\Datasets\\amazon.net";
        else:
            return "";

class data_set_loader:

    def prepare_dataset(self, graph, remove_edge_fraction):
        edge_subset = random.sample(graph.edges(), int(remove_edge_fraction * graph.number_of_edges()))
        
        graph_train = graph.copy()
        graph_train.remove_edges_from(edge_subset)
        return graph_train, edge_subset

    def load_dataset_with_edge_list(self, file_path):
        fh = open(file_path, 'rb')
        graph = nx.read_edgelist(fh)
        fh.close()
        return graph

    def load_dataset_from_gml_file_without_label(self, file_path):
        graph = nx.read_gml(file_path, label = 'id')
        return graph
    
    def load_dataset_from_gml_file(self, file_path):
        graph = nx.read_gml(file_path)
        return graph
    
    def load_dataset_from_graphml_file(self, file_path):
        graph = nx.read_graphml(file_path)
        return graph
    
    def load_dataset_from_net_file(self, file_path):
        graph = nx.read_pajek(file_path)
        return graph
    
    def print_dataset_base_info(self, graph):
        degree_sequence = list(graph.degree())
        nb_nodes = len(graph.nodes())
        nb_arr = len(graph.edges())
        max_degree = max(np.array(degree_sequence)[:,1])
        min_degree = min(np.array(degree_sequence)[:,1])
        
        print("Number of nodes : " + str(nb_nodes))
        print("Number of edges : " + str(nb_arr))
        print("Maximum degree : " + str(max_degree))
        print("Minimum degree : " + str(min_degree))

    def load_dataset(self,dataset):
        if (dataset in { Datasets.CA_GRQC , Datasets.CA_HepPh} ):
            graph = self.load_dataset_with_edge_list(dataset.get_path())
        elif (dataset in { Datasets.Celegansneural, Datasets.CA_HepTh, Datasets.Football, Datasets.Polblogs} ):
            graph = self.load_dataset_from_gml_file(dataset.get_path())
        elif (dataset in { Datasets.Power } ):
            graph = self.load_dataset_from_gml_file_without_label(dataset.get_path())
        elif (dataset in { Datasets.Macaque } ):
            graph = self.load_dataset_from_graphml_file(dataset.get_path())
        elif (dataset in { Datasets.USAir97, Datasets.Yeast, Datasets.Amazon } ):
            graph = self.load_dataset_from_net_file(dataset.get_path())

        return nx.Graph(graph)
