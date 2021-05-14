import networkx as nx
import random
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

class Common_Codes:
    
    def create_paper_graph(self):
        graph = nx.Graph()

        #------------ add nodes ------------------
        graph.add_node('A')
        graph.add_node('B')
        graph.add_node('C')
        graph.add_node('D')
        graph.add_node('E')
        graph.add_node('F')
        graph.add_node('G')
        graph.add_node('H')
        graph.add_node('I')
        graph.add_node('J')
        graph.add_node('K')

        #------------ add edges -------------------
        graph.add_edge('A', 'C')
        graph.add_edge('A', 'D')
        graph.add_edge('A', 'F')
        graph.add_edge('B', 'C')
        graph.add_edge('B', 'D')
        graph.add_edge('B', 'E')
        graph.add_edge('C', 'F')
        graph.add_edge('C', 'D')
        graph.add_edge('C', 'G')
        graph.add_edge('D', 'E')
        graph.add_edge('E', 'J')
        graph.add_edge('E', 'K')
        graph.add_edge('F', 'G')
        graph.add_edge('F', 'H')
        graph.add_edge('F', 'I')
        graph.add_edge('H', 'I')
        graph.add_edge('J', 'K')

        return graph

    def Convert_graph_to_adj_list(self, graph, save_file_path):
        f= open(save_file_path,"w+")
        nodes = list(graph.nodes())

        for node in nodes:
            neighbors = list(graph.neighbors(node))
            lst = node
            
            for neighbor in neighbors:
                lst = lst + ' ' + neighbor

            f.write(lst + '\n')

        f.close()            
        
    def Convert_Tuple_To_Dictionary(self, tup):
        result = {}

        for a, b in tup: 
            result.setdefault(a, []).append(b) 

        return result 

    def Convert_Degree_Tuple_To_Dictionary(self, tup):
        result = {}

        for a, b in tup: 
            result[a] = b

        return result

    def intersection(self, lst1, lst2): 
        return list(set(lst1) & set(lst2))

    def union(self, lst1, lst2): 
        return list(dict.fromkeys(sorted(lst1 + lst2)))

    def minus_list(self, lst1, lst2): 
        return list([x for x in lst1 if x not in lst2])

    def unique(self, lst):
        list_set = set(lst)
        unique_list = (list(list_set))
        return unique_list

    def correct_selected_edges(self, graph, ranks):
        result = np.zeros(len(ranks))
        cnt = 0
        graph_edges = list(graph.edges())

        for k in ranks:
            if any(k[0] == i for i in graph_edges):
                result[cnt] = 1
            else:
                result[cnt] = 0
                            
            cnt = cnt + 1

        return result
    
    def calculate_auc_for_link_prediction(self, n, removed_link_ranks, absent_link_ranks):
        n_prime = 0
        n_zegond = 0
        removed_link_ranks_lst = list(removed_link_ranks.values())
        absent_link_ranks_lst = list(absent_link_ranks.values())

        for i in range(n):
            removed_rank = int(random.choice(removed_link_ranks_lst))
            absent_rank = int(random.choice(absent_link_ranks_lst))

            if (removed_rank > absent_rank):
                n_prime = n_prime + 1
            elif (removed_rank == absent_rank):
                n_zegond = n_zegond + 1
        
        return ((n_prime + (0.5 * n_zegond)) / n)

    def calculate_auc(self, probs):
        y = list(np.ones(len(probs)))
        probs = list(probs)
        probs.append(0)
        y.append(0)
        auc = roc_auc_score(y, probs)
        return auc
