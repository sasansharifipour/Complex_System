import networkx as nx
from itertools import islice
from CommonCodes import *
from Feature_Calculator_Enum import *
import math
from sklearn.svm import OneClassSVM

class Feature_Calculator:

    def __init__(self, graph):
        self.graph = graph
        self.commoncodes = Common_Codes()
        self.clustering_coefficient = nx.clustering(graph)
        self.density = nx.density(graph)
        self.LNBCN_log = math.log10( (1 - self.density ) / self.density )

    def get_nodes(self):
        return(list(self.graph.nodes()))

    def take(self, n, iterable):
        return list(islice(iterable, n))

    def train_one_class_classifier(self, data):
        train = np.reshape(list(data.values()), (1, len(data))).T
        mdl = OneClassSVM().fit(train)
        
        return mdl
            
    def predict_by_one_class_classifier(self, mdl, data):
        test = np.reshape(list(data.values()), (1, len(data))).T
        keys = list(data.keys())
        
        predicted_values = mdl.score_samples(test)
        result = data.copy()

        for i in range(0, len(predicted_values)):
            result[keys[i]] = predicted_values[i]

        return result
        
    def get_common_neighbor_list(self):
        nodes = list(self.graph.nodes())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                value = list(nx.common_neighbors(self.graph, nodes[i], nodes[j]))
                result[key] = value

        return result

    def get_level2_common_neighbor_list(self, a, b):
        result = []
        common_neighbors_level_1 = list(nx.common_neighbors(self.graph, a, b))

        for i in common_neighbors_level_1:
            result.extend(list(nx.common_neighbors(self.graph, a, i)))

        result = self.commoncodes.minus_list(result, [a, b])
        result = self.commoncodes.unique(self.commoncodes.minus_list(result, common_neighbors_level_1))
        return result
    
    def get_suggested_edges(self, all_edges_ranks, count_of_suggests):
        result = {}
        srt_cn = dict(sorted(all_edges_ranks.items(), key=lambda item: item[1], reverse=True))

        suggested_items = self.take(count_of_suggests, srt_cn.items())

        return suggested_items
    
    def split_ranks_to_absent_removed_link_ranks(self, all_edges_ranks, removed_links):
        absent_link_ranks = all_edges_ranks.copy()
        removed_link_ranks = {}

        for i in removed_links:
            exist_item = absent_link_ranks.pop(i, None)
            removed_link_ranks[i] = exist_item

        return absent_link_ranks, removed_link_ranks
    
    def get_ranks_exist_links(self, all_edges_ranks):
        not_exist_data = all_edges_ranks.copy()
        exist_data = {}
        graph_edges = list(self.graph.edges())

        for i in graph_edges:
            exist_item = not_exist_data.pop(i, None)
            exist_data[i] = exist_item

        return not_exist_data, exist_data

    def Calculate_Feature(self, feature_calculator, feature_level):
        if (feature_calculator == Feature_Calculator_Enum.Common_Neighbor):
            if (feature_level == Feature_Level_Enum.Level_1):
                return self.calculate_common_neighbor_number()
            elif (feature_level == Feature_Level_Enum.Level_2):
                return self.calculate_common_neighbor_level_2_number()
            elif (feature_level == Feature_Level_Enum.Level_1_And_2):
                return self.calculate_common_neighbor_level_1_with_level_2_number()

        elif (feature_calculator == Feature_Calculator_Enum.Preferential_Attachment):
            if (feature_level == Feature_Level_Enum.Level_1):
                return self.calculate_Preferential_Attachment_number()
            elif (feature_level == Feature_Level_Enum.Level_2):
                return self.calculate_Preferential_Attachment_level_2_number()
            elif (feature_level == Feature_Level_Enum.Level_1_And_2):
                return self.calculate_Preferential_Attachment_level_1_with_level_2_number()
    
        elif (feature_calculator == Feature_Calculator_Enum.Resource_Allocation):
            if (feature_level == Feature_Level_Enum.Level_1):
                return self.calculate_Resource_Allocation_number()
            elif (feature_level == Feature_Level_Enum.Level_2):
                return self.calculate_Resource_Allocation_level_2_number()
            elif (feature_level == Feature_Level_Enum.Level_1_And_2):
                return self.calculate_Resource_Allocation_level_1_with_level_2_number()
    
        elif (feature_calculator == Feature_Calculator_Enum.Adamic_Adar):
            if (feature_level == Feature_Level_Enum.Level_1):
                return self.calculate_Adamic_Adar_number()
            elif (feature_level == Feature_Level_Enum.Level_2):
                return self.calculate_Adamic_Adar_level_2_number()
            elif (feature_level == Feature_Level_Enum.Level_1_And_2):
                return self.calculate_Adamic_Adar_level_1_with_level_2_number()
    
        elif (feature_calculator == Feature_Calculator_Enum.Jaccard):
            if (feature_level == Feature_Level_Enum.Level_1):
                return self.calculate_jaccard_number()
            elif (feature_level == Feature_Level_Enum.Level_2):
                return self.calculate_jaccard_level_2_number()
            elif (feature_level == Feature_Level_Enum.Level_1_And_2):
                return self.calculate_jaccard_level_1_with_level_2_number()
    
        elif (feature_calculator == Feature_Calculator_Enum.Clustering_Coefficient):
            if (feature_level == Feature_Level_Enum.Level_1):
                return self.calculate_Clustering_Coefficient_based_Link_Prediction_number()
            elif (feature_level == Feature_Level_Enum.Level_2):
                return self.calculate_Clustering_Coefficient_based_Link_Prediction_level_2_number()
            elif (feature_level == Feature_Level_Enum.Level_1_And_2):
                return self.calculate_Clustering_Coefficient_based_Link_Prediction_level_1_with_level_2_number()
    
        elif (feature_calculator == Feature_Calculator_Enum.Local_Naive_Bayes_based_Common_Neighbor):
            if (feature_level == Feature_Level_Enum.Level_1):
                return self.calculate_Local_Naive_Bayes_based_Common_Neighbor_number()
            elif (feature_level == Feature_Level_Enum.Level_2):
                return self.calculate_Local_Naive_Bayes_based_Common_Neighbor_level_2_number()
            elif (feature_level == Feature_Level_Enum.Level_1_And_2):
                return self.calculate_Local_Naive_Bayes_based_Common_Neighbor_level_1_with_level_2_number()
    
        elif (feature_calculator == Feature_Calculator_Enum.Node_and_Link_Clustering_Coefficient):
            if (feature_level == Feature_Level_Enum.Level_1):
                return self.calculate_Node_and_Link_Clustering_Coefficient_number()
            elif (feature_level == Feature_Level_Enum.Level_2):
                return self.calculate_Node_and_Link_Clustering_Coefficient_level_2_number()
            elif (feature_level == Feature_Level_Enum.Level_1_And_2):
                return self.calculate_Node_and_Link_Clustering_Coefficient_level_1_with_level_2_number()
    
        elif (feature_calculator == Feature_Calculator_Enum.CAR):
            if (feature_level == Feature_Level_Enum.Level_1):
                return self.calculate_CAR_number()
            elif (feature_level == Feature_Level_Enum.Level_2):
                return self.calculate_CAR_level_2_number()
            elif (feature_level == Feature_Level_Enum.Level_1_And_2):
                return self.calculate_CAR_level_1_with_level_2_number()
            
    def calculate_common_neighbor_number(self):
        nodes = list(self.graph.nodes())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                value = len(list(nx.common_neighbors(self.graph, nodes[i], nodes[j])))
                result[key] = value

        return result

    def calculate_Preferential_Attachment_number(self):
        nodes = list(self.graph.nodes())
        nodes_degree = self.commoncodes.Convert_Degree_Tuple_To_Dictionary(self.graph.degree())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                value = nodes_degree[nodes[i]] * nodes_degree[nodes[j]]
                result[key] = value

        return result

    def calculate_Resource_Allocation_number(self):
        nodes = list(self.graph.nodes())
        nodes_degree = self.commoncodes.Convert_Degree_Tuple_To_Dictionary(self.graph.degree())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                value = 0 
                neighbors = list(nx.common_neighbors(self.graph, nodes[i], nodes[j]))
                for k in neighbors :
                    value = value + 1 / nodes_degree[k]
                result[key] = value

        return result

    def calculate_Adamic_Adar_number(self):
        nodes = list(self.graph.nodes())
        nodes_degree = self.commoncodes.Convert_Degree_Tuple_To_Dictionary(self.graph.degree())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                value = 0 
                neighbors = list(nx.common_neighbors(self.graph, nodes[i], nodes[j]))
                for k in neighbors :
                    value = value + 1 / math.log10( max(0.000001, nodes_degree[k]) )
                result[key] = value

        return result
    
    def calculate_jaccard_number(self):
        nodes = list(self.graph.nodes())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                common_neighbors = len(list(nx.common_neighbors(self.graph, nodes[i], nodes[j])))
                union_neighbors = len(self.commoncodes.union(list(self.graph.neighbors(nodes[i])), list(self.graph.neighbors(nodes[j]))))
                value = common_neighbors / union_neighbors
                result[key] = value

        return result
    
    def calculate_Clustering_Coefficient_based_Link_Prediction_number(self):
        nodes = list(self.graph.nodes())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                value = 0 
                neighbors = list(nx.common_neighbors(self.graph, nodes[i], nodes[j]))
                for k in neighbors :
                    value = value + self.clustering_coefficient[k]
                result[key] = value

        return result
    
    def calculate_Local_Naive_Bayes_based_Common_Neighbor_number(self):
        nodes = list(self.graph.nodes())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                value = 0 
                neighbors = list(nx.common_neighbors(self.graph, nodes[i], nodes[j]))
                for k in neighbors :
                    value = value + math.log10(max(0.00001,(self.clustering_coefficient[k]) / (1.0000001 - self.clustering_coefficient[k]))) +  self.LNBCN_log
                result[key] = value

        return result
    
    def calculate_Node_and_Link_Clustering_Coefficient_number(self):
        nodes = list(self.graph.nodes())
        nodes_degree = self.commoncodes.Convert_Degree_Tuple_To_Dictionary(self.graph.degree())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                value = 0 
                neighbors = list(nx.common_neighbors(self.graph, nodes[i], nodes[j]))
                for k in neighbors :
                    a_and_c_common_neighbor = list(nx.common_neighbors(self.graph, nodes[i], k))
                    b_and_c_common_neighbor = list(nx.common_neighbors(self.graph, nodes[j], k))
                    value = value + ((self.clustering_coefficient[k]) / (nodes_degree[k] - 1)) * (len(a_and_c_common_neighbor) + len(b_and_c_common_neighbor))
                result[key] = value

        return result
       
    def calculate_CAR_number(self):
        nodes = list(self.graph.nodes())
        nodes_degree = self.commoncodes.Convert_Degree_Tuple_To_Dictionary(self.graph.degree())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                value = 0 
                neighbors = list(nx.common_neighbors(self.graph, nodes[i], nodes[j]))
                for k in neighbors :
                    c_neighbors = list(nx.neighbors(self.graph, k))
                    c_common_neighbor_with_a_and_b = self.commoncodes.intersection(neighbors, c_neighbors)
                    value = value + len(c_common_neighbor_with_a_and_b) / 2
                result[key] = value * len(neighbors)

        return result
        
    def calculate_common_neighbor_level_2_number(self):
        nodes = list(self.graph.nodes())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                level_2_from_i = self.get_level2_common_neighbor_list(nodes[i], nodes[j])
                level_2_from_j = self.get_level2_common_neighbor_list(nodes[j], nodes[i])
                level_2 = self.commoncodes.unique(self.commoncodes.union(level_2_from_i, level_2_from_j))
                key = (nodes[i], nodes[j])
                value = len(level_2)
                result[key] = value

        return result

    def calculate_common_neighbor_level_1_with_level_2_number(self):
        nodes = list(self.graph.nodes())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                level_1 = list(nx.common_neighbors(self.graph, nodes[i], nodes[j]))
                level_2_from_i = self.get_level2_common_neighbor_list(nodes[i], nodes[j])
                level_2_from_j = self.get_level2_common_neighbor_list(nodes[j], nodes[i])
                level_2 = self.commoncodes.unique(self.commoncodes.union(level_2_from_i, level_2_from_j))
                level_1_with_level_2 = self.commoncodes.unique(self.commoncodes.union(level_1, level_2))
                key = (nodes[i], nodes[j])
                value = len(level_1_with_level_2)
                result[key] = value

        return result

    def calculate_Preferential_Attachment_level_2_number(self):
        nodes = list(self.graph.nodes())
        nodes_degree = self.commoncodes.Convert_Degree_Tuple_To_Dictionary(self.graph.degree())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                value = 0 
                neighbors = list(nx.common_neighbors(self.graph, nodes[i], nodes[j]))
                for k in neighbors :
                    value += nodes_degree[k] * ( nodes_degree[nodes[i]] + nodes_degree[nodes[j]])
                result[key] = value

        return result

    def calculate_Preferential_Attachment_level_1_with_level_2_number(self):
        nodes = list(self.graph.nodes())
        nodes_degree = self.commoncodes.Convert_Degree_Tuple_To_Dictionary(self.graph.degree())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                value = nodes_degree[nodes[i]] * nodes_degree[nodes[j]] 
                neighbors = list(nx.common_neighbors(self.graph, nodes[i], nodes[j]))
                for k in neighbors :
                    value += nodes_degree[k] * ( nodes_degree[nodes[i]] + nodes_degree[nodes[j]])
                result[key] = value

        return result

    def calculate_Resource_Allocation_level_2_number(self):
        nodes = list(self.graph.nodes())
        nodes_degree = self.commoncodes.Convert_Degree_Tuple_To_Dictionary(self.graph.degree())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                value = 0 
                level_2_from_i = self.get_level2_common_neighbor_list(nodes[i], nodes[j])
                level_2_from_j = self.get_level2_common_neighbor_list(nodes[j], nodes[i])
                level_2 = self.commoncodes.unique(self.commoncodes.union(level_2_from_i, level_2_from_j))
                for k in level_2 :
                    value = value + 1 / nodes_degree[k]
                result[key] = value

        return result
    
    def calculate_Resource_Allocation_level_1_with_level_2_number(self):
        nodes = list(self.graph.nodes())
        nodes_degree = self.commoncodes.Convert_Degree_Tuple_To_Dictionary(self.graph.degree())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                value = 0
                level_1 = list(nx.common_neighbors(self.graph, nodes[i], nodes[j]))
                level_2_from_i = self.get_level2_common_neighbor_list(nodes[i], nodes[j])
                level_2_from_j = self.get_level2_common_neighbor_list(nodes[j], nodes[i])
                level_2 = self.commoncodes.unique(self.commoncodes.union(level_2_from_i, level_2_from_j))
                level_1_with_level_2 = self.commoncodes.unique(self.commoncodes.union(level_1, level_2))
                
                for k in level_1_with_level_2 :
                    value = value + 1 / nodes_degree[k]
                result[key] = value

        return result

    def calculate_Adamic_Adar_level_2_number(self):
        nodes = list(self.graph.nodes())
        nodes_degree = self.commoncodes.Convert_Degree_Tuple_To_Dictionary(self.graph.degree())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                value = 0 
                level_2_from_i = self.get_level2_common_neighbor_list(nodes[i], nodes[j])
                level_2_from_j = self.get_level2_common_neighbor_list(nodes[j], nodes[i])
                level_2 = self.commoncodes.unique(self.commoncodes.union(level_2_from_i, level_2_from_j))

                for k in level_2 :
                    value = value + 1 / math.log10(max(0.000001, nodes_degree[k] ))
                result[key] = value

        return result

    def calculate_Adamic_Adar_level_1_with_level_2_number(self):
        nodes = list(self.graph.nodes())
        nodes_degree = self.commoncodes.Convert_Degree_Tuple_To_Dictionary(self.graph.degree())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                value = 0
                level_1 = list(nx.common_neighbors(self.graph, nodes[i], nodes[j]))
                level_2_from_i = self.get_level2_common_neighbor_list(nodes[i], nodes[j])
                level_2_from_j = self.get_level2_common_neighbor_list(nodes[j], nodes[i])
                level_2 = self.commoncodes.unique(self.commoncodes.union(level_2_from_i, level_2_from_j))
                level_1_with_level_2 = self.commoncodes.unique(self.commoncodes.union(level_1, level_2))

                for k in level_1_with_level_2 :
                    value = value + 1 / math.log10( max(0.000001, nodes_degree[k] ))
                result[key] = value

        return result
    
    def calculate_Clustering_Coefficient_based_Link_Prediction_level_2_number(self):
        nodes = list(self.graph.nodes())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                value = 0 
                level_2_from_i = self.get_level2_common_neighbor_list(nodes[i], nodes[j])
                level_2_from_j = self.get_level2_common_neighbor_list(nodes[j], nodes[i])
                level_2 = self.commoncodes.unique(self.commoncodes.union(level_2_from_i, level_2_from_j))

                for k in level_2 :
                    value = value + self.clustering_coefficient[k]
                result[key] = value

        return result
    
    def calculate_Clustering_Coefficient_based_Link_Prediction_level_1_with_level_2_number(self):
        nodes = list(self.graph.nodes())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                value = 0
                level_1 = list(nx.common_neighbors(self.graph, nodes[i], nodes[j]))
                level_2_from_i = self.get_level2_common_neighbor_list(nodes[i], nodes[j])
                level_2_from_j = self.get_level2_common_neighbor_list(nodes[j], nodes[i])
                level_2 = self.commoncodes.unique(self.commoncodes.union(level_2_from_i, level_2_from_j))
                level_1_with_level_2 = self.commoncodes.unique(self.commoncodes.union(level_1, level_2))

                for k in level_1_with_level_2 :
                    value = value + self.clustering_coefficient[k]
                result[key] = value

        return result
      
    def calculate_Local_Naive_Bayes_based_Common_Neighbor_level_2_number(self):
        nodes = list(self.graph.nodes())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                
                value = 0 
                level_2_from_i = self.get_level2_common_neighbor_list(nodes[i], nodes[j])
                level_2_from_j = self.get_level2_common_neighbor_list(nodes[j], nodes[i])
                level_2 = self.commoncodes.unique(self.commoncodes.union(level_2_from_i, level_2_from_j))

                for k in level_2 :
                    value = value + math.log10( max( 0.000001,(self.clustering_coefficient[k]) / (1.0000001 - self.clustering_coefficient[k]))) +  self.LNBCN_log
                result[key] = value

        return result
    
    def calculate_Local_Naive_Bayes_based_Common_Neighbor_level_1_with_level_2_number(self):
        nodes = list(self.graph.nodes())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                value = 0
                level_1 = list(nx.common_neighbors(self.graph, nodes[i], nodes[j]))
                level_2_from_i = self.get_level2_common_neighbor_list(nodes[i], nodes[j])
                level_2_from_j = self.get_level2_common_neighbor_list(nodes[j], nodes[i])
                level_2 = self.commoncodes.unique(self.commoncodes.union(level_2_from_i, level_2_from_j))
                level_1_with_level_2 = self.commoncodes.unique(self.commoncodes.union(level_1, level_2))

                for k in level_1_with_level_2 :
                    value = value + math.log10( max(0.000001, (self.clustering_coefficient[k]) / (1.0000001 - self.clustering_coefficient[k]))) +  self.LNBCN_log
                result[key] = value

        return result
    
    def calculate_Node_and_Link_Clustering_Coefficient_level_2_number(self):
        nodes = list(self.graph.nodes())
        nodes_degree = self.commoncodes.Convert_Degree_Tuple_To_Dictionary(self.graph.degree())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                
                value = 0 
                level_2_from_i = self.get_level2_common_neighbor_list(nodes[i], nodes[j])
                level_2_from_j = self.get_level2_common_neighbor_list(nodes[j], nodes[i])
                level_2 = self.commoncodes.unique(self.commoncodes.union(level_2_from_i, level_2_from_j))

                for k in level_2 :
                    a_and_c_common_neighbor = list(nx.common_neighbors(self.graph, nodes[i], k))
                    b_and_c_common_neighbor = list(nx.common_neighbors(self.graph, nodes[j], k))
                    value = value + ((self.clustering_coefficient[k]) / (nodes_degree[k] - 1)) * (len(a_and_c_common_neighbor) + len(b_and_c_common_neighbor))
                result[key] = value

        return result

    def calculate_Node_and_Link_Clustering_Coefficient_level_1_with_level_2_number(self):
        nodes = list(self.graph.nodes())
        nodes_degree = self.commoncodes.Convert_Degree_Tuple_To_Dictionary(self.graph.degree())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                value = 0
                level_1 = list(nx.common_neighbors(self.graph, nodes[i], nodes[j]))
                level_2_from_i = self.get_level2_common_neighbor_list(nodes[i], nodes[j])
                level_2_from_j = self.get_level2_common_neighbor_list(nodes[j], nodes[i])
                level_2 = self.commoncodes.unique(self.commoncodes.union(level_2_from_i, level_2_from_j))
                level_1_with_level_2 = self.commoncodes.unique(self.commoncodes.union(level_1, level_2))

                for k in level_1_with_level_2 :
                    a_and_c_common_neighbor = list(nx.common_neighbors(self.graph, nodes[i], k))
                    b_and_c_common_neighbor = list(nx.common_neighbors(self.graph, nodes[j], k))
                    value = value + ((self.clustering_coefficient[k]) / (nodes_degree[k] - 1)) * (len(a_and_c_common_neighbor) + len(b_and_c_common_neighbor))
                result[key] = value

        return result
           
    def calculate_jaccard_level_2_number(self):
        nodes = list(self.graph.nodes())
        nodes_cnt = len(nodes)
        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                i_j_neghibors = self.commoncodes.union(list(self.graph.neighbors(nodes[i])), list(self.graph.neighbors(nodes[j])))
                level_1 = list(nx.common_neighbors(self.graph, nodes[i], nodes[j]))
                
                level_2_union_neighbors = []
                for k in i_j_neghibors:
                    level_2_union_neighbors = self.commoncodes.union(level_2_union_neighbors, list(self.graph.neighbors(k)))
                
                level_2_from_i = self.get_level2_common_neighbor_list(nodes[i], nodes[j])
                level_2_from_j = self.get_level2_common_neighbor_list(nodes[j], nodes[i])
                level_2_common_neighbors = self.commoncodes.unique(self.commoncodes.union(level_2_from_i, level_2_from_j))
                
                common_neighbors = len(level_2_common_neighbors)
                union_neighbors = len(level_2_union_neighbors)
                
                value = common_neighbors / union_neighbors
                result[key] = value

        return result
        
    def calculate_jaccard_level_1_with_level_2_number(self):
        nodes = list(self.graph.nodes())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                i_j_neghibors = self.commoncodes.union(list(self.graph.neighbors(nodes[i])), list(self.graph.neighbors(nodes[j])))
                level_1 = list(nx.common_neighbors(self.graph, nodes[i], nodes[j]))
                
                level_2_union_neighbors = []
                for k in i_j_neghibors:
                    level_2_union_neighbors = self.commoncodes.union(level_2_union_neighbors, list(self.graph.neighbors(k)))
                
                level_2_from_i = self.get_level2_common_neighbor_list(nodes[i], nodes[j])
                level_2_from_j = self.get_level2_common_neighbor_list(nodes[j], nodes[i])
                level_2_common_neighbors = self.commoncodes.unique(self.commoncodes.union(level_2_from_i, level_2_from_j))
                
                common_neighbors_level_2 = len(level_2_common_neighbors)
                union_neighbors_level_2  = len(level_2_union_neighbors)
                
                key = (nodes[i], nodes[j])
                common_neighbors = len(list(nx.common_neighbors(self.graph, nodes[i], nodes[j])))
                union_neighbors = len(self.commoncodes.union(list(self.graph.neighbors(nodes[i])), list(self.graph.neighbors(nodes[j]))))
                value = (common_neighbors / union_neighbors) + (common_neighbors_level_2 / union_neighbors_level_2)
                result[key] = value

        return result
    
    def calculate_CAR_level_2_number(self):
        nodes = list(self.graph.nodes())
        nodes_degree = self.commoncodes.Convert_Degree_Tuple_To_Dictionary(self.graph.degree())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                value_from_i = 0
                value_from_j = 0

                neighbors = list(nx.common_neighbors(self.graph, nodes[i], nodes[j]))
                i_neghibors = list(nx.neighbors(self.graph, nodes[i]))
                j_neghibors = list(nx.neighbors(self.graph, nodes[j]))
                
                level_2_from_i = self.get_level2_common_neighbor_list(nodes[i], nodes[j])
                level_2_from_j = self.get_level2_common_neighbor_list(nodes[j], nodes[i])

                cn_neighbors = []
                for k in neighbors :
                    cn_neighbors = self.commoncodes.union(cn_neighbors, list(nx.neighbors(self.graph, k)))
                    
                for k in level_2_from_i :
                    c_neighbors = list(nx.neighbors(self.graph, k))
                    c_common_neighbor_with_a_and_b = self.commoncodes.intersection(i_neghibors, c_neighbors)
                    c_common_neighbor_with_a_and_b = self.commoncodes.intersection(c_common_neighbor_with_a_and_b, cn_neighbors)
                    value_from_i = value_from_i + len(c_common_neighbor_with_a_and_b) / 2

                for k in level_2_from_j :
                    c_neighbors = list(nx.neighbors(self.graph, k))
                    c_common_neighbor_with_a_and_b = self.commoncodes.intersection(j_neghibors, c_neighbors)
                    c_common_neighbor_with_a_and_b = self.commoncodes.intersection(c_common_neighbor_with_a_and_b, cn_neighbors)
                    value_from_j = value_from_j + len(c_common_neighbor_with_a_and_b) / 2
                    
                result[key] = value_from_i * len(level_2_from_i) + value_from_j * len(level_2_from_j)

        return result
               
    def calculate_CAR_level_1_with_level_2_number(self):
        nodes = list(self.graph.nodes())
        nodes_degree = self.commoncodes.Convert_Degree_Tuple_To_Dictionary(self.graph.degree())
        nodes_cnt = len(nodes)

        result = {}
        
        for i in range (nodes_cnt):
            for j in range (i + 1, nodes_cnt):
                key = (nodes[i], nodes[j])
                value = 0
                value_from_i = 0
                value_from_j = 0
                
                neighbors = list(nx.common_neighbors(self.graph, nodes[i], nodes[j]))
                i_neghibors = list(nx.neighbors(self.graph, nodes[i]))
                j_neghibors = list(nx.neighbors(self.graph, nodes[j]))
                
                level_2_from_i = self.get_level2_common_neighbor_list(nodes[i], nodes[j])
                level_2_from_j = self.get_level2_common_neighbor_list(nodes[j], nodes[i])
                    
                cn_neighbors = []
                for k in neighbors :
                    cn_neighbors = self.commoncodes.union(cn_neighbors, list(nx.neighbors(self.graph, k)))
                    
                for k in neighbors :
                    c_neighbors = list(nx.neighbors(self.graph, k))
                    c_common_neighbor_with_a_and_b = self.commoncodes.intersection(neighbors, c_neighbors)
                    value = value + len(c_common_neighbor_with_a_and_b) / 2
                
                for k in level_2_from_i :
                    c_neighbors = list(nx.neighbors(self.graph, k))
                    c_common_neighbor_with_a_and_b = self.commoncodes.intersection(i_neghibors, c_neighbors)
                    c_common_neighbor_with_a_and_b = self.commoncodes.intersection(c_common_neighbor_with_a_and_b, cn_neighbors)
                    value_from_i = value_from_i + len(c_common_neighbor_with_a_and_b) / 2

                for k in level_2_from_j :
                    c_neighbors = list(nx.neighbors(self.graph, k))
                    c_common_neighbor_with_a_and_b = self.commoncodes.intersection(j_neghibors, c_neighbors)
                    c_common_neighbor_with_a_and_b = self.commoncodes.intersection(c_common_neighbor_with_a_and_b, cn_neighbors)
                    value_from_j = value_from_j + len(c_common_neighbor_with_a_and_b) / 2
                    
                result[key] = value * len(neighbors) + value_from_i * len(level_2_from_i) + value_from_j * len(level_2_from_j)

        return result
        
