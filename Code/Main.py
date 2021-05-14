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

#graph = CC.create_paper_graph()
#graph_train = graph

CC = Common_Codes()
ds_loader = data_set_loader()

n_for_auc = 1000
remove_edges_fraction = 0.2
dataset = [Datasets.Macaque, Datasets.Football, Datasets.Celegansneural, Datasets.USAir97, Datasets.Polblogs, Datasets.Yeast, Datasets.Amazon]

for ds in dataset:
    file_name = 'result_' + str(ds) + '_' + str(remove_edges_fraction) + '.csv'
    
    f = open(file_name, "a")
    writer = csv.writer(f, delimiter=',')

    data = [['idx', 'Dataset', 'Feature Calculator', 'Feature Level', 'Number Of Removed Links', 'Number Of Correct Suggestions', 'AUC', 'Number Of Correct Suggestions By OCC', 'AUC By OCC', 'AUC For Link Prediction']] 
    writer.writerows(data)
    f.flush()
    
    selected_graph = ds
    graph = ds_loader.load_dataset(selected_graph)

    for i in range(2):
        graph_train, removed_edges = ds_loader.prepare_dataset(graph, remove_edges_fraction)

        number_of_edge_should_suggest = len(graph.edges()) - len(graph_train.edges())

        FC = Feature_Calculator(graph_train)

        for fc in Feature_Calculator_Enum:
            for fl in Feature_Level_Enum:
                try:
                    feature_calculator = fc
                    feature_level = fl

                    ranks = FC.Calculate_Feature(feature_calculator, feature_level)
                    ranks_remove_exist_links, ranks_exist_links = FC.get_ranks_exist_links(ranks)
                    absent_link_ranks, removed_link_ranks = FC.split_ranks_to_absent_removed_link_ranks(ranks_remove_exist_links, removed_edges)

                    mdl = FC.train_one_class_classifier(ranks_exist_links)
                    ranks_by_OCC = FC.predict_by_one_class_classifier(mdl, ranks_remove_exist_links) 
                    
                    suggested_items = FC.get_suggested_edges(ranks_remove_exist_links, number_of_edge_should_suggest)                    
                    results = CC.correct_selected_edges(graph, suggested_items)
                    auc = CC.calculate_auc(results)
                    auc_for_link_prediction = CC.calculate_auc_for_link_prediction(n_for_auc, absent_link_ranks, removed_link_ranks)
                    
                    suggested_items_by_OCC = FC.get_suggested_edges(ranks_by_OCC, number_of_edge_should_suggest)
                    results_by_OCC = CC.correct_selected_edges(graph, suggested_items_by_OCC)
                    auc_by_OCC = CC.calculate_auc(results_by_OCC)
                    
                    data = [[i, ds, fc, fl, len(suggested_items), sum(results), auc, sum(results_by_OCC), auc_by_OCC, auc_for_link_prediction]] 
                    writer.writerows(data)

                except:
                    data = [[i, ds, fc, fl, '', '', 'exception', '', '', '']]
                    writer.writerows(data)

                f.flush()
    f.close()
