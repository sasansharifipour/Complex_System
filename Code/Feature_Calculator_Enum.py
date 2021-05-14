from enum import Enum

class Feature_Calculator_Enum(Enum):
    Common_Neighbor = 1
    Preferential_Attachment = 2
    Resource_Allocation = 3
    Adamic_Adar = 4
    Jaccard = 5
    Clustering_Coefficient = 6
    Local_Naive_Bayes_based_Common_Neighbor = 7
    Node_and_Link_Clustering_Coefficient = 8
    CAR = 9

class Feature_Level_Enum(Enum):
    Level_1 = 1
    Level_2 = 2
    Level_1_And_2 = 3
