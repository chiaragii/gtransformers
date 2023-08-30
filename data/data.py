"""
    File to load dataset based on user control from main file
"""
from data.BPIgraphs import GraphsDataset


def LoadData(DATASET_NAME, num_nodes):
    """
        This function is called in the main_xx.py file 
        returns:
        ; dataset object
    """
    # handling for (ZINC) molecule dataset
    if DATASET_NAME == 'BPI12_graph':
        return GraphsDataset(DATASET_NAME, num_nodes)
