'''
Run the graph embedding methods on Karate graph and evaluate them on 
graph reconstruction and visualization. Please copy the 
gem/data/karate.edgelist to the working directory
'''
import matplotlib.pyplot as plt
from time import time

from gem.utils      import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from gem.evaluation import evaluate_graph_reconstruction as gr

from gem.embedding.gf       import GraphFactorization
from gem.embedding.hope     import HOPE
from gem.embedding.lap      import LaplacianEigenmaps
from gem.embedding.lle      import LocallyLinearEmbedding
from gem.embedding.node2vec import node2vec
import networkx as nx
from gem.embedding.teammate     import Teammate
from argparse import ArgumentParser


if __name__ == '__main__':
    ''' Sample usage
    python run_karate.py -node2vec 1
    '''
    parser = ArgumentParser(description='Graph Embedding Experiments on Roller Derby graphs')
    parser.add_argument('-node2vec', '--node2vec',
                        help='whether to run node2vec (default: False)')
    args = vars(parser.parse_args())
    try:
        run_n2v = bool(int(args["node2vec"]))
    except:
        run_n2v = False

    # File that contains the edges. Format: source target
    # Optionally, you can add weights as third column: source target weight
    edge_tot = '../../Data/AllTeamsFullLTGraph.edgelist'
    edge_train = '../../Data/AllTeamsLTGraphTrainNormalized.edgelist'
    edge_test = '../../Data/AllTeamsLTGraphTestNormalized.edgelist'
    edge_val = '../../Data/AllTeamsLTGraphValNormalized.edgelist'
    # Specify whether the edges are directed
    isDirected = True

    # Load graph. Have to prune manually to keep number of nodes fixed
    G = nx.read_weighted_edgelist(edge_tot, nodetype=int)
    G_test_dummy = nx.read_weighted_edgelist(edge_test, nodetype=int)
    G_train_dummy = nx.read_weighted_edgelist(edge_train, nodetype=int)
    G_val_dummy = nx.read_weighted_edgelist(edge_val, nodetype=int)

    G = G.to_directed()

    G_train = G.copy()
    G_val = G.copy()
    G_test = G.copy()

    for edge in G.edges():
        if edge not in G_train_dummy.edges(): G_train.remove_edge(*edge)
        if edge not in G_test_dummy.edges(): G_test.remove_edge(*edge)
        if edge not in G_val_dummy.edges(): G_val.remove_edge(*edge)

    print(len(G_train))
    print(len(G_test))
    print(len(G_val))

    print(G.number_of_edges())
    print(G_train.number_of_edges())
    print(G_val.number_of_edges())
    print(G_test.number_of_edges())
    print(G_train_dummy.nodes)

    models = []
    # Load the models you want to run
    # models.append(GraphFactorization(d=2, max_iter=50000, eta=1 * 10**-4, regu=1.0, data_set='derby'))
    #models.append(HOPE(d=4, beta=0.01))
    #models.append(LaplacianEigenmaps(d=2))
    #models.append(LocallyLinearEmbedding(d=2))
    if run_n2v:
        models.append(
            node2vec(d=2, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1)
        )
    #alpha = 0 to have "traditional" second order loss
    models.append(Teammate(d=2, beta=5, alpha=0, nu1=1e-6, nu2=1e-6, K=2,n_units=[500, 300], rho=0.3, n_iter=200, xeta=0.01, n_batch=50,
                    modelfile=['enc_model.json', 'dec_model.json'],
                    weightfile=['enc_weights.hdf5', 'dec_weights.hdf5']))

    # For each model, learn the embedding and evaluate on graph reconstruction and visualization
    for embedding in models:
        print ('Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges()))
        t1 = time()
        # Learn embedding - accepts a networkx graph or file with edge list
        Y, t = embedding.learn_embedding(graph=G_train,valgraph=G_val,edge_f=None, is_weighted=True, no_python=True)
        print (embedding._method_name+':\n\tTraining time: %f' % (time() - t1))

        # Evaluate on graph reconstruction:train
        MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(G_train, embedding, Y, None, is_weighted=True, is_undirected=False)
        print("MSE train is ",pow(err,2)/G_train.number_of_edges())
        #print(("\tMAP: {} \t precision curve: {}\n\n\n\n"+'-'*100).format(MAP,prec_curv[:5]))
        viz.plot_embedding2D(embedding.get_embedding(), di_graph=G_train, node_colors=None)
        plt.show()
        plt.clf()

        # Evaluate on graph reconstruction:val
        MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(G_val, embedding, Y, None, is_weighted=True, is_undirected=False)
        print("MSE val is ",pow(err,2)/G_val.number_of_edges())
        #print(("\tMAP: {} \t precision curve: {}\n\n\n\n"+'-'*100).format(MAP,prec_curv[:5]))
        viz.plot_embedding2D(embedding.get_embedding(), di_graph=G_val, node_colors=None)
        plt.show()
        plt.clf()

"""
        # Evaluate on graph reconstruction:val
        MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(G_test, embedding, Y, None, is_weighted=True, is_undirected=False)
        print(("\tMAP: {} \t precision curve: {}\n\n\n\n"+'-'*100).format(MAP,prec_curv[:5]))
        viz.plot_embedding2D(embedding.get_embedding(), di_graph=G_test, node_colors=None)
        plt.show()
        plt.clf()

"""



