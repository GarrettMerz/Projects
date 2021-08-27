import numpy as np

precision_pos = [2, 10, 100, 200, 300, 500, 1000]

def computePrecisionCurve(predicted_edge_list, true_digraph, max_k=-1):
    #print("pred")
    #print(predicted_edge_list)
    #print("true")
    #print(true_digraph.nodes)

    if max_k == -1:
        max_k = len(predicted_edge_list)
    else:
        max_k = min(max_k, len(predicted_edge_list))

    sorted_edges = sorted(predicted_edge_list, key=lambda x: x[2], reverse=True)
    precision_scores = []
    delta_factors = []
    correct_edge = 0
    for i in range(max_k):
        if true_digraph.has_edge(sorted_edges[i][0], sorted_edges[i][1]):
            correct_edge += 1
            delta_factors.append(1.0)
        else:
            delta_factors.append(0.0)
        precision_scores.append(1.0 * correct_edge / (i + 1))
    return precision_scores, delta_factors

def computeMAP(predicted_edge_list, true_digraph, max_k=-1):
    node_num = len(true_digraph.nodes)
    #print(true_digraph.nodes)
    #print(node_num)
    node_edges = []
    for i in range(node_num):
        node_edges.append([])
    for (st, ed, w) in predicted_edge_list:
        node_edges[st].append((st, ed, w))
    node_AP = [0.0] * node_num
    count = 0
    for i in range(node_num):
        if true_digraph.out_degree(i) == 0:
            continue
        count += 1
        precision_scores, delta_factors = computePrecisionCurve(node_edges[i], true_digraph, max_k)
        precision_rectified = [p * d for p,d in zip(precision_scores,delta_factors)]
        if(sum(delta_factors) == 0):
            node_AP[i] = 0
        else:
            node_AP[i] = float(sum(precision_rectified) / sum(delta_factors))
    return sum(node_AP) / count

def computeMANE(predicted_edge_list, true_digraph, max_k=-1):
    node_num = len(true_digraph.nodes)
    #print(true_digraph.nodes)
    #print(node_num)
    node_edges = []
    for i in range(node_num):
        node_edges.append([])
    for (st, ed, w) in predicted_edge_list:
        node_edges[st].append((st, ed, w))
    node_AP = [0.0] * node_num
    count = 0
    totalmane = 0
    for i in range(node_num):
        if true_digraph.out_degree(i) == 0:
            continue
        count += 1
        #get rankings of other nodes connected to i in true graph
        true_edges = sorted(true_digraph.out_edges(i,data=True), key=lambda t: t[2].get('weight', 1), reverse = True)
        true_edges_noweight =[(edge[0],edge[1]) for edge in true_edges]
        #get rankings of other nodes connected to i in pred graph
        sorted_pred_edges = sorted(node_edges[i], key=lambda x: x[2], reverse=True)
        sorted_pred_edges_noweight =[(edge[0],edge[1]) for edge in sorted_pred_edges]

        totalrankdiff = 0
        #loop over true edges
        for (edge,true_rank) in enum(true_edges):
            rankdiff = -999
            if edge not in sorted_pred_edges_noweight: continue
            else:
                pred_rank = sorted_pred_edges_noweight.index(edge)
            rankdiff = abs(true_rank - predrank)
            totalrankdiff += rankdiff
        mane_i = totalrankdiff / len(true_edges)
        totalmane += mane_i

    return totalmane/count

def computeAvgRecAtk(predicted_edge_list, true_digraph, max_k=-1):
    #print("pred")
    #print(predicted_edge_list)
    #print("true")
    #print(true_digraph.nodes)

    if max_k == -1:
        max_k = len(predicted_edge_list)
    else:
        max_k = min(max_k, len(predicted_edge_list))

    sorted_edges = sorted(predicted_edge_list, key=lambda x: x[2], reverse=True)
    k_weightsum_pred = 0
    for i in range(max_k):
        k_weightsum_pred += sorted_edges[i][2]

    true_edges = sorted(true_digraph.out_edges(i,data=True), key=lambda t: t[2].get('weight', 1), reverse = True)
    k_weightsum_true = 0
    for i in range(max_k):
        k_weightsum_true += true_edges[i][2]

    return k_weightsum_pred/max_k, k_weightsum_true/max_k

def getMetricsHeader():
    header = 'MAP\t' + '\t'.join(['P@%d' % p for p in precision_pos])
    header = header + '\tP@EdgeNum'
    return header

def getPrecisionReport(prec_curv, edge_num):
    result_str = ''
    temp_pos = precision_pos[:] + [edge_num]
    for p in temp_pos:
        if(p < len(prec_curv)):
            result_str += '\t%f' % prec_curv[p-1]
        else:
            result_str += '\t-'
    return result_str[1:]

# We define StabilityDeviation of nxd embeddings X1 and X2, 
# nxn adjacenecy matrices S1 and S2 as:
# StabDev = (||S1||_F * ||X2 - X1||_F) / (||X1||_F * ||S2 - S1||_F)
def getStabilityDev(X1, X2, S1, S2):
    n1,d = X1.shape
    return (np.linalg.norm(S1) * np.linalg.norm(X2[:n1,:]-X1)) / (np.linalg.norm(X1) * np.linalg.norm(S2[:n1,:n1]-S1))

def getEmbeddingShift(X1, X2, S1, S2):
    n1,d = X1.shape
    return (np.linalg.norm(X2[:n1,:]-X1))/(n1*d)

def getNodeAnomaly(X_dyn):
    T = len(X_dyn)
    n_nodes = X_dyn[0].shape[0]
    node_anom = np.zeros((n_nodes, T-1))
    for t in range(T-1):
        node_anom[:, t] = np.linalg.norm(X_dyn[t+1][:n_nodes, :] - X_dyn[t][:n_nodes, :], axis = 1)
    return node_anom
