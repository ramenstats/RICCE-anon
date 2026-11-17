def hierarchical_precision(taxonomy,pred,gt): return taxonomy.hierarchical_scores(pred,gt)[0]
def hierarchical_recall(taxonomy,pred,gt): return taxonomy.hierarchical_scores(pred,gt)[1]
def hierarchical_f1(taxonomy,pred,gt): return taxonomy.hierarchical_scores(pred,gt)[2]
