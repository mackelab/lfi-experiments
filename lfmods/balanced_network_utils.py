def get_cluster_connection_probs(REE, k, p_ee):
    """
    When using clustered connectivity in the E population the avergae sparseness should still be p_ee.
    For a given clustering coef REE this method calculates the sparseness within, p_in, and between, p_out, clusters.
    For the uniform case, REE=1, p_ee = p_in = p_out.
    :param REE:
    :param k:
    :param p_ee:
    :return:
    """
    p_out = p_ee * k / (REE + k - 1)
    p_in = REE * p_out
    return p_in, p_out