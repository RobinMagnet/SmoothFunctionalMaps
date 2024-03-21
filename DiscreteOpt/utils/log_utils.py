def get_emb_size_str_base(emb_dim_K1=0, emb_dim_K2=0, emb_dim_p=0, emb_dim_bias=0):
    """
    Write the formula as function of K1, K2, p and the bias term:

    eg K1 + 3*K2 + 2*P + 1
    """

    emb_size_str = "Emb Size : "

    previous_terms = False

    if emb_dim_K1 > 0:
        emb_size_str += f"{f'{emb_dim_K1}*' if emb_dim_K1>1 else ''}K1"
        previous_terms = True
    
    if emb_dim_K2 > 0:
        


        emb_size_str += f"{' + ' if previous_terms else ''}{f'{emb_dim_K2}*' if emb_dim_K2>1 else ''}K2"
        previous_terms = True
    
    if emb_dim_p > 0:
        emb_size_str += f"{' + ' if previous_terms else ''}{f'{emb_dim_p}*' if emb_dim_p>1 else ''}P"
        previous_terms = True
    
    if emb_dim_bias > 0:
        emb_size_str += f"{' + ' if previous_terms else ''}{emb_dim_bias}"
        previous_terms = True
    
    if not previous_terms:
        emb_size_str += "0"
    
    return emb_size_str
        
