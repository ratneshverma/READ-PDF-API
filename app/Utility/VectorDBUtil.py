from langchain_community.vectorstores import FAISS
from typing import List, Optional
import numpy as np

def RemoveVectors(vectorstore: FAISS, docstore_ids: Optional[List[str]]):
    """
    Function to remove documents from the vectorstore.
    
    Parameters
    ----------
    vectorstore : FAISS
        The vectorstore to remove documents from.
    docstore_ids : Optional[List[str]]
        The list of docstore ids to remove. If None, all documents are removed.
    
    Returns
    -------
    n_removed : int
        The number of documents removed.
    n_total : int
        The total number of documents in the vectorstore.
    
    Raises
    ------
    ValueError
        If there are duplicate ids in the list of ids to remove.
    """
    if docstore_ids is None:
        vectorstore.docstore = {}
        vectorstore.index_to_docstore_id = {}
        n_removed = vectorstore.index.ntotal
        n_total = vectorstore.index.ntotal
        vectorstore.index.reset()
        return n_removed, n_total
    set_ids = set(docstore_ids)
    if len(set_ids) != len(docstore_ids):
        raise ValueError("Duplicate ids in list of ids to remove.")
    index_ids = [
        i_id
        for i_id, d_id in vectorstore.index_to_docstore_id.items()
        if d_id in docstore_ids
    ]
    n_removed = len(index_ids)
    n_total = vectorstore.index.ntotal
    vectorstore.index.remove_ids(np.array(index_ids, dtype=np.int64))
    for i_id, d_id in zip(index_ids, docstore_ids):
        del vectorstore.docstore._dict[
            d_id
        ]  # remove the document from the docstore

        del vectorstore.index_to_docstore_id[
            i_id
        ]  # remove the index to docstore id mapping
    vectorstore.index_to_docstore_id = {
        i: d_id
        for i, d_id in enumerate(vectorstore.index_to_docstore_id.values())
    }
    return n_removed, n_total