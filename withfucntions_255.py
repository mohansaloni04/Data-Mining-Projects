# coding: utf-8

# In[67]:
import gzip

import pandas as pd


def impfunction():
    import pandas as pd
    import numpy as np
    import gzip
    import scipy.sparse as sp
    from numpy.linalg import norm
    from collections import Counter, defaultdict
    from scipy.sparse import csr_matrix
    from pyparsing import anyOpenTag, anyCloseTag
    from xml.sax.saxutils import unescape as unescape
    from bs4 import BeautifulSoup
    from bs4 import SoupStrainer
    from collections import Counter
    from scipy.sparse import csr_matrix


# In[ ]:

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


# In[68]:

def html_removal(doc):
    from pyparsing import anyOpenTag, anyCloseTag
    from xml.sax.saxutils import unescape as unescape
    unescape_xml_entities = lambda s: unescape(s, {"&apos;": "'", "&quot;": '"', "&nbsp;": " "})
    stripper = (anyOpenTag | anyCloseTag).suppress()
    review = unescape_xml_entities(stripper.transformString(doc))
    return review


# In[69]:

def special_removal(doc):
    import re
    review = re.sub('[^a-zA-Z0-9]', ' ', doc)
    return review


# In[70]:

def lower_split(doc):
    doc_lower = doc.lower()
    doc_split = doc_lower.split()
    return doc_split


# In[71]:

def filterLen(doc, minlen):
    r""" filter out terms that are too short. docs is a list of lists, each inner list is a document represented as a list of words minlen is the minimum length of the word to keep """
    return [t for t in doc if len(t) >= minlen]


# In[96]:

def build_matrix(docs):
    r""" Build sparse matrix from a list of documents,
    each of which is a list of word/terms in the document.
    """
    import numpy as np
    import pandas as pd
    import numpy as np
    import gzip
    import scipy.sparse as sp
    from numpy.linalg import norm
    from collections import Counter, defaultdict
    from scipy.sparse import csr_matrix
    from pyparsing import anyOpenTag, anyCloseTag
    from xml.sax.saxutils import unescape as unescape
    from bs4 import BeautifulSoup
    from bs4 import SoupStrainer
    from collections import Counter
    from scipy.sparse import csr_matrix
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    for d in docs:
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)

    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows + 1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        cnt = Counter(d)
        keys = list(k for k, _ in cnt.most_common())
        l = len(keys)
        for j, k in enumerate(keys):
            ind[j + n] = idx[k]
            val[j + n] = cnt[k]
        ptr[i + 1] = ptr[i] + l
        n += l
        i += 1

    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()

    return mat


# In[97]:

def csr_info(mat, name="", non_empy=False):
    import numpy as np
    import pandas as pd
    import numpy as np
    import gzip
    import scipy.sparse as sp
    from numpy.linalg import norm
    from collections import Counter, defaultdict
    from scipy.sparse import csr_matrix
    from pyparsing import anyOpenTag, anyCloseTag
    from xml.sax.saxutils import unescape as unescape
    from bs4 import BeautifulSoup
    from bs4 import SoupStrainer
    from collections import Counter
    from scipy.sparse import csr_matrix
    r""" Print out info about this CSR matrix. If non_empy, 
    report number of non-empty rows and cols as well
    """
    import numpy as np
    if non_empy:
        print("%s [nrows %d (%d non-empty), ncols %d (%d non-empty), nnz %d]" % (
            name, mat.shape[0],
            sum(1 if mat.indptr[i + 1] > mat.indptr[i] else 0
                for i in range(mat.shape[0])),
            mat.shape[1], len(np.unique(mat.indices)),
            len(mat.data)))
    else:
        print("%s [nrows %d, ncols %d, nnz %d]" % (name,
                                                   mat.shape[0], mat.shape[1], len(mat.data)))


# In[98]:

def csr_idf(mat, copy=False, **kargs):
    import numpy as np
    import pandas as pd
    import numpy as np
    import gzip
    import scipy.sparse as sp
    from numpy.linalg import norm
    from collections import Counter, defaultdict
    from scipy.sparse import csr_matrix
    from pyparsing import anyOpenTag, anyCloseTag
    from xml.sax.saxutils import unescape as unescape
    from bs4 import BeautifulSoup
    from bs4 import SoupStrainer
    from collections import Counter
    from scipy.sparse import csr_matrix
    r""" Scale a CSR matrix by idf. 
    Returns scaling factors as dict. If copy is True, 
    returns scaled matrix and scaling factors.
    """
    import numpy as np
    import pandas as pd
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # document frequency
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    # inverse document frequency
    for k, v in df.items():
        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
    # scale by idf
    for i in range(0, nnz):
        val[i] *= df[ind[i]]

    return df if copy is False else mat


# In[99]:

def csr_l2normalize(mat, copy=False, **kargs):
    import numpy as np
    import pandas as pd
    import numpy as np
    import gzip
    import scipy.sparse as sp
    from numpy.linalg import norm
    from collections import Counter, defaultdict
    from scipy.sparse import csr_matrix
    from pyparsing import anyOpenTag, anyCloseTag
    from xml.sax.saxutils import unescape as unescape
    from bs4 import BeautifulSoup
    from bs4 import SoupStrainer
    from collections import Counter
    from scipy.sparse import csr_matrix
    r""" Normalize the rows of a CSR matrix by their L-2 norm. 
    If copy is True, returns a copy of the normalized matrix.
    """
    import numpy as np
    import pandas as pd
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0
        for j in range(ptr[i], ptr[i + 1]):
            rsum += val[j] ** 2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0 / np.sqrt(rsum)
        for j in range(ptr[i], ptr[i + 1]):
            val[j] *= rsum

    if copy is True:
        return mat


# In[105]:

def training_dict():
    with open("/Users/Student/PycharmProjects/myproject/src/data/filtered_file.dat", "r") as fr:
        f_reviews = fr.readlines()
    print len(f_reviews)
    import re
    for i in range(0, 18534):
        f_reviews[i] = re.sub('[^a-zA-Z0-9]', ' ', f_reviews[i])
    docs1 = [l.split() for l in f_reviews]
    return docs1


# In[ ]:

def getclass():
    df = getDF('/Users/Student/PycharmProjects/myproject/src/data/reviews_Health_and_Personal_Care_5.json.gz')
    df.drop(['helpful', 'reviewerName', 'reviewerID', 'unixReviewTime', 'reviewTime', 'asin'], axis=1)
    df_final = df.groupby('asin').agg({'overall': 'mean',
                                       'reviewText': lambda x: ' , '.join(x),
                                       'summary': lambda x: ' , '.join(x)}).reset_index()
    train_rating = df_final['overall'].values.tolist()
    return train_rating


# In[106]:

def knnRegres(trainmat, testmat, traincls):
    from sklearn.neighbors import KNeighborsRegressor
    neigh = KNeighborsRegressor(n_neighbors=10, weights='distance', algorithm='auto', leaf_size=30, p=3,
                                metric='euclidean', metric_params=None, n_jobs=10)
    neigh.fit(trainmat, traincls)
    test_ratings_knn = neigh.predict(testmat)
    return test_ratings_knn


# In[107]:

# review = " this is me and i cannot function. This is not good. But I have a work around"
def startinghere(review1):
    impfunction()
    html_rem = html_removal(review1)
    print html_rem
    spe_rem = special_removal(html_rem)
    print spe_rem
    ls1 = lower_split(spe_rem)
    print ls1
    ureview_final = filterLen(ls1, 4)
    print ureview_final
    print len(ureview_final)
    print ureview_final[0]
    train_reviews = training_dict()
    print len(train_reviews)
    train_reviews.append(ureview_final)
    print len(train_reviews)
    mat1 = build_matrix(train_reviews)
    csr_info(mat1)
    mat2 = csr_idf(mat1, copy=True)
    mat3 = csr_l2normalize(mat2, copy=True)
    print mat3.shape[0]
    train_mat = mat3[:18534, :]
    print train_mat.shape[0]
    test_mat = mat3[18534:, :]
    print test_mat.shape[0]
    train_rating = getclass()
    final_rate = knnRegres(train_mat, test_mat, train_rating)
    return final_rate

# In[108]:

# startinghere(review)


# In[ ]:


# In[ ]:



