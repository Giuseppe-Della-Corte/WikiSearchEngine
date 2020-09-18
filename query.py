''' This file handles the query part of the search engine '''

import numpy
from numpy import zeros
from numpy import dot
from numpy.linalg import norm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import re
import math
import sys

def filesToDicts():
    ind = {}
    mapp = {}
    pos = {}
    
    with open('index.txt', 'r') as ind_file, open("mapping.txt", 'r') as mapp_file, open("positions.txt", "r") as pos_file:
        ind_lines = [[line.rstrip('\n')] for line in ind_file]
        ind_lines = [line[0] for line in ind_lines]
        mapp_lines = [[line.rstrip('\n')] for line in mapp_file]
        mapp_lines = [line[0].split(' ') for line in mapp_lines]
        pos_lines = [[line.rstrip('\n')] for line in pos_file]
        pos_lines = [line[0].split(' ') for line in pos_lines]
    
    # Matrix
    mat = numpy.loadtxt('vectorSpace.txt')

    # Index
    for line in ind_lines:
        split = line.split(' ')
        word, postings = split[0], [i for i in split[1:]]
        ind[word] = postings

    # Mapping
    for line in mapp_lines:
        mapp[line[0]] = line[1]
    
    # Positions
    for line in pos_lines:
        pos[line[0]] = line[1]
    
    return mat, ind, mapp, pos

# Query sanitizer (stopwords, lowercase, stemming, tokenizing)
def query_sanity(query, language):
    stemmer = PorterStemmer()
    query = re.sub(r'[^\w\s]', '', str(query))
    tokens = nltk.word_tokenize(query.lower())
    if len(tokens) >= 1:
        filtered = [w for w in tokens if not w in stopwords.words(language)]
        sane_query= [stemmer.stem(word) for word in filtered]
    else:
        sane_query = []
    
    return sane_query

# Query vector generator
def query_vector(query, pos, ind):
  if query == []:
    return "the query search you were looking for is a stop word, which is not supported by our search engine"
  else:
    vect = zeros(len(ind.keys()))
    for word in query:
        if pos.get(word):
            vect[int(pos[word])] = 1
  return vect

# Posting list retrieval for standard union query
def standardPostings(query, ind):
    union = []
    for word in query:
        if word in ind.keys():
            union += [docId for docId in index[word] ]
    return sorted(list(set(union)))

# Posting list retrieval for intersected query
def manyIntersect(query, ind):
    # Sorting by length of postings list
    try:
        query.sort(key = lambda l: len(index[l]))
    except KeyError:
        print('At least one of the terms doesn\'t exist in our corpus.')
        exit(0)

    result = index[query[0]]
    del query[0]
    
    while query and result:
        # comp, result = skipIntersect(index, result, term_list[0])
        result = skipIntersect(index, result, query[0])
        del query[0]

    # Return the final intersection 
    return result


def skipIntersect(index, post1, term2):
    # We sort the lists when using lower_index.txt
    post1.sort()
    post2 = index[term2]
    post2.sort()
    result = []
    # Initialize the skip lists
    skip_1 = skipList(post1)
    skip_2 = skipList(post2)
    i, j = 0, 0
    # starting of the algorithm 
    while i < len(post1) and j < len(post2):
        if post1[i] == post2[j]:
            result.append(post1[i])
            i += 1
            j += 1
        elif post1[i] < post2[j]:
            if skip_1 and skip_1[0] <= post2[j]:
                while skip_1 and skip_1[0] <= post2[j]:
                    i = post1.index(skip_1[0])
                    del skip_1[0]
            else:
                i += 1
        else:
            if skip_2 and skip_2[0] <= post1[i]:
                while skip_2 and skip_2[0] <= post1[i]:
                    j = post2.index(skip_2[0])
                    del skip_2[0]
            else:
                j += 1
    return result

def skipList(post):
    gap = round(math.sqrt(len(post)))
    # Returns the skip list composed of evenly separated numbers from the postings list
    return [post[i] for i in range(1, len(post), gap)]


# Cosine similarity computation for ranking
def cosine(q_vector, mat, mapp, postings):
    relevance = {}
    for postID in postings:
        row = int(mapp[postID])
        docVector = mat[row]

        cosine = float(dot(q_vector, docVector) / (norm(q_vector) * norm(docVector)))
        relevance[postID] = cosine

    ranking = sorted(relevance, key=relevance.get, reverse=True)
    return ranking
    
def url_gen(ranking, threshold, outputfile):
    base = 'https://en.wikipedia.org/wiki?curid='
    with open(outputfile, 'w') as f:
        try:
            for i in range(0, threshold):
                url = base + ranking[i]
                f.write(url + '\n')
                print('Rank', str(i+1) + ':', url)
        except IndexError:
            f.write('No other relevant documents found.')



if __name__ == '__main__':

    if len(sys.argv) <= 1:
        print('Wrong number of arguments. You must run the program with the following command:\n python query.py <query text>')
        sys.exit(0)

    matrix, index, mapping, positions = filesToDicts()
    q = ' '.join(sys.argv[1:])
    sane_q = query_sanity(q, 'english')
    vector = query_vector(sane_q, positions, index)

    union_list = standardPostings(sane_q, index)
    intersect_list = manyIntersect(sane_q, index)

    union_rank = cosine(vector, matrix, mapping, union_list)
    intersect_rank = cosine(vector, matrix, mapping, intersect_list)

    if union_rank:
        print('Ranking of the TOP 10 relevant documents containing at least one of the query terms.')
        url_gen(union_rank, 10, 'union_results.txt')
        print('DONE.')
    else:
        print('No documents were found in the union ranking.')

    if intersect_rank:
        print('Ranking of TOP 10 relevant documents containing all query terms')
        url_gen(intersect_rank, 10, 'intersect_results.txt')
        print('DONE.')
    else:
        print('No documents were found in the intersect ranking.')




