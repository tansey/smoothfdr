import numpy as np
from scipy.sparse import csc_matrix


class ProxyDistribution:
    '''Simple proxy distribution to enable specifying signal distributions from the command-line'''
    def __init__(self, name, pdf_method, sample_method):
        self.name = name
        self.pdf_method = pdf_method
        self.sample_method = sample_method

    def pdf(self, x):
        return self.pdf_method(x)

    def sample(self, count=1):
        if count == 1:
            return self.sample_method()
        return np.array([self.sample_method() for _ in xrange(count)])

    def __repr__(self):
        return self.name

def calc_fdr(probs, fdr_level):
    '''Calculates the detected signals at a specific false discovery rate given the posterior probabilities of each point.'''
    post_orders = np.argsort(probs)[::-1]
    avg_fdr = 0.0
    end_fdr = 0
    
    for idx in post_orders:
        test_fdr = (avg_fdr * end_fdr + (1.0 - probs[idx])) / (end_fdr + 1.0)
        if test_fdr > fdr_level:
            break
        avg_fdr = test_fdr
        end_fdr += 1

    is_finding = np.zeros(probs.shape)
    is_finding[post_orders[0:end_fdr]] = 1
    return is_finding

def filter_nonrectangular_data(data, filter_value=0):
    '''Convert the square matrix to a vector containing only the values different than the filter values.'''
    x = data != filter_value
    nonrect_vals = np.arange(x.sum())
    nonrect_to_data = np.zeros(data.shape, dtype=int) - 1
    data_to_nonrect = np.where(x.T)
    data_to_nonrect = (data_to_nonrect[1],data_to_nonrect[0])
    nonrect_to_data[data_to_nonrect] = nonrect_vals
    nonrect_data = data[x]
    return (nonrect_data, nonrect_to_data, data_to_nonrect)

def sparse_2d_penalty_matrix(data_shape, nonrect_to_data=None):
    '''Create a sparse 2-d penalty matrix. Optionally takes a map to corrected indices, useful when dealing with non-rectangular data.'''
    row_counter = 0
    data = []
    row = []
    col = []

    if nonrect_to_data is not None:
        for j in xrange(data_shape[1]):
            for i in xrange(data_shape[0]-1):            
                idx1 = nonrect_to_data[i,j]
                idx2 = nonrect_to_data[i+1,j]
                if idx1 < 0 or idx2 < 0:
                    continue
                row.append(row_counter)
                col.append(idx1)
                data.append(-1)

                row.append(row_counter)
                col.append(idx2)
                data.append(1.)

                row_counter += 1
        for j in xrange(data_shape[1]-1):
            for i in xrange(data_shape[0]):
                idx1 = nonrect_to_data[i,j]
                idx2 = nonrect_to_data[i,j+1]
                if idx1 < 0 or idx2 < 0:
                    continue
                row.append(row_counter)
                col.append(idx1)
                data.append(-1)

                row.append(row_counter)
                col.append(idx2)
                data.append(1.)

                row_counter += 1
    else:
        for j in xrange(data_shape[1]):
            for i in xrange(data_shape[0] - 1):
                idx1 = i+j*data_shape[0]
                idx2 = i+j*data_shape[0]+1

                row.append(row_counter)
                col.append(idx1)
                data.append(-1.)

                row.append(row_counter)
                col.append(idx2)
                data.append(1.)

                row_counter += 1

        col_counter = 0
        for i in xrange(data_shape[0]):
            for j in xrange(data_shape[1] - 1):
                idx1 = col_counter
                idx2 = col_counter+data_shape[0]

                row.append(row_counter)
                col.append(idx1)
                data.append(-1.)

                row.append(row_counter)
                col.append(idx2)
                data.append(1.)

                row_counter += 1
                col_counter += 1

    num_rows = row_counter
    num_cols = max(col) + 1
    return csc_matrix((data, (row, col)), shape=(num_rows, num_cols))

def sparse_1d_penalty_matrix(data_len):
    penalties = np.eye(data_len, dtype=float)[0:-1] * -1
    for i in xrange(len(penalties)):
        penalties[i,i+1] = 1
    return csc_matrix(penalties)