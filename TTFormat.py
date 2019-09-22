import numpy as np
import copy


def get_memory(tensor):
    ans = 0
    if type(tensor) == TTFormat:
        for g in tensor.cores:
            ans += np.prod(g.shape)
    else:
        ans += np.prod(tensor.shape)
    return ans


class TTFormat:
    def __init__(self, tensor=None, accuracy=0.):
        if tensor is None:
            self.ndim = None
            self.shape = None
            self.size = None
            self.cores = []
        else:
            d = tensor.ndim
            self.ndim = tensor.ndim  # dimension of current tensor A
            self.shape = tensor.shape  # shape of current tensor A
            self.size = tensor.size  # number of elements
            q = (accuracy / np.sqrt(d - 1)) * np.linalg.norm(tensor)  # truncation parameter
            ranks = [1, ]  # TT-ranks
            c = tensor
            self.cores = []  # cores of current TT-decomposition
            for k in np.arange(1, d):
                temp_prod = int(ranks[k - 1] * tensor.shape[k - 1])
                c = np.reshape(c, (temp_prod, c.size // temp_prod))

                # computing q-truncated SVD and sigma-rank:
                u, sigma, v = np.linalg.svd(c)
                q_rank = 1
                b = (sigma[:q_rank] * u[:, :q_rank]) @ v[:q_rank, :]
                while np.linalg.norm(c - b) > q and q_rank != len(sigma):
                    q_rank += 1
                    b = (sigma[:q_rank] * u[:, :q_rank]) @ v[:q_rank, :]
                ranks.append(q_rank)

                # creating new core
                self.cores.append(np.reshape(u[:, :q_rank], (ranks[k - 1], tensor.shape[k - 1], ranks[k])))
                c = np.diag(sigma[:q_rank]) @ v[:q_rank, :]

            temp_core = np.zeros((c.shape[0], c.shape[1], 1))
            for i in np.arange(c.shape[0]):
                for j in np.arange(c.shape[1]):
                    temp_core[i, j, 0] = c[i, j]
            self.cores.append(temp_core)  # the last core, doing this for common style

    def __getitem__(self, indices):
        if len(indices) != self.ndim:  # checking for appropriate number of indices
            raise IndexError('Not appropriate number of indices')

        answer = self.cores[0][:, indices[0], :]
        for i in range(1, self.ndim):  # computing the element
            answer = answer @ self.cores[i][:, indices[i], :]
        return round(answer[0][0], 10)

    def __add__(self, other):  # Addition
        TTFormat.check_compatibility(self, other)

        res = TTFormat()
        res.ndim = self.ndim
        res.shape = self.shape
        res.size = self.size
        res.cores = [np.array([])] * res.ndim

        # creating first core
        res.cores[0] = np.array([np.concatenate((self.cores[0][0], other.cores[0][0]), axis=1)])

        # creating cores from 2 to d - 1
        for k in np.arange(1, res.ndim - 1):
            d1, d2, d3 = self.cores[k].shape[0] + other.cores[k].shape[0], res.shape[k], \
                         self.cores[k].shape[2] + other.cores[k].shape[2]
            res.cores[k] = np.zeros((d1, d2, d3))
            for i in np.arange(res.shape[k]):
                temp_matrix = np.zeros((d1, d3))
                temp_matrix[:self.cores[k].shape[0], :self.cores[k].shape[2]] = self.cores[k][:, i, :]
                temp_matrix[self.cores[k].shape[0]:, self.cores[k].shape[2]:] = other.cores[k][:, i, :]
                res.cores[k][:, i, :] = temp_matrix

        # creating the last core
        res.cores[-1] = np.concatenate((self.cores[-1], other.cores[-1]), axis=0)
        self.tt_rounding()
        return res

    def __mul__(self, other):  # multiplication by a number
        if type(other) == float or type(other) == int or type(other) == complex:
            res = copy.deepcopy(self)
            res.cores[0] *= other
        else:
            raise TypeError('Not appropriate type of argument: ' + type(other))
        return res

    def __rmul__(self, other):
        return self * other

    @staticmethod
    def check_compatibility(self, other):  # function for checking compatibility of given tensors: type and shapes
        if type(self) != TTFormat or type(other) != TTFormat:
            raise TypeError('Not appropriate type of arguments: ' + type(self) + ' and ' + type(other))
        elif self.shape != other.shape:
            raise ValueError('Not appropriate shapes of arguments: ' + str(self.shape) + ' and ' + str(other.shape))

    @staticmethod
    def hadamard_product(self, other):
        TTFormat.check_compatibility(self, other)

        res = TTFormat()  # Creating main attributes
        res.ndim = self.ndim
        res.shape = self.shape
        res.size = self.size
        res.cores = [np.array([])] * res.ndim  # Creating cores
        for k in np.arange(res.ndim):
            res.cores[k] = np.kron(self.cores[k], other.cores[k])
        return res

    @staticmethod
    def multidimensional_contraction(tt_tensor, *vectors):
        if type(tt_tensor) != TTFormat:
            raise TypeError('Not appropriate type of argument')
        elif tt_tensor.ndim != len(vectors):
            raise ValueError('Not appropriate number of vectors: ' + str(len(vectors)))

        g = [np.zeros((tt_tensor.cores[i].shape[0], tt_tensor.cores[i].shape[2])) for i in np.arange(tt_tensor.ndim)]
        for k in np.arange(tt_tensor.ndim):
            for i in np.arange(tt_tensor.shape[k]):
                g[k] += tt_tensor.cores[k][:, i, :] * vectors[k][i]
        v = g[0]
        for k in np.arange(1, tt_tensor.ndim):
            v = v @ g[k]
        return round(v[0][0], 10)  # because in this case v is matrix of size 1

    @staticmethod
    def scalar_product(self, other):
        TTFormat.check_compatibility(self, other)

        v = np.kron(self.cores[0][:, 0, :], other.cores[0][:, 0, :])
        for i in np.arange(1, self.shape[0]):
            v += np.kron(self.cores[0][:, i, :], other.cores[0][:, i, :])

        for k in np.arange(1, self.ndim):
            temp_res = v @ np.kron(self.cores[k][:, 0, :], other.cores[k][:, 0, :])
            for i in np.arange(1, self.shape[k]):
                temp_res += v @ np.kron(self.cores[k][:, i, :], other.cores[k][:, i, :])
            v = temp_res

        return round(v[0][0], 10)  # because in this case v is matrix of size 1

    @staticmethod
    def fro_norm(tt_tensor):
        if type(tt_tensor) != TTFormat:
            raise TypeError('Not appropriate type of argument: ' + type(tt_tensor))

        return np.sqrt(TTFormat.scalar_product(tt_tensor, tt_tensor))

    def tt_rounding(self, accuracy=0.):
        q = (accuracy / np.sqrt(self.ndim - 1)) * TTFormat.fro_norm(self)  # truncation parameter (useless)
        for k in np.arange(self.ndim - 1, 0, -1):
            r1, n, r2 = self.cores[k].shape
            self.cores[k], r = np.linalg.qr(np.reshape(self.cores[k], (r1, n * r2)).T)
            r1 = self.cores[k].shape[1]
            self.cores[k] = np.reshape(self.cores[k].T, (r1, n, r2))
            self.cores[k - 1] = np.tensordot(self.cores[k - 1], r.T, axes=1)

        for k in np.arange(self.ndim - 2):
            r1, n, r2 = self.cores[k].shape
            self.cores[k], sigma, vt = np.linalg.svd(np.reshape(self.cores[k], (r1 * n, r2)), full_matrices=False)
            sigma = list(filter(lambda x: x >= q, sigma))
            s = np.diag(sigma)
            self.cores[k + 1] = np.tensordot((s @ vt).T, self.cores[k + 1], axes=([0], [0]))
            self.cores[k] = np.reshape(self.cores[k], (r1, n, self.cores[k].shape[1]))
