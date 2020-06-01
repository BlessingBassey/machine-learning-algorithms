#from main import *
from Data_preprocessing import *
from models import *

# Ridge Regression (RR)
def solveRR(y, X, lam):
    n, p = X.shape
    assert (len(y) == n)

    A = X.T.dot(X)
    # Adjust diagonal due to Ridge
    A[np.diag_indices_from(A)] += lam * n
    b = X.T.dot(y)

    # Hint:
    beta = np.linalg.solve(A, b)
    # Finds solution to the linear system Ax = b
    return (beta)
    

# Weighted Ridge Regression (WRR)
def solveWRR(y, X, w, lam):
    n, p = X.shape
    assert (len(y) == len(w) == n)

    y1 = np.sqrt(w) * y
    X1 = (np.sqrt(w) * X.T).T
    # Hint:
    # Find y1 and X1 such that:
    beta = solveRR(y1, X1, lam)
    return (beta) 


# Logistic Ridge Regression (LRR)
def solveLRR(y, X, lam):
    n, p = X.shape
    assert (len(y) == n)
    
    #Parameters
    max_iter=100
    eps=1e-3
    sigmoid = lambda a: 1/(1 + np.exp(-a))
    
    #Initialize
    beta = np.zeros(p)
    
            
    # Hint: Use IRLS
    for i in range(max_iter):
        beta_old = beta
        f= X.dot(beta_old)
        W=sigmoid(f)*sigmoid(-f)
        z=f.T+y/sigmoid(y*f)
        beta = solveWRR(z, X, W, 2*lam)
    # Break condition
        if np.sum((beta-beta_old)**2) < eps:
            break
    return (beta)
    
def compute_cost(W, X, Y):
    # calculate hinge loss
    reg_strength = 10000 # regularization strength
    learning_rate = 0.001
    N = X.shape[0]
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0  # equivalent to max(0, distance)
    hinge_loss = reg_strength * (np.sum(distances) / N)
    
    # calculate cost
    cost = 1 / 2 * np.dot(W, W) + hinge_loss
    return cost
    
def calculate_cost_gradient(W, X_batch, Y_batch):
    reg_strength = 10000 # regularization strength
    learning_rate = 0.001
    if type(Y_batch) == np.float64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])
    d = 1 - (Y_batch * np.dot(X_batch, W))
    dw = W if max(0, d) == 0 else (W - (reg_strength * Y_batch * X_batch))
    

    return dw

def sgd(features, outputs):
    max_epochs = 5000
    weights = np.zeros(features.shape[1])
    # stochastic gradient descent
    for epoch in range(1, max_epochs): 
        # shuffle to prevent repeating update cycles
        X, Y = shuffle(features, outputs)
        for ind, x in enumerate(X):
            ascent = calculate_cost_gradient(weights, x, Y[ind])
            weights = weights - (learning_rate * ascent)
            
    return weights
    
def sgd1(features, outputs):
    max_epochs = 5000
    weights = np.zeros(features.shape[1])
    nth = 0
    prev_cost = float("inf")
    cost_threshold = 0.01  # in percent
    reg_strength = 10000 # regularization strength
    learning_rate = 0.001
    # stochastic gradient descent
    for epoch in range(1, max_epochs):
        # shuffle to prevent repeating update cycles
        X, Y = shuffle(features, outputs)
        #print('After shffle', X)
        for ind, x in enumerate(X):
            ascent = calculate_cost_gradient(weights, x, Y[ind])
            weights = weights - (learning_rate * ascent)
        # convergence check on 2^nth epoch
        if epoch == 2 ** nth or epoch == max_epochs - 1:
            cost = compute_cost(weights, features, outputs)
            print("Epoch is:{} and Cost is: {}".format(epoch, cost))
            # stoppage criterion
            if abs(prev_cost - cost) < cost_threshold * prev_cost:
                return weights
            prev_cost = cost
            nth += 1
    return weights


def init():
   
    print("splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = tts(dtrain, y, test_size=0.5, random_state=42)
    

    # train the model
    print("training started...")
    W = sgd1(X_train, y_train)
    print("training finished.")
    print("weights are: {}".format(W))
    
    
    # testing the model on test set
    y_test_predicted = np.array([])
    for i in range(X_test.shape[0]):
        yp = np.sign(np.dot(W, X_test[i])) #model
        y_test_predicted = np.append(y_test_predicted, yp)
    print("accuracy on test dataset: {}".format(accuracy_score(y_test, y_test_predicted)))
    print("recall on test dataset: {}".format(recall_score(y_test, y_test_predicted)))
    print("precision on test dataset: {}".format(recall_score(y_test, y_test_predicted)))
    
def cvxopt_qp(P, q, G, h, A, b):
    P = .5 * (P + P.T)
    cvx_matrices = [
        cvxopt.matrix(M) if M is not None else None for M in [P, q, G, h, A, b] 
    ]
    #cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(*cvx_matrices, options={'show_progress': False})
    return np.array(solution['x']).flatten()

solve_qp = cvxopt_qp


def rbf_kernel_element_wise(x, y, sigma=1):
    '''
    returns the RBF (Gaussian) kernel k(x, y)
    
    Input:
    ------
    x and y are p-dimensional vectors 
    '''
    K = np.exp(- np.sum((x - y)**2) / (2 * sigma ** 2))
    return K
 
def rbf_kernel(X1, X2, sigma=10):
    '''
    Returns the kernel matrix K(X1_i, X2_j): size (n1, n2)
    
    Input:
    ------
    X1: an (n1, p) matrix
    X2: an (n2, p) matrix
    '''
    # For loop with rbf_kernel_element works but is slow in python
    # Use matrix operations!
    X2_norm = np.sum(X2 ** 2, axis=-1)
    X1_norm = np.sum(X1 ** 2, axis=-1)
    gamma = 1 / (2 * sigma ** 2)
    K = np.exp(- gamma * (X1_norm + X2_norm - 2 * np.dot(X1, X2.T)))
    return K
    
def linear_kernel(X1, X2):
    '''
    Returns the kernel matrix K(X1_i, X2_j): size (n1, n2)
    where K is the linear kernel
    
    Input:
    ------
    X1: an (n1, p) matrix
    X2: an (n2, p) matrix
    '''
    return X1.dot(X2.T)

def polynomial_kernel(X1, X2, degree=2):
    '''
    Returns the kernel matrix K(X1_i, X2_j): size (n1, n2)
    where K is the polynomial kernel of degree `degree`
    
    Input:
    ------
    X1: an (n1, p) matrix
    X2: an (n2, p) matrix
    degree: int
    '''
    return (1 + linear_kernel(X1, X2))**degree    

def sigma_from_median(X):
    '''
    Returns the median of ||Xi-Xj||
    
    Input
    -----
    X: (n, p) matrix
    '''
    pairwise_diff = X[:, :, None] - X[:, :, None].T
    pairwise_diff *= pairwise_diff
    euclidean_dist = np.sqrt(pairwise_diff.sum(axis=1))
    return np.median(euclidean_dist)
    
class KernelRidge():
    '''
    Kernel Ridge Regression
    
    Methods
    ----
    fit
    predict
    '''
    def __init__(self, sigma=None, lambd=0.1):
        self.kernel = rbf_kernel
        self.sigma = sigma
        self.lambd = lambd

    def fit(self, X, y):
        n, p = X.shape
        assert (n == len(y))
    
        self.X_train = X
        
        # Compute default sigma from data
        if self.sigma is None:
            self.sigma = sigma_from_median(X)
        
        A = self.kernel(X, X, sigma=self.sigma) + n * self.lambd * np.eye(n)
        
        ## self.alpha = (K + n lambda I)^-1 y
        # Solution to A x = y
        self.alpha = np.linalg.solve(A , y)

        return self
        
    def predict(self, X):
        # Prediction rule: 
        K_x = self.kernel(X, self.X_train, sigma=self.sigma)
        return K_x.dot(self.alpha)
        

def rbf_kernel(X1, X2, sigma=10):
    '''
    Returns the kernel matrix K(X1_i, X2_j): size (n1, n2)
    where K is the RBF kernel with parameter sigma
    
    Input:
    ------
    X1: an (n1, p) matrix
    X2: an (n2, p) matrix
    sigma: float
    '''
    # For loop with rbf_kernel_element works but is slow in python
    # Use matrix operations!
    X2_norm = np.sum(X2 ** 2, axis = -1)
    X1_norm = np.sum(X1 ** 2, axis = -1)
    gamma = 1 / (2 * sigma ** 2)
    K = np.exp(- gamma * (X1_norm[:, None] + X2_norm[None, :] - 2 * np.dot(X1, X2.T)))
    return K
    
def sigma_from_median(X):
    '''
    Returns the median of ||Xi-Xj||
    
    Input
    -----
    X: (n, p) matrix
    '''
    pairwise_diff = X[:, :, None] - X[:, :, None].T
    pairwise_diff *= pairwise_diff
    euclidean_dist = np.sqrt(pairwise_diff.sum(axis=1))
    return np.median(euclidean_dist)


class KernelMethodBase(object):
    '''
    Base class for kernel methods models
    
    Methods
    ----
    fit
    predict
    '''
    kernels_ = {
        'linear': linear_kernel,
        'polynomial': polynomial_kernel,
        'rbf': rbf_kernel,
        # 'custom_kernel': custom_kernel, # Your kernel
    }
    def __init__(self, kernel='linear', **kwargs):
        self.kernel_name = kernel
        self.kernel_function_ = self.kernels_[kernel]
        self.kernel_parameters = self.get_kernel_parameters(**kwargs)
        
    def get_kernel_parameters(self, **kwargs):
        params = {}
        if self.kernel_name == 'rbf':
            params['sigma'] = kwargs.get('sigma', 1.)
        if self.kernel_name == 'polynomial':
            params['degree'] = kwargs.get('degree', 2)
        # if self.kernel_name == 'custom_kernel':
        #     params['parameter_1'] = kwargs.get('parameter_1', None)
        #     params['parameter_2'] = kwargs.get('parameter_2', None)
        return params

    def fit(self, X, y, **kwargs):
        return self
        
    def decision_function(self, X):
        pass

    def predict(self, X):
        pass
class KernelRidgeRegression(KernelMethodBase):
    '''
    Kernel Ridge Regression
    '''
    def __init__(self, lambd=0.1, **kwargs):
        self.lambd = lambd
        # Python 3: replace the following line by
        # super().__init__(**kwargs)
        super(KernelRidgeRegression, self).__init__(**kwargs)

    def fit(self, X, y):

        self.X_train = X
        self.y_train = y
        n = len(self.y_train)
        
        A = self.kernel_function_(X, X, **self.kernel_parameters)
        A[np.diag_indices_from(A)] += self.lambd * n
        # self.alpha = (K + n lambda I)^-1 y
        self.alpha = np.linalg.solve(A , self.y_train)

        return self
    
    
    def decision_function(self, X):
        K_x = self.kernel_function_(X, self.X_train, **self.kernel_parameters)
        return np.dot(K_x,self.alpha)
    
    def predict(self, X):
        proba = self.decision_function(X)
        predicted_classes = np.where(proba <=0.5, 0, 1)
        #pred= np.where(predicted_classes==-1,0,1)
        return predicted_classes
    
    def error(self,ypred, ytrue):
        e = (ypred == ytrue).mean()
        return e
    
    
class WeightedKernelRidgeRegression(KernelRidgeRegression):
    '''
    Weighted Kernel Ridge Regression
    
    This is just used for the KernelLogistic following up
    '''
    def fit(self, K, y, sample_weights=None):

        self.y_train = y
        n = len(self.y_train)
        
        w = np.ones_like(self.y_train) if sample_weights is None else sample_weights
        W = np.diag(np.sqrt(w))
        
        A = W.dot(K).dot(W)
        A[np.diag_indices_from(A)] += self.lambd * n
        # self.alpha = W (K + n lambda I)^-1 W y
        self.alpha = W.dot(np.linalg.solve(A , W.dot(self.y_train)))

        return self

class KernelRidgeClassifier(KernelRidgeRegression):
    '''
    Kernel Ridge Classification
    '''
    def predict(self, X):
        return np.sign(self.decision_function(X))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class KernelLogisticRegression(KernelMethodBase):
    '''
    Kernel Logistic Regression
    '''
    def __init__(self, lambd=0.1, **kwargs):
        self.lambd = lambd
        # Python 3: replace the following line by
        # super().__init__(**kwargs)
        super(KernelLogisticRegression, self).__init__(**kwargs)

    def fit(self, X, y, max_iter=100, tol=1e-5):
    
        self.X_train = X
        self.y_train = y
        
        K = self.kernel_function_(X, X, **self.kernel_parameters)
        
        # IRLS
        WKRR = WeightedKernelRidgeRegression(
            lambd=self.lambd,
            kernel=self.kernel_name,
            **self.kernel_parameters
        )
        # Initialize
        alpha = np.zeros_like(self.y_train)
        # Iterate until convergence or max iterations
        for n_iter in range(max_iter):
            alpha_old = alpha
            f = K.dot(alpha_old)
            w = sigmoid(f) * sigmoid(-f)
            z = f + y / sigmoid(-y*f)
            alpha = WKRR.fit(K, z, sample_weights=w).alpha
            # Break condition (achieved convergence)
            if np.sum((alpha-alpha_old)**2) < tol:
                break
        self.n_iter = n_iter
        self.alpha = alpha

        return self
            
    def decision_function(self, X_test):
        K_x = self.kernel_function_(X_test, self.X_train, **self.kernel_parameters)
        # probability of y=1(between o and 1)
        return sigmoid(K_x.dot(self.alpha))

    def predict(self, X):
        probas =self.decision_function(X)
        predicted_classes = np.where(probas < 0.5, -1,1)
        return predicted_classes
    
    def error(self,ypred, ytrue):
        e = (ypred == ytrue).mean()
        return e

def svm_dual_soft_to_qp_kernel(K, y, C=1):
    n = K.shape[0]
    assert (len(y) == n)
        
    # Dual formulation, soft margin
    P = np.diag(y).dot(K).dot(np.diag(y))

    # As a regularization, we add epsilon * identity to P
    eps = 1e-12
    P =P+ eps * np.eye(n)
    q = - np.ones(n)
    G = np.vstack([-np.eye(n), np.eye(n)])
    h = np.hstack([np.zeros(n), C * np.ones(n)])
    A = y[np.newaxis,:]
    A = A.astype('float')
    b = np.array([0.])
    return P, q, G, h, A, b

#K = polynomial_kernel(Xtrain, Xtrain)
#alphas = solve_qp(*svm_dual_soft_to_qp_kernel(K, y, C=1.))


class KernelSVM(KernelMethodBase):
    '''
    Kernel SVM Classification
    
    Methods
    ----
    fit
    predict
    '''
    def __init__(self, C=0.1, **kwargs):
        self.C = C
        # Python 3: replace the following line by
        # super().__init__(**kwargs)
        super(KernelSVM, self).__init__(**kwargs)

    def fit(self, X, y, tol=1e-3):
        n, p = X.shape
        assert (n == len(y))
    
        self.X_train = X
        self.y_train = y
        
        # Kernel matrix
        
        K = self.kernel_function_(X,X, **self.kernel_parameters)
        
        # Solve dual problem
        self.alpha = solve_qp(*svm_dual_soft_to_qp_kernel(K, y, C=self.C))
        
        # Compute support vectors and bias b
        #sv = (self.alpha > 1e-3)
        sv = np.logical_and(self.alpha > tol, (self.C - self.alpha > tol))
        
        self.beta = self.alpha*y
        
        self.X_sv = X[sv]
        self.Y_sv = y[sv]

        self.bias = y[sv]-K[sv].dot(self.alpha*y)#self.X_sv.dot(X.T).dot(self.beta)
        
        self.bias = self.bias.mean()
        self.support_vector_indices = np.nonzero(sv)[0]

        return self
        
    def decision_function(self, X):
        return self.kernel_function_(X, self.X_train,**self.kernel_parameters).dot(self.beta)+self.bias
    
    def ensemble(self,X):
        prob = self.decision_function(X)
           
        return prob
        


    def predict(self, X):
        
        prob = np.sign(self.decision_function(X))
        
        predicted_classes = np.where(prob==-1, 0, 1)

        
        return predicted_classes
    
    def error(self,ypred, ytrue):
        e = (ypred == ytrue).mean()
        return e

class MultinomialNB:
    
    def fit(self, X, y, ls=0.01):
        self.ls = ls
        self.y_classes, y_counts = np.unique(y, return_counts=True)
        self.x_classes = [np.unique(x) for x in X.T]
        self.phi_y = 1.0 * y_counts/y_counts.sum()
        self.phi_x = self.mean_X(X, y)
        self.c_x = self.count_x(X, y)
        return self
    
    def mean_X(self, X, y):
        return [[self.ls_mean_x(X, y, k, j) for j in range(len(self.x_classes))] for k in self.y_classes]
    
    def ls_mean_x(self, X, y, k, j):
        x_data = (X[:,j][y==k].reshape(-1,1) == self.x_classes[j])
        return (x_data.sum(axis=0) + self.ls ) / (len(x_data) + (len(self.x_classes) * self.ls))
    
    def get_mean_x(self, y, j):
        return 1 + self.ls / (self.c_x[y][j] + (len(self.x_classes) * self.ls))
        
    def count_x(self, X, y):
        return [[len(X[:,j][y==k].reshape(-1,1) == self.x_classes[j])
                       for j in range(len(self.x_classes))]
                      for k in self.y_classes]

    def predict(self, X):
        return np.apply_along_axis(lambda x: self.compute_probs(x), 1, X)
    
    def compute_probs(self, x):
        probs = np.array([self.compute_prob(x, y) for y in range(len(self.y_classes))])
        return self.y_classes[np.argmax(probs)]
    
    def compute_prob(self, x, y):
        Pxy = 1
        for j in range(len(x)):
            x_clas = self.x_classes[j]
            if x[j] in x_clas:
                i = list(x_clas).index(x[j])
                p_x_j_y = self.phi_x[y][j][i] # p(xj|y)
                Pxy *= p_x_j_y
            else:
                Pxy *= self.get_mean_x(y, j)
        return Pxy * self.phi_y[y]
    
    def evaluate(self, X, y):
        return (self.predict(X) == y).mean()
        
class GaussianNB:
    
    def fit(self, X, y, epsilon = 1e-10):
        self.y_classes, y_counts = np.unique(y, return_counts=True)
        self.x_classes = np.array([np.unique(x) for x in X.T])
        self.phi_y = 1.0 * y_counts/y_counts.sum()
        self.u = np.array([X[y==k].mean(axis=0) for k in self.y_classes])
        self.var_x = np.array([X[y==k].var(axis=0)  + epsilon for k in self.y_classes])
        return self
    
    def predict(self, X):
        return np.apply_along_axis(lambda x: self.compute_probs(x), 1, X)
    
    def compute_probs(self, x):
        probs = np.array([self.compute_prob(x, y) for y in range(len(self.y_classes))])
        return self.y_classes[np.argmax(probs)]
    
    def compute_prob(self, x, y):
        c = 1.0 /np.sqrt(2.0 * np.pi * (self.var_x[y]))
        return np.prod(c * np.exp(-1.0 * np.square(x - self.u[y]) / (2.0 * self.var_x[y])))
    
    def evaluate(self, X, y):
        return (self.predict(X) == y).mean()
        
        