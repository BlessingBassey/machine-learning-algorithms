from utils import *
from Data_preprocessing import *
from models import *
from sklearn.model_selection import train_test_split


X_train= pd.read_csv('Xtr_mat100.csv',header=None)
X_test = pd.read_csv('Xte_mat100.csv', header=None)
df_tr = pd.read_csv('Xtr.csv')
df_te = pd.read_csv('Xte.csv')

Y_train = pd.read_csv('Ytr.csv')
y = Y_train['Bound'].to_numpy()
y=2*y-1


kmer_size = 3

def base2int(c):
    return {'A': 0, 'C':1, 'G':2, 'T':3}.get(c,0)

def get_kmers(sequence, kmer_size=3):
    return [sequence[i: i + kmer_size] for i in range(len(sequence) - kmer_size)]
    
def base2int(c):
    return {'A': 0, 'C': 1, 'G': 2, 'T': 3}.get(c,0)

def index(kmer):
    #Transform the kmer into sequence of character indices
    base_indices = np.array([base2int(base) for base in kmer])
    multiplier = 4**np.arange(len(kmer))  #[4**0, 4**1, 4**2, ...]
    kmer_index = multiplier.dot(base_indices)
    
    return kmer_index
    
def spectrum_embedding1(sequence):
    kmers = get_kmers(sequence)
    kmer_indices = [index(kmer) for kmer in kmers]
    one_hot_vector = np.zeros(4**kmer_size)
    for kmer_index in kmer_indices:
        one_hot_vector[kmer_index] +=1
    return one_hot_vector     

def spectrum_embedding(data, data_test, k):
    out, out2 = [], []
    for i in range(len(data)):
        line = data.iloc[i]['seq']
        kmers = get_kmers(line,k)
        out.append(kmers)
        
    for i in range(len(data_test)):
        line = data_test.iloc[i]['seq']
        kmers = get_kmers(line,k)
        out2.append(kmers)
        
    data = pd.DataFrame(data=out,columns=['txt'+str(i) for i in range(len(kmers))])
    data_test = pd.DataFrame(data=out2,columns=['txt'+str(i) for i in range(len(kmers))])
    
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_data = encoder.fit(data)
        
    data_train = encoded_data.transform(data)
    data_test = encoded_data.transform(data_test)
    return data_train, data_test

dtrain, dtest = spectrum_embedding(df_tr, df_te, k=5)

Data_tr = [spectrum_embedding1(i) for i in df_tr['seq']]
Data_te = [spectrum_embedding1(j) for j in df_te['seq']]

Data_tr = pd.DataFrame(Data_tr) 
Data_te = pd.DataFrame(Data_te)

Data_tr = Data_tr.reset_index()
Data_te = Data_te.reset_index()

Data_tr = Data_tr.rename(columns={'index':'Id'})
Data_te = Data_te.rename(columns={'index':'Id'})

Data_tr = Data_tr.to_numpy()
Data_te = Data_te.to_numpy()


def string_to_float(data):
    X=[]
    n = data.shape[0]
    k=10
    for i in data.index:
        lis = data.iloc[i, :].str.split()
        newlis = np.array(lis.tolist(),dtype=float)
        newlis = newlis.reshape(1,-1)
        X.append(newlis)
   
    return X  
    
Xtr=string_to_float(X_train)
Xte=string_to_float(X_test)
New_Xtr = np.array(Xtr).squeeze()
New_Xte = np.array(Xte).squeeze()



data = pd.DataFrame(New_Xtr)

data_normalized = MinMaxScaler().fit_transform(New_Xtr)
data = pd.DataFrame(data_normalized)

seqq_tra = df_tr['seq']
seqq_tes = df_te['seq']

result_tra = map(lambda x: [x[i:i+1] for i in range(0, len(x), 1)],seqq_tra )
result_tes = map(lambda x: [x[i:i+1] for i in range(0, len(x), 1)],seqq_tes )

df_tra = list(result_tra)
df_tes = list(result_tes)

dff_tra = pd.DataFrame(df_tra)
dff_tes = pd.DataFrame(df_tes)

enc = OneHotEncoder(handle_unknown='ignore')
hot_tra = pd.DataFrame(enc.fit_transform(dff_tra).toarray())
hot_tes = pd.DataFrame(enc.fit_transform(dff_tes).toarray())



def init():
   
    print("splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = tts(dtrain, y, test_size=0.5, random_state=42)
    reg_strength = 10000 # regularization strength
    learning_rate = 0.001

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

def error(ypred, ytrue):
    e = (ypred == ytrue).mean()
    return e


print('Implementation of Linear regression with accuracy of 29.00%')
print('Training started ...')
X_train, X_test, Y_train, Y_test =  train_test_split(Data_tr, y,test_size=0.7,random_state=42)
lam = 0.1
beta = solveLRR(Y_train, X_train, lam)
probas_pred = 1/(1+np.exp(-beta.T.dot(X_test.T)))
y_pred = np.round(probas_pred)
print("Our model's performance:")
print('Accuracy: {:.2%}'.format(accuracy_score(Y_test, y_pred)))
#print('AUC: {:.2%}'.format(roc_auc_score(Y_test, probas_pred)))




#print('Implementation of SVM with accuracy of 51.5%')
#print('Training started ...')
#reg_strength = 10000 # regularization strength
#learning_rate = 0.001
#init()



print('Implementation of Logistic regression with accuracy of 62.70%')
print('Training started ...')
X_train, X_test, y_train, y_test = tts(dtrain, y, test_size = 0.5, random_state=101)
kernel = 'linear'
degree = 2
sigma = 10.
lambd = 0.001
fig_title = 'Logistic Regression, {} Kernel'.format(kernel)

model = KernelLogisticRegression(lambd=lambd, kernel=kernel, sigma=sigma, degree=degree)
y_pred = model.fit(X_train, y_train).predict(X_test)
#plot_decision_function(model, X_train, y_train, title=fig_title)
print('Accuracy: {:.2%}'.format(error(y_pred, y_test)))




print('Implementation of kernel SVM with accuracy of 30.25%')
print('Training started ...')
Y_train = pd.read_csv('Ytr.csv')
y = Y_train['Bound'].to_numpy()
y=2*y-1
X_train, X_test, y_train, y_test = tts(Data_tr, y, test_size = 0.2, random_state=42)
kernel = 'polynomial'
degree = 2
sigma = 5.
C = 10.
model = KernelSVM(C=C, kernel=kernel, sigma=sigma, degree=degree)
model.fit(X_train, y_train)
y_pred =model.predict(X_test)
print('Accuracy: {:.2%}'.format(model.error(y_pred, y_test)))




print('Implementation of kernel ridge regression with accuracy of 67.5%, this gave us our best score on the public leaderboard')
print('Training started ...')
kernel = 'rbf'
lam = 0.001
sigma = 10.
C=10.
degree=2
y=(y+1)/2
X_cross=dtrain
y_cross=y
from sklearn.model_selection import KFold
kfold=KFold(n_splits=5)
accuracy = []
for i, (train_index, validate_index) in enumerate(kfold.split(X_cross)):
    X_train, y_train = X_cross[train_index], y_cross[train_index]
    X_valid, y_valid = X_cross[validate_index], y_cross[validate_index]
    
    model_curr = KernelRidgeRegression(
        kernel=kernel,
        lambd=lam,
        sigma=sigma
    ).fit(X_train, y_train)

    y_hat = model_curr.predict(X_valid)
    accuracy = model_curr.error(y_hat,y_valid)
    
    print 'accurracy fold {i}:', {accuracy}
print 'Average accuracy is:', {np.mean(accuracy)}




print('Implementation of Multinomial Naive bayes with accuracy of 64.5% on the public leaderboard')
print('Training started ...')
X_cross=hot_tra.to_numpy()
y_cross=y
from sklearn.model_selection import KFold
kfold=KFold(n_splits=5)
accuracy = []
for i, (train_index, validate_index) in enumerate(kfold.split(X_cross)):
    X_train, y_train = X_cross[train_index], y_cross[train_index]
    X_valid, y_valid = X_cross[validate_index], y_cross[validate_index]   
    
    model_curr = MultinomialNB().fit(X_train,y_train)

    y_hat = model_curr.predict(X_valid)
    accuracy = model_curr.evaluate(X_valid,y_valid)

    print ('accurracy fold {i}:', {accuracy})
print ('Average accuracy is:', {np.mean(accuracy)})





print('Implementation of Gaussian Naive bayes with accuracy of 61.75%')
print('Training started ...')
X_train, X_test, y_train, y_test = tts(hot_tra.to_numpy(), y, test_size = 0.2, random_state=42)
m1 = GaussianNB()
m1.fit(X_train,y_train)
p=m1.predict(X_test)
print(m1.evaluate(X_test, y_test))
print('The accuracy is :', m1.evaluate(X_test, y_test))
