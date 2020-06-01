# Data preprocessing 
from utils import *
#from main import *
from models import *


X_train= pd.read_csv('Xtr_mat100.csv',header=None)
X_test = pd.read_csv('Xte_mat100.csv', header=None)

Y_train = pd.read_csv('Ytr.csv')
y = Y_train['Bound'].to_numpy()
y=2*y-1

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

df_tr = pd.read_csv('Xtr.csv')
df_te = pd.read_csv('Xte.csv')

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