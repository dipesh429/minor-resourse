import pandas as pd
import numpy as np
import torch
import torch.nn as nn  #for implementing neural network
import torch.optim as optim #for optimizer

#importing datasets
a=0
b=0
data = pd.read_csv('combine.csv',header= None)

#omitting coach
#x= data[[0,1,3,6,7,8,9,10,11,12,13,14,15,27,39]].values

x= data[[0,1,6,7,8,9,10,11,12,13,14,15]].values
#w= data[[0,1,3,6,7,8,9,10,11,12,13,14,15]]

y= data[5].values

#manipulating datasets to desired form

#manipulating output

for i,val in enumerate(y):
    splt= val.replace(' ','').split('-')
    if splt[0]>splt[1]:
        y[i]=0
    elif splt[0]==splt[1]:   
        y[i]=1
    else:
        y[i]=2
        
        
#manipulating input
#3-7  8-12
a=38

for j,value in enumerate(x):
    count1 = 0
    count2 = 0
    for i,val in enumerate(value):
        
        if i in range(2,7):
            if val=='W':
                count1+=1
            if val=='D':
                count1+=0.5
        
        if i in range(7,12):
            if val=='W':
                count2+=1
            if val=='D':
                count2+=0.5
        
       
    form1= count1/5
    form2= count2/5
        
    x[j][2]=form1
    x[j][3]=form2


x=pd.DataFrame(x)


x.drop([4,5,6,7,8,9,10,11],axis=1,inplace= True)

x=x.values

#y= pd.DataFrame(y)   
    

# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_team = LabelEncoder()

labelencoder_team.fit(x[:,0:2].reshape(-1))



#np.save('classes.npy',labelencoder_team.classes_)
x[:, 0] = labelencoder_team.transform(x[:, 0])
x[:, 1] = labelencoder_team.transform(x[:, 1])



#labelencoder_coach = LabelEncoder()
#
#labelencoder_coach.fit(x[:,5:7].reshape(-1))
#
#x[:, 5] = labelencoder_coach.transform(x[:, 5])
#x[:, 6] = labelencoder_coach.transform(x[:, 6])
#

#onehotencoder = OneHotEncoder(categorical_features = [0,1,5,6])
onehotencoder = OneHotEncoder(categorical_features = [0,1])
onehotencoder.fit(x)
x = onehotencoder.transform(x).toarray()


x=pd.DataFrame(x)

#deleting each field from home and away team
x.drop([0,51],axis=1,inplace= True)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# Creating the architecture of the Neural Network

#input=x.values.shape

class SoftmaxRegressionModel(nn.Module):
    
    def __init__(self):
#        super(SoftmaxRegressionModel,self).__init__()
        super().__init__()
#        self.linear = nn.Linear(53,3)
        
        self.fc1 = nn.Linear(52,13)
#        self.sigmoid = nn.Sigmoid()
        
#        self.fc1 = nn.Linear(182,100)
        self.sigmoid = nn.Sigmoid()
        
#        self.tanh = nn.Tanh()
#        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(13,3)
#        self.linear = nn.Linear(20,3)
        
    def forward(self,x):
#        out = self.linear(x)
        out = self.fc1(x)
        out = self.sigmoid(out)
      
#        out = self.tanh(out)
        out = self.fc2(out)
        return out
    

model = SoftmaxRegressionModel()

criterion = nn.CrossEntropyLoss() 

learning_rate = 0.0005

optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

num_epochs = 50

for epoch in range(num_epochs):

    for i,each in enumerate(x_train):
        
#        data_input = each.reshape(-1,input)
        
        data_input = torch.FloatTensor(each)
                
        optimizer.zero_grad()
        
        outputs = model(data_input)
        
        outputs= outputs.reshape(-1,3)
              
        
        label=torch.LongTensor(np.array([y_train[i]]))
        
       
        loss = criterion(outputs, label)
        
        loss.backward()
        
        optimizer.step()
        
    print('epoch: '+str(epoch)+' loss: '+ str(loss))
    

total=b
right=a
for i,each in enumerate(x_test):
   if(i==303):
       print(each)
       break
   
   
   data_input = torch.FloatTensor(each)
   
   total+=1
                
   outputs = model(data_input)
   
   outputs= outputs.reshape(-1,3)
   
   m = nn.Softmax()
   
   n=m(outputs)
   predicted = torch.max(n,1)[1]
   
   labels=torch.LongTensor(np.array([y_test[i]]))
            
   if(predicted==labels):
       right+=1

   
print("Accuracy is "+str(right/total))    





##save model
#torch.save(model.state_dict(),'model.pkl')#saves only parameters

##load model
#model.load_state_dict(torch.load('model.pkl'))

#single_x = np.array(["STOKE CITY","LIVERPOOL"])

#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#from sklearn.externals import joblib

#joblib.dump(labelencoder_team, 'label.pkl')
#labelencode = joblib.load('label.pkl')
#labelencoder_team = LabelEncoder()
#labelencoder_team.classes_ = np.load('classes.npy')

#single_x[0] = labelencode.transform(single_x[0].reshape(-1))[0]
#single_x[1] = labelencode.transform(single_x[1].reshape(-1))[0]
  
#joblib.dump(onehotencoder, 'onehot.pkl')   
#onehotencod = joblib.load('onehot.pkl')
 

#single_x = onehotencoder.transform(single_x.reshape(1,-1)).toarray()
#single_x
#single_x=single_x.values 

#single_x =pd.DataFrame(single_x )

#deleting each field from home and away team
#single_x.drop([0,51],axis=1,inplace= True)

#single_x=single_x.values

#single_x=np.append(single_x, 0.8,axis=0)
#single_x.push(0.8)
#single_x[51]=0.4

#joblib.dump(model, 'model.pkl')
#model = joblib.load('model.pkl')


