from tensorflow import keras
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
import matplotlib as plt

data = np.loadtxt('data.csv', delimiter = ',', dtype = 'float32')

#-----------------------------------------------------------------------------------------------------
#Data visualization
#-----------------------------------------------------------------------------------------------------

fig = plt.pyplot.figure('Data visualization', figsize = (14,8))
axes = fig.subplots(4,2)
fig.suptitle('Dane')
fig.subplots_adjust(wspace = 0.2, hspace = 0.5)

axes[0,0].hist(data[:,0], bins = np.arange(data[:,0].min(), data[:,0].max()+1), align = 'left', rwidth = 0.5)
axes[0,0].set_title('Number of times pregnant')
axes[0,0].set_xticks(np.arange(0,18,1))

axes[0,1].hist(data[:,1], bins = range(0, 201, 5), align = 'mid', rwidth = 0.9)
axes[0,1].set_xticks(range(0,201,20))
axes[0,1].set_title('Plasma glucose concentration a 2 hours')

axes[1,0].hist(data[:,2], bins = np.arange(data[:,2].min(), data[:,2].max()+1, 2), align = 'mid', rwidth = 0.5)
axes[1,0].set_xticks(range(0,131,10))
axes[1,0].set_title('Diastolic blood pressure (mm Hg)')

axes[1,1].hist(data[:,3], bins = np.arange(data[:,3].min(), data[:,3].max()+1, 2), align = 'mid', rwidth = 0.5)
axes[1,1].set_title('Triceps skin fold thickness (mm)')

axes[2,0].hist(data[:,4], bins = np.arange(data[:,4].min(), data[:,4].max()+1, 10), align = 'mid', rwidth = 0.5)
axes[2,0].set_title('2-Hour serum insulin (mu U/ml)')

axes[2,1].hist(data[:,5], bins = np.arange(data[:,5].min(), data[:,5].max()+1, 1), align = 'mid', rwidth = 0.5)
axes[2,1].set_title('Body mass index (weight in kg/(height in m)^2)')

axes[3,0].hist(data[:,6], bins = np.arange(data[:,6].min(), data[:,6].max()+1, 0.05), align = 'mid', rwidth = 0.5)
axes[3,0].set_title('Diabetes pedigree function')

axes[3,1].hist(data[:,7], bins = np.arange(data[:,7].min(), data[:,7].max()+1, 1), align = 'mid', rwidth = 0.5)
axes[3,1].set_title('Age (years)')

fig.show()

#----------------------------------------------------------------------------------------------------
#Data correlation
#----------------------------------------------------------------------------------------------------
correlation = []
for i in range (9):    
    cor = np.corrcoef(data[:,i], data[:,-1])
    correlation.append(round(cor[0,1], 4))
print(correlation)
#----------------------------------------------------------------------------------------------------
#Deleting 
#----------------------------------------------------------------------------------------------------
data = np.delete(data,[2,3],1)
#----------------------------------------------------------------------------------------------------
#Missing data problem
#----------------------------------------------------------------------------------------------------

median_1 = np.median(data[:,1])
data[:,1][data[:,1]==0] = median_1

median_2 = np.median(data[:,2])
data[:,2][data[:,2]==0] = median_2

median_3 = np.median(data[:,3])
data[:,3][data[:,3]==0] = median_3

#----------------------------------------------------------------------------------------------------
#Visualization after data preprocessing
#----------------------------------------------------------------------------------------------------

fig2 = plt.pyplot.figure('Data after preprocesing', figsize = (14,8))
axes2 = fig2.subplots(3,2)
fig2.suptitle('Dane 2')
fig2.subplots_adjust(wspace = 0.2, hspace = 0.5)

axes2[0,0].hist(data[:,0], bins = np.arange(data[:,0].min(), data[:,0].max()+1), align = 'left', rwidth = 0.5)
axes2[0,0].set_title('Number of times pregnant')
axes2[0,0].set_xticks(np.arange(0,18,1))

axes2[0,1].hist(data[:,1], bins = range(0, 201, 5), align = 'mid', rwidth = 0.9)
axes2[0,1].set_title('Plasma glucose concentration a 2 hours')

axes2[1,0].hist(data[:,2], bins = np.arange(data[:,2].min(), data[:,2].max()+1, 10), align = 'mid', rwidth = 0.5)
axes2[1,0].set_title('2-Hour serum insulin (mu U/ml)')

axes2[1,1].hist(data[:,3], bins = np.arange(data[:,3].min(), data[:,3].max()+1), align = 'mid', rwidth = 0.5)
axes2[1,1].set_title('Body mass index (weight in kg/(height in m)^2)')

axes2[2,0].hist(data[:,4], bins = np.arange(data[:,4].min(), data[:,4].max()+1, 0.05), align = 'mid', rwidth = 0.5)
axes2[2,0].set_title('Diabetes pedigree function')

axes2[2,1].hist(data[:,5], bins = np.arange(data[:,5].min(), data[:,5].max()+1), align = 'mid', rwidth = 0.5)
axes2[2,1].set_title('Age (years)')

fig2.show()

#----------------------------------------------------------------------------------------------------
#Data spliting
#----------------------------------------------------------------------------------------------------

data_y = data[:,-1]
data_x = data[:,:-1]
train_data_x, test_data_x, train_data_y, test_data_y = model_selection.train_test_split(
        data_x, data_y.reshape(-1,1), test_size = 0.2)

#----------------------------------------------------------------------------------------------------
#Feature scaling
#----------------------------------------------------------------------------------------------------
scaler = MinMaxScaler()
train_data_x = scaler.fit_transform(train_data_x)
test_data_x = scaler.fit_transform(test_data_x)

#----------------------------------------------------------------------------------------------------
#Visualization after replecing missing data with median
#----------------------------------------------------------------------------------------------------

fig3 = plt.pyplot.figure('Data before learning process', figsize = (14,8))
axes3 = fig3.subplots(3,2)
fig3.suptitle('Dane 3')
fig3.subplots_adjust(wspace = 0.2, hspace = 0.5)

axes3[0,0].hist(train_data_x[:,0], bins = np.arange(train_data_x[:,0].min(), train_data_x[:,0].max()+1, 0.1), align = 'left', rwidth = 0.5)
axes3[0,0].set_title('Number of times pregnant')

axes3[0,1].hist(train_data_x[:,1], bins = np.arange(train_data_x[:,1].min(), train_data_x[:,1].max()+1, 0.05), align = 'mid', rwidth = 0.9)
axes3[0,1].set_title('Plasma glucose concentration a 2 hours')

axes3[1,0].hist(train_data_x[:,2], bins = np.arange(train_data_x[:,2].min(), train_data_x[:,2].max()+1, 0.1), align = 'mid', rwidth = 0.5)
axes3[1,0].set_title('2-Hour serum insulin (mu U/ml)')

axes3[1,1].hist(train_data_x[:,3], bins = np.arange(train_data_x[:,3].min(), train_data_x[:,3].max()+1, 0.05), align = 'mid', rwidth = 0.5)
axes3[1,1].set_title('Body mass index (weight in kg/(height in m)^2)')

axes3[2,0].hist(train_data_x[:,4], bins = np.arange(train_data_x[:,4].min(), train_data_x[:,4].max()+1, 0.05), align = 'mid', rwidth = 0.5)
axes3[2,0].set_title('Diabetes pedigree function')

axes3[2,1].hist(train_data_x[:,5], bins = np.arange(train_data_x[:,5].min(), train_data_x[:,5].max()+1, 0.05), align = 'mid', rwidth = 0.5)
axes3[2,1].set_title('Age (years)')

fig3.show()

#----------------------------------------------------------------------------------------------------
#Data correlation after data preprocessing
#----------------------------------------------------------------------------------------------------
correlation = []
for i in range (7):    
    cor = np.corrcoef(data[:,i], data[:,-1])
    correlation.append(round(cor[0,1], 4))
print(correlation)
#----------------------------------------------------------------------------------------------------
#Model
#----------------------------------------------------------------------------------------------------

regularizer = keras.regularizers.l1_l2(l1=0.01, l2=0.01)
model = keras.Sequential()
model.add(keras.Input(6))
model.add(keras.layers.Dense(6, 'relu', kernel_initializer = 'he_uniform', kernel_regularizer = regularizer))
model.add(keras.layers.Dense(1, 'sigmoid'))

opt = keras.optimizers.SGD(lr = 0.005, momentum = 0.9)
adam = keras.optimizers.Adam(learning_rate = 0.003)

model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit(train_data_x, train_data_y,
                    batch_size = 18,
                    epochs = 100,
                    verbose = 1,
                    validation_data = (test_data_x, test_data_y))

score = model.evaluate(test_data_x, test_data_y, verbose=0)


fig4 = plt.pyplot.figure('Learning process')
ax = fig4.add_subplot()
ax.plot(history.history['acc'], label = 'train')
ax.plot(history.history['val_acc'], label = 'test')
ax.legend()
fig4.show()

print('Test loss:', score[0])
print('Test accuracy:', score[1])





