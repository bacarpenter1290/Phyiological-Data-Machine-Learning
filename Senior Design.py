# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:47:15 2020

@author: Baylee
"""
import wfdb as wf
import numpy as np
import graphviz
#from biosppy.signals import ecg
#import heartpy as hp
import matplotlib.pyplot as plt
#from scipy.signal import resample
#import scikitplot as skplt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn import tree
from sklearn import preprocessing

signals6, fields6 = wf.io.rdsamp("drive06")
signals7, fields7 = wf.io.rdsamp("drive07")
signals8, fields8 = wf.io.rdsamp("drive08")
signals11, fields11 = wf.io.rdsamp("drive11")
signals10, fields10 = wf.io.rdsamp("drive10")
print(signals6)
print(fields6)
ecg6_list = []
emg6_list = []
fgsr6_list = []
hgsr6_list = []
hr6_list = []
marker6_list = []
resp6_list = []

for i in range(len(signals6)):
    ecg6_list.append(signals6[i][0])
    emg6_list.append(signals6[i][1])
    fgsr6_list.append(signals6[i][2])
    hgsr6_list.append(signals6[i][3])
    hr6_list.append(signals6[i][4])
    marker6_list.append(signals6[i][5])
    resp6_list.append(signals6[i][6])
    
   
marker7_list = []

for i in range(len(signals7)):
    marker7_list.append(signals7[i][5])
ecg6_np = np.asarray(ecg6_list)

marker8_list = []

for i in range(len(signals8)):
    marker8_list.append(signals8[i][5])

marker11_list = []
for i in range(len(signals11)):
    marker11_list.append(signals11[i][5])

marker10_list = []
for i in range(len(signals10)):
    marker10_list.append(signals10[i][5])

print(ecg6_np)
    
#for i in range(10):
#   print(ecg6_list[i], end='\n')
#   print(emg6_list[i], end='\n')
    
plt.figure(1)
plt.subplot(411)
plt.plot(ecg6_list)
plt.plot(emg6_list)
plt.plot(fgsr6_list)
plt.plot(hgsr6_list)
plt.plot(hr6_list)
plt.plot(resp6_list)

plt.figure(1)
plt.subplot(412)
plt.plot(marker8_list)

plt.subplot(413)
plt.plot(marker10_list)

plt.subplot(414)
plt.plot(marker11_list)
# wf.plot_all_records()

# process it and plot
#ts,filtered,rpeaks,templates_ts,templates,heart_rate_ts,heart_rate = ecg.ecg(signal=ecg6_list, sampling_rate=100, show=True)

sectionLength = 65

restStart6 = 4000
cityStart6 = 37000
HWStart6 = 45000

restStart8 = 4000
cityStart8 = 28000
HWStart8 = 45000

restStart10 = 4000
cityStart10 = 20000
HWStart10 = 32000

restStart11 = 4000
cityStart11 = 20000
HWStart11 = 31000

trainDataRest6 = signals6[restStart6:restStart6 + sectionLength][:]
trainLabelsRest6 = np.ones(len(trainDataRest6))

trainDataCity6 = signals6[cityStart6:cityStart6 + sectionLength][:]
trainLabelsCity6 = np.ones(len(trainDataCity6)) * 3

trainDataHW6 = signals6[HWStart6:HWStart6 + sectionLength][:]
trainLabelsHW6 = np.ones(len(trainDataHW6)) * 2

trainDataRest8 = signals8[restStart8:restStart8 + sectionLength][:]
trainLabelsRest8 = np.ones(len(trainDataRest8))

trainDataCity8 = signals8[cityStart8:cityStart8 + sectionLength][:]
trainLabelsCity8 = np.ones(len(trainDataCity8)) * 3

trainDataHW8 = signals8[HWStart8:HWStart8 + sectionLength][:]
trainLabelsHW8 = np.ones(len(trainDataHW8)) * 2

trainDataRest10 = signals10[restStart10:restStart10 + sectionLength][:]
trainLabelsRest10 = np.ones(len(trainDataRest10))

trainDataCity10 = signals10[cityStart10:cityStart10 + sectionLength][:]
trainLabelsCity10 = np.ones(len(trainDataCity10)) * 3

trainDataHW10 = signals10[HWStart10:HWStart10 + sectionLength][:]
trainLabelsHW10 = np.ones(len(trainDataHW10)) * 2

trainDataRest11 = signals11[restStart11:restStart11 + sectionLength][:]
trainLabelsRest11 = np.ones(len(trainDataRest11))

trainDataCity11 = signals11[cityStart11:cityStart11 + sectionLength][:]
trainLabelsCity11 = np.ones(len(trainDataCity11)) * 3

trainDataHW11 = signals11[HWStart11:HWStart11 + sectionLength][:]
trainLabelsHW11 = np.ones(len(trainDataHW11)) * 2


length = sectionLength * 3


trainData = [[] for i in range(4*length)]
trainLabels = []
for i in range(7):
    for j in range(length):
        if j < sectionLength:
            trainData[j].append(trainDataRest6[j][i])
        elif j >= sectionLength and j < 2*sectionLength:
            trainData[j].append(trainDataCity6[j-sectionLength][i])
        else:
            trainData[j].append(trainDataHW6[j-(2*sectionLength)][i])
     
for i in range(7):
    for j in range(length):
        if j < sectionLength:
            trainData[j+length].append(trainDataRest8[j][i])
        elif j >= sectionLength and j < 2*sectionLength:
            trainData[j+length].append(trainDataCity8[j-sectionLength][i])
        else:
            trainData[j+length].append(trainDataHW8[j-(2*sectionLength)][i])
   
for i in range(7):
    for j in range(length):
        if j < sectionLength:
            trainData[j+(2*length)].append(trainDataRest10[j][i])
        elif j >= sectionLength and j < 2*sectionLength:
            trainData[j+(2*length)].append(trainDataCity10[j-sectionLength][i])
        else:
            trainData[j+(2*length)].append(trainDataHW10[j-(2*sectionLength)][i])
   
for i in range(7):
    for j in range(length):
        if j < sectionLength:
            trainData[j+(3*length)].append(trainDataRest11[j][i])
        elif j >= sectionLength and j < 2*sectionLength:
            trainData[j+(3*length)].append(trainDataCity11[j-sectionLength][i])
        else:
            trainData[j+(3*length)].append(trainDataHW11[j-(2*sectionLength)][i])
      
for j in range(length):
    if j < sectionLength:
        trainLabels.append(trainLabelsRest6[j])
    elif j >= sectionLength and j < 2*sectionLength:
        trainLabels.append(trainLabelsCity6[j-sectionLength])
    else:
        trainLabels.append(trainLabelsHW6[j-(2*sectionLength)])

for j in range(length):
    if j < sectionLength:
        trainLabels.append(trainLabelsRest8[j])
    elif j >= sectionLength and j < 2*sectionLength:
        trainLabels.append(trainLabelsCity8[j-sectionLength])
    else:
        trainLabels.append(trainLabelsHW8[j-(2*sectionLength)])
  
for j in range(length):
    if j < sectionLength:
        trainLabels.append(trainLabelsRest10[j])
    elif j >= sectionLength and j < 2*sectionLength:
        trainLabels.append(trainLabelsCity10[j-sectionLength])
    else:
        trainLabels.append(trainLabelsHW10[j-(2*sectionLength)])

for j in range(length):
    if j < sectionLength:
        trainLabels.append(trainLabelsRest11[j])
    elif j >= sectionLength and j < 2*sectionLength:
        trainLabels.append(trainLabelsCity11[j-sectionLength])
    else:
        trainLabels.append(trainLabelsHW11[j-(2*sectionLength)])
        
testDataRest = signals7[4000:10000][:]
testLabelsRest = np.ones(len(testDataRest))

testDataCity = signals7[20000:25000][:]
testLabelsCity = np.ones(len(testDataCity)) * 3

testDataHW = signals7[33000:36000][:]
testLabelsHW = np.ones(len(testDataHW)) * 2

length = 14000

testData = [[] for i in range(length)]
testLabels = []
for i in range(7):
    for j in range(length):
        if j < 6000:
            testData[j].append(testDataRest[j][i])
        elif j >= 6000 and j < 11000:
            testData[j].append(testDataCity[j-6000][i])
        else:
            testData[j].append(testDataHW[j-11000][i])
            
for j in range(length):
    if j < 6000:
        testLabels.append(testLabelsRest[j])
    elif j >= 6000 and j < 11000:
        testLabels.append(testLabelsCity[j-6000])
    else:
        testLabels.append(testLabelsHW[j-11000])
        
testData = np.delete(testData, 5, 1)
trainData = np.delete(trainData, 5, 1)
signals7_del = np.delete(signals7, 5, 1)
#print(testData)
#print(trainLabels)

trainLabels_np = np.array(trainLabels)

trainData_norm = normalize(trainData,axis = 0)
plt.figure(3)
plt.plot(trainData_norm)
plt.title("Norm train data")
#plt.plot(trainLabels_np)

testData_norm = normalize(testData,axis = 0)
plt.figure(4)
plt.plot(testData_norm)
plt.title("Norm test data")
#plt.plot(testLabels)


#svc = SVC(gamma='auto')
#svc.fit(trainData_norm, trainLabels_np)
#
#mlp = MLPClassifier(max_iter=16000)
#mlp.fit(trainData_norm, trainLabels_np)
#
#km = KMeans()
#km.fit(trainData_norm, trainLabels_np)

trainData_scale = preprocessing.scale(trainData)
testData_scale = preprocessing.scale(testData)

plt.figure(5)
plt.plot(trainData_scale)
plt.title("scale train data")

testData_norm = normalize(testData,axis = 0)
plt.figure(6)
plt.plot(testData_scale)
plt.title("scale test data")

plt.figure(7)

train_in, val_in, train_out, val_out = model_selection.train_test_split(trainData_scale, trainLabels, test_size=0.2)



dtc = DecisionTreeClassifier()
tree.plot_tree(dtc.fit(train_in, train_out))
#dtc.fit(trainData, trainLabels_np)

#plot_decision_regions(trainData, trainLabels_np , clf=dtc)

#print(svc.score(testData_norm, testLabels))

#print(svc.predict(signals7_del[5000:6000]))
#print(svc.predict(signals7_del[20000:21000]))
#print(svc.predict(signals7_del[35000:36000]))

#print(mlp.score(testData_norm, testLabels))

#print(mlp.predict(signals7_del[5000:6000]))
#print(mlp.predict(signals7_del[20000:21000]))
#print(mlp.predict(signals7_del[35000:36000]))

#print(km.score(testData_norm, testLabels))
#print("Actual")
#print(val_out)
#print("predictions: ")
#print(dtc.predict(val_in))

print("Accuracy: ")
print(dtc.score(val_in, val_out))


#print(dtc.predict(signals7_del[5000:6000]))
#print(dtc.predict(signals7_del[20000:21000]))
#print(dtc.predict(signals7_del[35000:36000]))

total1 = 0
total2 = 0
total3 = 0

testRange = 1000 

for i in range(testRange):
    total1 += dtc.predict(testData_scale[1000 + i].reshape(1,-1))
#    print(dtc.predict(testData_norm[1000 + i].reshape(1,-1)))
    total2 += dtc.predict(testData_scale[7000 + i].reshape(1,-1))
    total3 += dtc.predict(testData_scale[12000 + i].reshape(1,-1))
   
avg1 = total1 / testRange
avg2 = total2 / testRange
avg3 = total3 / testRange
print("Average class during rest ", avg1)
print("Average class during city ", avg2)
print("Average class during highway ", avg3)

print(dtc.score(testData_scale, testLabels))

feature_names = ["ECG", "EMG", "Foot GSR", "Hand GSR", "HR", "RESP"]
target_names = ["Underworked", "Working Efficiently", "Overworked"]

dot_data = tree.export_graphviz(dtc, out_file=None,
                                feature_names=feature_names,  
                                class_names=target_names,
                                filled=True, rounded=True,  
                                special_characters=True) 
graph = graphviz.Source(dot_data) 
#graph.render("Decision Tree - 4 samples - 2000 section length")

plt.figure(8)
plt.plot(ecg6_list, label = "ECG (mV)")
plt.plot(emg6_list, label = "EMG (mV)")
plt.plot(fgsr6_list, label = "Foot GSR (mV)")
plt.plot(hgsr6_list, label = "Hand GSR (mV)")
plt.plot(hr6_list, label = "Heart Rate (bpm)")
plt.plot(resp6_list, label = "Respiration (mV)")
plt.title("Drive 6 Signals")
plt.xlabel("Sample")
plt.ylabel("Signal Values")
plt.legend()

plt.figure(9)
plt.plot(marker6_list)
plt.title("Drive 6 Markers")
plt.xlabel("Sample")
plt.ylabel("Marker Value")

plt.figure(10)
#for i in range(len(heart_rate)):
   #print(heart_rate[i], end=' ')

# =============================================================================
# X = heart_rate.reshape(-1,1)
# y = heart_rate_ts
# rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# 
# rf.fit(X[0:1000],y[0:1000])
# 
# # Use the forest's predict method on the test data
# predictions = rf.predict(X[1001:1100])# Calculate the absolute errors
# print(predictions)
# =============================================================================

# =============================================================================
# sample_rate = 250
# data = ECG_np
# 
# filtered = hp.filter_signal(data, cutoff = 0.05, sample_rate = sample_rate, filtertype='notch')
# 
# #visualize again
# plt.figure(figsize=(12,4))
# plt.plot(filtered)
# plt.show()
# 
# #and zoom in a bit
# plt.figure(figsize=(12,4))
# plt.plot(data[0:2500], label = 'original signal')
# plt.plot(filtered[0:2500], alpha=0.5, label = 'filtered signal')
# plt.legend()
# plt.show()
# 
# #run analysis
# wd, m = hp.process(hp.scale_data(filtered), sample_rate)
# 
# #visualise in plot of custom size
# plt.figure(figsize=(12,4))
# hp.plotter(wd, m)
# 
# #display computed measures
# for measure in m.keys():
#     print('%s: %f' %(measure, m[measure]))
#     
# #resample the data. Usually 2, 4, or 6 times is enough depending on original sampling rate
# resampled_data = resample(filtered, len(filtered) * 2)
# 
# #And run the analysis again. Don't forget to up the sample rate as well!
# wd, m = hp.process(hp.scale_data(resampled_data), sample_rate * 2)
# 
# #visualise in plot of custom size
# plt.figure(figsize=(12,4))
# hp.plotter(wd, m)
# 
# #display computed measures
# for measure in m.keys():
#     print('%s: %f' %(measure, m[measure]))
# =============================================================================

