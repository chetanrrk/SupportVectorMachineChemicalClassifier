# -*- coding: utf-8 -*-
"""
Created on Thu Jun 09 11:36:04 2016

@author: CRupakhe
"""
import os
import sys
import pickle
import copy
import threading
import Queue
import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

"""
reads features file and also returns indices of "0" class
"""
def readCSV(inFile):
    fin = open(inFile,"r")
    X = []
    y = []
    zeroMolIdx = []    
    lineNum=1
    for line in fin:
        token=line.rstrip().split(",")
        if lineNum>=2:
            des=[]
            for t in token[2:]:
                try:des.append(float(t))
                except ValueError:continue
            if len(des)==797:
                X.append(np.array(des))
                try:
                    y.append(float(token[1]))
                except ValueError:
                    continue
                if float(token[1])==0.0:zeroMolIdx.append(lineNum-1)
        lineNum+=1        
    X = np.array(X)
    y = np.array(y)    
    return X,y,zeroMolIdx


"""
scales training data to [0,1] range using MinMaxScaler
using the same scalin transforms the test data
and selects "bestFeat" number of features using chi-sq test
returns user defined fraction of class"1" to class "0" data
"""
def ScaleTrainTest(trainF,testF,fac,bestFeat = 10):
    XTr,yTr,zTrainIdx = readCSV(trainF)
    print "scaling features"    
    scaler = preprocessing.MinMaxScaler()
    XTr = scaler.fit_transform(XTr)
    XTr = SelectKBest(chi2, k=bestFeat).fit_transform(XTr, yTr) ### selecting features
    np.random.shuffle(zTrainIdx)
    posLen = len([d for d in yTr if d==1])
    XTemp = np.array([XTr[i] for i in zTrainIdx[:posLen*fac]])     
    yTemp = np.array([yTr[i] for i in zTrainIdx[:posLen*fac]])
   
    XTrain=[]; yTrain=[]
    
    for x in XTemp:XTrain.append(x)
    for y in yTemp:yTrain.append(y)    
    for i in range(len(XTr)): 
        if yTr[i]==1:
            XTrain.append(XTr[i])        
            yTrain.append(yTr[i])
            
    XTrain = np.array(XTrain);yTrain = np.array(yTrain)        
    XTest,yTest,zTestIdx = readCSV(testF) ### reading test features
    xTest = scaler.transform(XTest) ### scaling test features
    XTest = SelectKBest(chi2, k=bestFeat).fit_transform(XTest, yTest) ## selecting features
    
    return XTrain,yTrain,xTest,yTest

def selectFeatures(numberFeatures,xTrain,xPred):
    """
    scaler = preprocessing.MinMaxScaler()
    sc = scaler.fit(xTrain)
    pickle.dump(sc,open("scaler.p","w"))
    xTrainScaled = sc.transform(xTrain)
    xTestScaled = sc.transform(xPred)
    """
    pca = PCA(n_components=numberFeatures)
    pca.fit(xTrain)
    #pickle.dump(pca,open("pcaTransform.p","w"))
    print "numDimension:",numberFeatures,np.sum(pca.explained_variance_ratio_) 
    #print>>open("pca_summary.txt","w"),"Dimensions:"+str(numberFeatures)+ \
    #        " VarianceExplained:"+str(np.sum(pca.explained_variance_ratio_))
    selectedTrainFeatures = pca.transform(xTrain)
    selectedPredictFeatures = pca.transform(xPred)
    return selectedTrainFeatures,selectedPredictFeatures,pca

def ScaleAll(trainF,negFac,bestFeat = 10, nFolds=5,multipleSplit=False):
    XTr,yTr,zeroTrainIdx = readCSV(trainF)    

    scaler = preprocessing.MinMaxScaler()
    sc = scaler.fit(XTr)
    pickle.dump(sc,open("scaler.p","w"))

    XTr = sc.transform(XTr)
    
    
    if multipleSplit:
        return RandMultipleSplit(XTr,yTr,negFac,nFolds,zeroTrainIdx)
    else:
        XTrains,yTrains = PrepareCrossValSets(XTr,yTr,nFolds)
        return XTrains,yTrains
    
    
def RandMultipleSplit(XTr,yTr,negFac,nFolds,zeroTrainIdx):
    
    oneTrainIdx = [i for i in range(len(yTr)) if yTr[i]==1]    
    posLen = len(oneTrainIdx)
    
    """
    lets do multiple xvals by randomly picking "nFolds" different training sets
    """
    XTrains=[];yTrains=[]
        
    sampleSize = int((posLen*negFac)/nFolds)
        
    
    for i in range(nFolds):
        
        np.random.shuffle(zeroTrainIdx) ### to select compounds for training
        np.random.shuffle(oneTrainIdx)
            
        XTemp = np.array([XTr[i] for i in zeroTrainIdx[:sampleSize]])     
        yTemp = np.array([yTr[i] for i in zeroTrainIdx[:sampleSize]])
            
        XTrain=[]; yTrain=[]
    
        for x in XTemp:XTrain.append(x)
        for y in yTemp:yTrain.append(y)
            
        """        
        for i in range(len(XTr)): 
            if yTr[i]==1:
                XTrain.append(XTr[i])        
                yTrain.append(yTr[i])   
        """
        
        for i in oneTrainIdx[:sampleSize]:
            XTrain.append(XTr[i])
            yTrain.append(yTr[i])
    
        XTrain = np.array(XTrain);yTrain = np.array(yTrain)
        XTrains.append(XTrain);yTrains.append(yTrain)
        
        
    pickle.dump(XTrains, open("XTrains"+str(negFac)+".p","w"))
    pickle.dump(yTrains, open("yTrains"+str(negFac)+".p","w"))
    
    return XTrains,yTrains


"""
Runs the parameter scan and picks the best one
"""
def TrainAndTest(XTrain,YTrain,XTest,YTest,c,Kernel,g=0.01):
    if Kernel=="rbf":    
        clf = SVC(C=c,gamma=g,kernel=Kernel)
    elif Kernel=="linear":
        clf = SVC(C=c,kernel=Kernel)
    clf.fit(XTrain, YTrain)
    
    tePred = clf.predict(XTest)
    testPred =  [int(p) for p in tePred]       
    fpr,tpr,thr = roc_curve(YTest, testPred, pos_label=1)
    aucTest = auc(fpr,tpr)

    
    if np.isnan(aucTest): aucTest = 0.0
  
    return aucTest,clf    
    

"""
predict on the validation set using the best model from cross validation
"""
def predict(testF,negFac,numDomains,path,training):
    testX,testY,testZIdx = readCSV(testF)
    scaler = pickle.load(open(path+"scaler.p","r"))
    testXS = scaler.transform(testX)
    print "loading"
    models = pickle.load(open(path+"models"+str(negFac)+".p","r"))
    
    trainAv = pickle.load(open(path+"trainAv_pos1_neg"+str(negFac)+".p","r"))
    bestParam = trainAv.keys()[np.argmax(trainAv.values())]
    
    for i in range(len(models)):
         if models.keys()[i]== bestParam:
             bestModelIdx = i
    
    model = models.values()[bestModelIdx]    

    transTestX = []
    for i in range(len(model)):
        pc = pickle.load(open(path+"pc_fold"+str(i)+"_"+
                    str(negFac)+".p","r"))
        testXT = pc.transform(testXS)
        transTestX.append(testXT)
   
    """
    Test if test compounds is within some trained models
    """
    validY=[];probs=[];concensusPreds=[];bigTestPred=[]
        
    for i in range(len(testXT)):
        pred = []
        for j in range(len(model)):
            if isValidModel(transTestX[j][i],j,path):
                p = model[j].predict([transTestX[j][i]])
                pred.append(p)
        if len(pred)>=numDomains:
            bigTestPred.append(pred) 
            if np.sum(pred)/len(pred)*1.0 > 0.5:
                concensusPreds.append(1.0)
            else:concensusPreds.append(0.0)
            validY.append(testY[i])
        else:
            probPred=[]
            for k in range(len(model)):
                p = model[k].predict([transTestX[k][i]])
                probPred.append(p)
            probs.append([{"input_index":i},{"x":testXT},{"truth":testY[i]},
                          {"prediction":probPred}])       
    pickle.dump(probs,open("modelDomainProblems.p","w"))   
    pickle.dump(bigTestPred,open("bigTestPred.p","w"))
    if training:
        aucs=[];sensitivity=[];specificity=[]
        for i in range(len(model)):
            p = model[i].predict(transTestX[i])
            fpr,tpr,thr = roc_curve(testY,p, pos_label=1)
            ba = auc(fpr,tpr)
            aucs.append(ba)
            
            sens = computeSensitivity(p,testY)
            spec = computeSpecificity(p,testY)
            sensitivity.append(sens)
            specificity.append(spec)            
        
        print "all aucs(mean/std): "+str(np.mean(aucs))+" "+str(np.std(aucs))
        print "all sensitivity(mean/std): "+str(np.mean(sensitivity))+" "+ \
                str(np.std(sensitivity))    
        print "all specificity(mean/std): "+str(np.mean(specificity))+" "+ \
                str(np.std(specificity))        
    
        
        print "concensus predictions..."
        #concensusScore = np.sum(np.array(preds),axis=0)/len(model)
        #concensusPreds = [1.0 if i>0.5 else 0.0 for i in concensusScore]
        
        #print "validY)",validY
        #print "len(concensusPreds)",len(concensusPreds)
        fpr,tpr,thr = roc_curve(validY, concensusPreds, pos_label=1)
        concensusAuc = auc(fpr,tpr)
        print "concensus auc: "+ str(concensusAuc)
        concensusSensitivity = computeSensitivity(concensusPreds,validY)     
        concensusSpecificity = computeSpecificity(concensusPreds,validY)    
        print "concensusSensitivity: "+ str(concensusSensitivity)    
        print "concensusSpecificity: "+str(concensusSpecificity)
    else:
        pickle.dump(concensusPreds,open("predictionSetLabels.p","w"))        
    print "coverage: "+str(len(validY)*1.0/len(testY))    
    
"""
Get the scaled training data (XTr) and performs 'nFold 'cross validation
"""
def PrepareCrossValSets(XTrain,yTrain,nFold):
    folds = [i%nFold for i in range(len(XTrain))]
    XTrains=[[] for i in range(nFold)];yTrains=[[] for i in range(nFold)]
    idxs = range(len(XTrain))
    np.random.shuffle(idxs) ### randomizing the training examples
    XShuffled = [XTrain[i] for i in idxs]
    yShuffled = [yTrain[i] for i in idxs]
    for j in range(nFolds):
        for i in range(len(folds)):
            if folds[i]==j:
                XTrains[folds[i]].append(XShuffled[i])
                yTrains[folds[i]].append(yShuffled[i])
               
    return XTrains,yTrains


"""
1) n-Fold cross validates each supplied parameter
2) computes the aucs for each parameter
"""
class ThreadedCrossVal(threading.Thread):
    def __init__(self,threadName,foldsTrain,foldsTest,c,g,Kernel):
        threading.Thread.__init__(self)
        self.threadName= threadName
        self.foldsTrain = foldsTrain
        self.foldsTest = foldsTest
        self.c = c
        self.g = g
        self.Kernel = Kernel
        self.aucsTestQ = Queue.Queue()
        self.modelsQ = Queue.Queue()
        
    def run(self):       
        print "starting",self.name
        for i in range (len(self.foldsTrain)):
            print self.name+" cross validating fold"+str(i)+" ..."
            """                
            idx = list(set(range(len(self.XTrains)))-set([i]))
            XTrain = np.concatenate([self.XTrains[j] for j in idx],axis=0)
            YTrain = np.concatenate([self.YTrains[j] for j in idx],axis=0)
            XTest = self.XTrains[i]
            YTest = self.YTrains[i]
            """    
            if self.Kernel=="linear":
                aucTest,clf = TrainAndTest(self.foldsTrain[i][0],
                        self.foldsTrain[i][1],self.foldsTest[i][0],
                        self.foldsTest[i][1],self.c,self.Kernel)
                key = str(self.c)    
            else:
                aucTest,clf = TrainAndTest(self.foldsTrain[i][0],
                        self.foldsTrain[i][1],self.foldsTest[i][0],
                        self.foldsTest[i][1],self.c,self.Kernel,self.g)
                key = str(self.c)+"_"+str(self.g)    
                
            self.aucsTestQ.put({key:aucTest})
            self.modelsQ.put({key:clf})                    
            

def AssembleHelper(q,resultCollector):
    
    for i in range(q.qsize()):
        res = q.get()
        #print res
        if not resultCollector.has_key(res.keys()[0]): 
            resultCollector[res.keys()[0]] = [res.values()[0]]
        else:
            resultCollector[res.keys()[0]].append(res.values()[0])
    return resultCollector
    

def AssembleResults(threadsC):
    testResultCollector = {}
    modelsCollector = {}
    for t in threadsC:    
        testAucs = AssembleHelper(t.aucsTestQ,testResultCollector)
        models = AssembleHelper(t.modelsQ,modelsCollector)    
                 
    return testAucs,models    
    
    
def ComputeResultsStats(resultsDic):
    resultsAvg = {}
    resultsStd = {}
    for k in resultsDic:
        resultsAvg[k] = np.average(resultsDic[k],axis=0)
        resultsStd[k] = np.std(resultsDic[k],axis=0)
    return resultsAvg,resultsStd
    
    
"""
cross validates on the already selected random samples
randCrossVal() does nFold cross validation and returns result for each param
run() inatializes params to search and also collects and returns best params
"""
def CrossValidate(XTrs,yTrs,paramsCs,paramsGamma,Kernel,bestFeats):  

    threadCount = 0
    myThreads = [] ## keeps track of thread    
    activeThreads=[]
    
    """
    Lets do pca based feature selection here
    """    
    foldsTrain = []
    foldsTest = []
    for i in range (len(XTrs)):
        idx = list(set(range(len(XTrs)))-set([i]))
        XTrain = np.concatenate([XTrs[j] for j in idx],axis=0)
        YTrain = np.concatenate([yTrs[j] for j in idx],axis=0)
        XTest = XTrs[i]
        YTest = yTrs[i]
        XTrain,XTest,pc = selectFeatures(bestFeats,XTrain,XTest)
        print>>open("pca_summary.txt","w"),"Dimensions:"+str(bestFeats)+ \
            " VarianceExplained:"+str(np.sum(pc.explained_variance_ratio_))
        pickle.dump(pc,open("pc_fold"+str(i)+"_"+str(negFac)+".p","w"))
        foldsTrain.append([XTrain,YTrain])
        foldsTest.append([XTest,YTest])
        covarXInv = np.linalg.inv(XTrain.transpose().dot(XTrain))         
        pickle.dump(covarXInv,open("covarXInv_fold"+str(i)+".p","w"))        
        pickle.dump(XTrain,open("XTrain_fold"+str(i)+".p","w"))            
        
    for c in paramsCs:   ###sending 'c' to each thread
        for g in paramsGamma:
            tName = "thread-"+str(threadCount)
            th = ThreadedCrossVal(tName,foldsTrain,foldsTest,c,g,Kernel)            
            myThreads.append(th)
            activeThreads.append(th)
            """
            submits 20 threads at a time
            """
            
            if len(activeThreads)>=20:
                while(True):
                    if len(activeThreads)>1:
                        for t in activeThreads:
                            if not t.isAlive():
                                activeThreads.pop(activeThreads.index(t))
                    else:
                        break
                     
            th.start()
        
    """
    Makes sure main-thread waits until workers are finished
    """
    threadsCopied = copy.copy(myThreads)
    while True:
        for t in myThreads:
            if not t.isAlive():
                myThreads.pop(myThreads.index(t))
        if len(myThreads)==0:
            print "workers done"
            break
                
    return threadsCopied 

"""
xTrain are descriptors produced after features selection
xi are descriptors of a given compound after features selection
"""
def computeLeverage(covXInvTrain,xi):
    return (xi.transpose()).dot(covXInvTrain).dot(xi)

def computeMaxLeverage(xTrain,covXInvTrain):
    maxLev = np.max([computeLeverage(covXInvTrain,x) for x in xTrain])
    return maxLev                            

"""
checks if a compued is within the applicability domain of a model
decision criteria: if train(hMax) >= test(h) then True else False
"""
def isValidModel(xTest,foldNum,path):
    ###load all XTrains and see if compound falls in atleast 3 of the Mod.dom.
    covXInvTrain = pickle.load(open(path+"covarXInv_fold"+str(foldNum)+".p","r"))
    xTrain = pickle.load(open(path+"XTrain_fold"+str(foldNum)+".p","r"))        
    hMax = computeMaxLeverage(xTrain,covXInvTrain)
    #print "covXInvTrain: "+str(covXInvTrain.shape)
    #print "xTest: "+str(xTest.shape)    
    hX = computeLeverage(covXInvTrain,xTest)
    if hMax > hX:return True
    else:return False    
        

def computeSensitivity(preds,truth):
    posCorr = 0
    for i in range(len(preds)):
        if int(preds[i])==1 and int(truth[i])==1:
            posCorr+=1
            #print "posCorr",posCorr
    return posCorr*1.0/(np.sum(truth))

def computeSpecificity(preds,truth):
    negCorr = 0
    for i in range(len(preds)):
        if int(preds[i])==0 and int(truth[i])==0:
            negCorr+=1
            #print "negCorr",negCorr
    return negCorr*1.0/(len(preds)-np.sum(truth))
        
def compute(trainF,negFac,bestFeat,nFolds,paramsCs,
            paramsGamma,randSplits,Kernel="rbf"):
    
        print "========split pos:neg-->"+str(1)+":"+str(negFac)+"========"
        xTrains,yTrains = ScaleAll(trainF,negFac,
                                bestFeat,nFolds,multipleSplit=randSplits)
        
        threads = CrossValidate(xTrains,yTrains,paramsCs,
                                paramsGamma,Kernel,bestFeat)                                    
 
        trainAucs,models = AssembleResults(threads)                                    
        

        trainAvg,trainStd = ComputeResultsStats(trainAucs)
        
        pickle.dump(trainAvg,open("trainAv_pos1_neg"+str(negFac)+".p","w"))    
        pickle.dump(trainStd,open("trainSd_pos1_neg"+str(negFac)+".p","w"))
        pickle.dump(models,open("models"+str(negFac)+".p","w"))
        
        

def doAll(trainF,testF,paramsCs,paramsGs,bestFeats,Kernel):
    XTr,yTr,zTrainIdx = readCSV(trainF)    
    XTs,yTs,zTestIdx = readCSV(testF)
    
    xAll = np.concatenate((XTr,XTs),axis=0)
    yAll = np.concatenate((yTr,yTs),axis=0)
    print "scaling features"    
    scaler = preprocessing.MinMaxScaler()
    sc = scaler.fit(xAll)
    xAllTr = sc.transform(xAll)
    pickle.dump(sc,open("scaler.p","w"))    
    
    
    XTr = xAllTr[:len(XTr)]; yTr = yAll[:len(yTr)]   
    xTest= xAllTr[len(xAllTr)-len(XTs):]
    yTest = yAll[len(yAll)-len(yTs):]

    print "selecting features"

    XTr,xTest,pc = selectFeatures(bestFeats,XTr,xTest)
    print>>open("pca_summary.txt","w"),"Dimensions:"+str(bestFeats)+ \
    " VarianceExplained:"+str(np.sum(pc.explained_variance_ratio_))
    pickle.dump(pc,open("pcAll.p","w"))

    maxAuc = 0.0; 
    
    for c in paramsCs:
        for g in paramsGs:
            """
            Here, we train and test on the same set ...
            """
            if Kernel=="linear":
                TrainAndTest(XTr,yTr,xTest,yTest,c,Kernel,g=0.01)
                aucTest,clf = TrainAndTest(XTr,yTr,
                                                 xTest,yTest,c,Kernel,g=0.01)
            else:
                aucTest,clf = TrainAndTest(XTr,yTr,xTest,
                                                    yTest,c,Kernel,g)                                        
            if aucTest>maxAuc:
                maxAuc = aucTest
                bestModel = clf
                
            print "best test auc so far:",maxAuc    
            
    print "best test auc:",maxAuc       
    pickle.dump(bestModel,open("bestModelAll.p","w"))


class Params:
    def __init__(self,high,low,size):
        self.high = high
        self.low = low
        self.size = size
    def GetParams(self):
        return np.logspace(self.low,self.high,self.size)

        
if __name__=="__main__":
    
    threadLock = threading.Lock()
    negFac = 3 ### e.g: 2 times more "0" class compared to "1" class during x-val step
    trainF = sys.argv[1] # training set
    testF = sys.argv[2] # test set
    Kernel="rbf"  # applies rbf kernel for svm
    bestFeats = 40 # applices pca and chooses 40 dims 
    paramsCs = Params(-1,1,50).GetParams()
    paramsGamma = [np.power(2,i)/1000.0 for i in range(0,10,2)] #[0.01]
    nFolds = 5    
    randSplit = False
    numDomains = 3
    path = "./" # supply a path to dump the models 
    training = False ### prediction on training or validation set?    
    validationF = sys.argv[3] # external prediction file to predict the labels 
    #compute(trainF,negFac,bestFeats,nFolds,paramsCs,
    #        paramsGamma,randSplit,Kernel)                 
    #predict(testF,negFac,numDomains,path,training)        
    
    predict(validationF,negFac,numDomains,path,training)        
    
    
    
    
