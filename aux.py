import numpy as np
from env import *
from sklearn.metrics import *
from sklearn.model_selection import *

def makeAllSameProgDir(data, biomkProgDir, direction):
  nrBiomk = data.shape[1]
  for b in range(nrBiomk):
    if biomkProgDir[b] != direction and biomkProgDir[b] != 0:
      data[:,b] *= -1

  return data

def arrayToStrNoBrackets(a):
  return '\n' + '\n'.join(' '.join('%.4f' % cell for cell in row) for row in a) + '\n'

def allocateRunIndicesToProcess(nrExperiments, nrProcesses, runIndex):
  indicesCurrProcess = [x for x in range(nrExperiments)][(runIndex-1):nrExperiments:nrProcesses]
  return indicesCurrProcess

def findIdealThresh(stagingProb, diag):

  nrSubj, nrStages = stagingProb.shape

  diagClasses = np.sort(np.unique(diag))
  assert(len(diagClasses) == 2)
  assert(diagClasses[0] == CTL)
  patientID = diagClasses[1]
  balAccuracies = np.zeros(nrStages)
  confMat = np.zeros((nrStages, 2,2))
  for th in range(nrStages):
    predDiag = CTL * np.ones((nrSubj,1))
    probControl = np.sum(stagingProb[:,0:th], axis=1)
    assert(predDiag.shape[0] == probControl.shape[0])
    predDiag[probControl < 0.5] = patientID

    confMat[th,:,:] = confusion_matrix(diag, predDiag)
    (balAccuracies[th],_,_,_) = getSensitivitySpecificity(confMat[th,:,:])


  optimThresh = np.argmax(balAccuracies)
  #print('findIdealTHresh', confMat[optimThresh,:,:], balAccuracies[optimThresh], balAccuracies)

  return optimThresh


def findDiagStatsGivenTh(stagingProb, diag, thresh):
  nrSubj, nrStages = stagingProb.shape

  diagClasses = np.sort(np.unique(diag))
  assert (len(diagClasses) == 2)
  assert (diagClasses[0] == CTL)
  patientID = diagClasses[1]
  balAccuracies = np.zeros(nrStages)

  predDiag = CTL * np.ones((nrSubj, 1))
  probControl = np.sum(stagingProb[:, 0:thresh], axis=1)
  predDiag[probControl < 0.5] = patientID

  confMat = confusion_matrix(diag, predDiag)
  balancedAccuracy, sensitivity, specificity, accuracy = getSensitivitySpecificity(confMat)

  nrDiag1 = np.sum(diag == diagClasses[0])
  nrDiag2 = np.sum(diag == diagClasses[1])

  # print(diag, predDiag)
  # print(adsa)
  #print('diffDiagStats', confMat, accuracy, sensitivity, specificity, balancedAccuracy, nrDiag1, nrDiag2)

  return balancedAccuracy, sensitivity, specificity, accuracy, nrDiag1, nrDiag2

def getSensitivitySpecificity(confMat):

  tp = confMat[0, 0]
  fp = confMat[0, 1]
  fn = confMat[1, 0]
  tn = confMat[1, 1]

  accuracy = (tp + tn) / (tp + tn + fp + fn)
  sensitivity = tp / (tp + fn)
  specificity = tn / (tn + fp)
  balancedAccuracy = 0.5 * (tp / (tp + fp) + tn / (tn + fn))

  return balancedAccuracy, sensitivity, specificity, accuracy
