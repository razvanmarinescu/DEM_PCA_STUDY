import scipy.io as sio
import sys

# from DisProgBuilder import *
from evaluationFramework import *
from DEM import *
from aux import *
import numpy as np
from drcDEM import *

minTstaging = -40
maxTstaging = 30
nrStages = 100
params = {'tsStages' : np.linspace(minTstaging, maxTstaging, num=nrStages)}
plotTrajParams = {}
plotTrajParams['xLim'] = (-20, 30)
plotTrajParams['diagStr'] = {CTL: 'CTL', PCA: 'PCA', AD: 'AD'
  , EAR: 'EAR', PER: 'PER', SPA: 'SPA'}
plotTrajParams['diagLabels'] = {CTL: 'CTL', PCA: 'PCA', AD: 'AD'
  , EAR: 'EAR', PER: 'PER', SPA: 'SPA'}
plotTrajParams['diagColors'] = {CTL: 'g', PCA: 'b', AD: 'r'
  , EAR: 'y', PER: 'm', SPA: 'c'}


def main():
  #Function chooser
  runIndex = int(sys.argv[1])
  nrProcesses = int(sys.argv[2])
  modelToRun = int(sys.argv[3])
  params['runIndex'] = runIndex
  params['nrProcesses'] = nrProcesses
  params['modelToRun'] = modelToRun
  mriLarge(runIndex, nrProcesses, modelToRun)

def mriLarge(runIndex, nrProcesses, modelToRun):
  #print matplotlib.get_backend()
  # matData = sio.loadmat('../data/DRC/mriLargeData.mat')
  matData = sio.loadmat('../data/DRC/mriLargeDataSebPaper.mat')
  print(matData.keys())
  # print(adsas)

  np.random.seed(8)

  # expName = 'mriLarge'
  expName = 'mriLargeSebPaper'

  #selectedBiomk = np.array(range(25))
  selectedBiomk = np.array([x for x in range(25) if x not in [19,20,21]])

  # filter AD subjects
  #diagInd = np.array(np.where(matData['diag'] == PCA)[0])
  diag = np.array(np.squeeze(matData['diag']))
  data = matData['selectedData'][:,selectedBiomk]
  labels = matData['selectedLabels']
  labels = [labels[i][0][0] for i in range(len(labels)) if i in selectedBiomk]
  scanTimepts = np.squeeze(matData['scanTimepoint'])
  partCode = np.squeeze(matData['participantCode'])
  ageAtScan = np.squeeze(matData['ageAtScan'])

  params['data'] = data
  params['diag'] = diag
  params['labels'] = labels
  params['scanTimepts'] = scanTimepts
  params['partCode'] = partCode
  params['ageAtScan'] = ageAtScan
  params['lengthScaleFactors'] = np.ones(len(selectedBiomk))
  params['subgroups'] = np.squeeze(matData['subgroupPCA'])

  # find numbers of each subgroup
  for g in [EAR, PER, SPA]:
    print(np.unique(params['partCode'][params['subgroups'] == g]).shape)
  print('CTLs', len(np.unique(params['partCode'][params['diag'] == CTL])))
  # print(adsa)

  nrBiomk = params['data'].shape[1]
  biomkProgDir = DECR * np.ones((nrBiomk,1))
  biomkProgDir[1] = INCR # ventricles are the only ones that increase

  params['data'] = makeAllSameProgDir(params['data'], biomkProgDir, uniformDir)

  nrRows = int(np.sqrt(nrBiomk) * 0.95) 
  nrCols = int(np.ceil(float(nrBiomk) / nrRows)) 
  assert(nrRows * nrCols >= nrBiomk)

  plotTrajParams['axisPos'] = [0.06, 0.06, 0.9, 0.6]
  plotTrajParams['legendPos'] = (0.5, 1.5)
  plotTrajParams['legendPosSubplotsPcaAd'] = (0.5, 0)
  plotTrajParams['legendCols'] = 3
  plotTrajParams['nrRows'] = nrRows
  plotTrajParams['nrCols'] = nrCols
  plotTrajParams['trajAlignMaxWinSize'] = (900, 850)
  plotTrajParams['trajPcaAdMaxWinSize'] = (1200, 900)
  plotTrajParams['axisHeightChopRatio'] = 0.8
  plotTrajParams['trajSubfigXlim'] = [-40,30]
  plotTrajParams['trajSubfigYlim'] = [-5, 5]
  plotTrajParams['expName'] = expName
  plotTrajParams['stagingHistNrBins'] = 20

  params['plotTrajParams'] = plotTrajParams

  params['runPartStd'] = ['R', 'R']  # [gaussian fit, aligner, plot, staging]
  params['runPartMain'] = ['R', 'R', 'R'] # std DEM, stage subj, plot fitted params (trajectorie)
  params['runPartDirDiag'] = ['I', 'I', 'I']
  params['runPartStaging'] = ['I', 'I', 'I']
  params['runPartDiffDiag'] = ['I', 'I', 'I']
  params['runPartSubgroupStd'] = ['R', 'R']
  params['runPartSubgroupMain'] = ['I', 'I', 'I', 'I']
  params['runPartPcaAd'] = ['L']

  params['masterProcess'] = runIndex == 0

  if params['masterProcess']:
    params['runPartStd'] = ['L', 'L']  # [gaussian fit, aligner, plot, staging]
    params['runPartMain'] = ['R', 'L', 'L']  # [stdDEM, stageSubj, plotFittedParams (trajectorie)]
    params['runPartDirDiag'] = ['R', 'R', 'I']
    params['runPartStaging'] = ['L', 'L', 'I']
    params['runPartDiffDiag'] = ['R', 'R', 'I']
    params['runPartSubgroupStd'] = ['L', 'R']
    params['runPartSubgroupMain'] = ['R', 'I', 'R', 'R']
    params['runPartPcaAd'] = ['L']


  runAllExpFunc = runAllExpDRC
  modelNames, res = runModels(params, expName, modelToRun, runAllExpFunc)

  if params['masterProcess']:
    printResDRC(modelNames, res)


def mriTiny(runIndex, nrProcesses, modelToRun):
  # contains only 3 biomk: occip, parietal, temporal
  matData = sio.loadmat('../data/pcaData.mat')
  print(matData.keys())

  np.random.seed(8)

  expName = 'mriTiny'

  # ATTENTION: The following indices use MATLAB 0-indexing, subtract 1 when doing in python

  MOG = [85, 86]
  IOG = [71, 72]
  SOG = [133, 134]
  OFuG = [99, 100]
  OCCIPITAL = MOG + IOG + SOG + OFuG

  MTG = [95, 96]
  ITG = [73, 74]
  STG = [137, 138]
  TMP = [139, 140]
  TEMPORAL = STG + ITG + MTG

  SPL = [135, 136]
  PCu = [107, 108]
  AnG = [53, 54]
  SMG = [131, 132]
  PARIETAL = SPL + PCu + AnG + SMG

  biomkClust = [OCCIPITAL, TEMPORAL, PARIETAL]
  biomkClust = [[y - 1 for y in x] for x in
                biomkClust]  # subtract 1 for converting to python 0-indexing from matlab indexing
  labels = ['Occipital', 'Temporal', 'Parietal']

  nrBiomk = len(biomkClust)

  # filter AD subjects
  # diagInd = np.array(np.where(matData['diag'] == PCA)[0])
  diag = np.array(np.squeeze(matData['pcaDiag']))
  data = np.zeros((matData['allData'].shape[0], nrBiomk))
  nrSubj = diag.shape[0]
  allLabels = [x[0] for x in matData['allLabels'][0]]
  # print allLabels
  for b in range(nrBiomk):
    # print "biomk %d" % b
    data[:, b] = np.sum(np.reshape(matData['allData'][:, biomkClust[b]], (nrSubj, -1)), axis=1)
    # print "labels:", labels[b]

  scanTimepts = np.squeeze(matData['scanTimepoint'])
  partCode = np.squeeze(matData['participantCode'])
  ageAtScan = np.squeeze(matData['ageAtScan'])

  params['data'] = data
  params['diag'] = diag
  params['labels'] = labels
  params['scanTimepts'] = scanTimepts
  params['partCode'] = partCode
  params['ageAtScan'] = ageAtScan

  biomkProgDir = DECR * np.ones((nrBiomk, 1))

  params['data'] = makeAllSameProgDir(params['data'], biomkProgDir, uniformDir)

  nrRows = int(np.sqrt(nrBiomk) * 0.95)
  nrCols = int(np.ceil(float(nrBiomk) / nrRows))
  assert (nrRows * nrCols >= nrBiomk)

  plotTrajParams['axisPos'] = [0.06, 0.1, 0.9, 0.75]
  plotTrajParams['legendPos'] = (0.5, 1.1)
  plotTrajParams['legendPosSubplotsPcaAd'] = (0.5, 0)
  plotTrajParams['legendCols'] = 4
  plotTrajParams['nrRows'] = nrRows
  plotTrajParams['nrCols'] = nrCols
  plotTrajParams['trajAlignMaxWinSize'] = (900, 700)
  plotTrajParams['trajPcaAdMaxWinSize'] = (1200, 500)
  plotTrajParams['axisHeightChopRatio'] = 0.8
  plotTrajParams['expName'] = expName

  params['plotTrajParams'] = plotTrajParams

  runAllExpFunc = runAllExpDRC
  modelNames, res = runModels(params, expName, modelToRun, runAllExpFunc)

if __name__ == '__main__':
  main()

