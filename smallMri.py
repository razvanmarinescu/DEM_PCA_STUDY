import scipy.io as sio
import sys

# from DisProgBuilder import *
from evaluationFramework import *
from DEM import *
from aux import *
import numpy as np
from drcDEM import *

minTstaging = -20
maxTstaging = 30
nrStages = 200
params = {'tsStages' : np.linspace(minTstaging, maxTstaging, num=nrStages)}
plotTrajParams = {}
plotTrajParams['xLim'] = (-10, 20)
plotTrajParams['diagStr'] = {CTL: 'Controls', PCA: 'PCA', AD: 'AD'
  , EAR: 'EAR', PER: 'PER', SPA: 'SPA'}
plotTrajParams['diagLabels'] = {CTL: 'Controls', PCA: 'PCA', AD: 'AD'
  , EAR: 'EAR', PER: 'PER', SPA: 'SPA'}
plotTrajParams['diagColors'] = {CTL: 'g', PCA: 'b', AD: 'r'
  , EAR: 'y', PER: 'm', SPA: 'c'}

plotTrajParams['SubfigGroupCompWinSize'] = [1200, 600]
plotTrajParams['TrajSubfigWinSize'] = [1800, 1000]

plotTrajParams['trajAlignMaxWinSize'] = (600, 500)
plotTrajParams['trajSubfigMaxWinSize'] = (1400, 900)
plotTrajParams['trajPcaAdMaxWinSize'] = (1200, 600)
plotTrajParams['axisHeightChopRatio'] = 0.8
# plotTrajParams['trajSubfigXlim'] = [-10, 20]
plotTrajParams['trajSubfigXlim'] = [-15, 20]
plotTrajParams['trajSubfigYlim'] = [-10, 2]


def main():
  print('---------------------------')
  runIndex = int(sys.argv[1])
  nrProcesses = int(sys.argv[2])
  modelToRun = int(sys.argv[3])
  params['runIndex'] = runIndex
  params['nrProcesses'] = nrProcesses
  params['modelToRun'] = modelToRun
  mriSmall(runIndex, nrProcesses, modelToRun)

def mriSmall(runIndex, nrProcesses, modelToRun):
  # contains only around 10-11 biomarkers: whole, centricles, hippo, ento, ante cingu, post. cing., orbital, occip, temporal, frontal, parietal
  # matData = sio.loadmat('../data/DRC/pcaData.mat')
  matData = sio.loadmat('pcaDataPaperSeb.mat')
  print(matData.keys())


  np.random.seed(8)

  # expName = 'mriSmall'
  expName = 'mriSmallSebPaper'

  # ATTENTION: The following indices use MATLAB 0-indexing, subtract 1 when doing in python

  WHOLE = [145]
  VENTS = [146]
  # Amy = [10,11] % signal in AD only
  HIPPO = [25, 26]  # relative to allLabels (includes ID & Diag)
  ENTORHINAL = [61, 62]

  # ACgG = [47,48]
  # PCgG = [105, 106]
  # MCgG = [79,80]
  # CINGULATE = ACgG + PCgG + MCgG

  ANTE_CINGULATE = [47, 48]
  POST_CINGULATE = [105, 106]

  AOrG = [51, 52]
  POrG = [117, 118]
  MOrG = [87, 88]
  LOrG = [77, 78]
  ORBITAL = AOrG + POrG + MOrG + LOrG

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

  MFG = [83, 84]
  IFG = [101, 102, 103, 104, 141, 142]  # sum these all up
  SFG = [93, 94, 127, 128]  # added medial segment also
  FRONTAL = SFG + IFG + MFG

  SPL = [135, 136]
  PCu = [107, 108]
  AnG = [53, 54]
  SMG = [131, 132]
  PARIETAL = SPL + PCu + AnG + SMG

  biomkClust = [WHOLE, VENTS, HIPPO, ENTORHINAL, ANTE_CINGULATE, POST_CINGULATE  # ORBITAL
    , OCCIPITAL, TEMPORAL, FRONTAL, PARIETAL]
  biomkClust = [[y - 1 for y in x] for x in
                biomkClust]  # subtract 1 for converting to python 0-indexing from matlab indexing
  labels = ['Whole Brain', 'Ventricles', 'Hippocampus', 'Entorhinal', 'Ante. Cingulate', 'Post. Cingulate', 'Occipital',
            'Temporal', 'Frontal', 'Parietal']

  selectedBiomk = np.array([x for x in range(len(labels)) if x not in [4,5]])

  assert (len(biomkClust) == len(labels))

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
    print("labels:", labels[b])
    print('allLabels', [allLabels[i] for i in biomkClust[b]])
    print('-------------------')

  # print(ads)

  scanTimepts = np.squeeze(matData['scanTimepoint'])
  partCode = np.squeeze(matData['participantCode'])
  ageAtScan = np.squeeze(matData['ageAtScan'])

  print('scanTimepts', scanTimepts)
  # print(adsa)

  # remove patrCodes [8, 145] because they have abnormal increase in hippocampus.
  # filterMask = ~np.in1d(partCode, [8,145])
  # print(np.sum(~filterMask))
  # print(asda)

  nrBiomk = len(selectedBiomk)
  params['data'] = data[:,selectedBiomk]
  params['diag'] = diag
  params['labels'] = [labels[i] for i in selectedBiomk]
  params['scanTimepts'] = scanTimepts
  params['partCode'] = partCode
  params['ageAtScan'] = ageAtScan
  params['lengthScaleFactors'] = np.ones(nrBiomk)
  # params['lengthScaleFactors'][4] = 1.7 # for biomk with less signal or for

  biomkProgDir = DECR * np.ones((nrBiomk, 1))
  biomkProgDir[1] = INCR  # ventricles are the only ones that increase

  params['data'] = makeAllSameProgDir(params['data'], biomkProgDir, uniformDir)

  # nrRows = int(np.sqrt(nrBiomk) * 0.95)
  # nrCols = int(np.ceil(float(nrBiomk) / nrRows))
  nrRows = 2
  nrCols = 4
  assert (nrRows * nrCols >= nrBiomk)

  plotTrajParams['axisPos'] = [0.1, 0.1, 0.9, 0.75]
  plotTrajParams['legendPos'] = (0.5, 1.25)
  plotTrajParams['legendPosSubplotsPcaAd'] = (0.5, 0)
  plotTrajParams['legendCols'] = 3
  plotTrajParams['nrRows'] = nrRows
  plotTrajParams['nrCols'] = nrCols
  plotTrajParams['expName'] = expName
  plotTrajParams['stagingHistNrBins'] = 10
  plotTrajParams['kernelWidthFactor'] = 1.0/4.5
  params['noiseZfactor'] = 1

  params['plotTrajParams'] = plotTrajParams

  params['runPartStd'] = ['R', 'R']  # [gaussian fit, aligner]
  params['runPartMain'] = ['R', 'R', 'L'] # std DEM, plotTrajectories, stageSubj
  params['runPartDirDiag'] = ['I', 'I', 'I']
  params['runPartStaging'] = ['I', 'I', 'I']
  params['runPartDiffDiag'] = ['I', 'I', 'I']
  params['runPartSubgroupStd'] = ['R', 'R']
  params['runPartSubgroupMain'] = ['R', 'R', 'R', 'R']
  params['runPartPcaAd'] = ['R']

  params['masterProcess'] = runIndex == 0

  if params['masterProcess']:
    params['runPartStd'] = ['L', 'L']  # [gaussian fit, aligner]
    params['runPartMain'] = ['R', 'R', 'L']  # [stdDEM, plotTrajectories, stageSubj]
    params['runPartDirDiag'] = ['R', 'R', 'I']
    params['runPartStaging'] = ['L', 'L', 'I']
    params['runPartDiffDiag'] = ['R', 'R', 'I']
    params['runPartSubgroupStd'] = ['L', 'R']
    params['runPartSubgroupMain'] = ['R', 'I', 'R', 'R']
    params['runPartPcaAd'] = ['R']

  runAllExpFunc = runAllExpDRC
  modelNames, res = runModels(params, expName, modelToRun, runAllExpFunc)
  input("Press Enter to continue...")

  if params['masterProcess']:
    printResDRC(modelNames, res)

if __name__ == '__main__':
  main()

