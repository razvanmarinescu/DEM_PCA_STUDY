import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from env import *

from evaluationFramework import *
from DEM import *
from aux import *
import numpy as np
from drcDEM import *


minTstaging = -20
maxTstaging = 30
nrStages = 200
params = {'tsStages': np.linspace(minTstaging, maxTstaging, num=nrStages)}
plotTrajParams = {}
plotTrajParams['xLim'] = (-10, 20)
plotTrajParams['diagStr'] = {CTL: 'Controls', PCA: 'PCA', AD: 'AD'
  , EAR: 'EAR', PER: 'PER', SPA: 'SPA'}
plotTrajParams['diagLabels'] = {CTL: 'Controls', PCA: 'PCA', AD: 'AD'
  , EAR: 'EAR', PER: 'PER', SPA: 'SPA'}
plotTrajParams['diagColors'] = {CTL: 'g', PCA: 'b', AD: 'r'
  , EAR: 'y', PER: 'm', SPA: 'c'}

plotTrajParams['SubfigGroupCompWinSize'] = [1200, 600]
plotTrajParams['TrajSubfigWinSize'] = [1800, 800]

plotTrajParams['trajAlignMaxWinSize'] = (600, 500)
plotTrajParams['trajSubfigMaxWinSize'] = (1400, 900)
plotTrajParams['trajPcaAdMaxWinSize'] = (1200, 600)
plotTrajParams['axisHeightChopRatio'] = 0.8
plotTrajParams['trajSubfigXlim'] = [-15, 15]
plotTrajParams['trajSubfigYlim'] = [-40, 2]


def cognitive_dem(runIndex, nrProcesses, modelToRun):
    expName = 'cogBmConf'

    df = pd.read_csv('../data/DRC/visible/180903-final_sheet_CsfFilt_MriFilt.csv')
    # df = df[np.in1d(df['o_diagnosis'], ['PCA', 'Control'])]
    df['p_date_at_ax'] = pd.to_datetime(df['p_date_at_ax'], errors='coerce')

    labels = ['p_a_cancel_nmiss', 'p_digitspan_b',
              'p_digitspan_f12', 'p_dotcount_ncorrect',
              'p_mmse', 'p_srmtf', 'p_srmtw',
              'p_shapedis', 'p_objectdecision',
              'p_fragmentedletters', 'p_gda_add_12', 'p_gda_sub_12',
              'p_gda_24']

    paperLabels = ['A cancel.', 'Digit span', 'Digit span (F)',
    'Dot Count N cor.', 'MMSE', 'SRMT faces', 'SRMT words',
    'Shape dis.', 'Obj. decision', 'Frag. letters', 'GDA addition',
    'GDA subtract.', 'GDA total']

    df = df.dropna(subset=['age'])
    df.reset_index(inplace=True)

    # take out all subjects who did not have AD-confirmation through mollecular/amyloid biomarkers
    bmTable = pd.read_csv('../data/DRC/visible/BM_for_Raz.csv')
    bmTable = bmTable[bmTable.final_conf == 1]
    drcCodesWithConf = bmTable.o_drc_code
    indsWithAdConf = np.logical_or(np.in1d(df.o_drc_code, drcCodesWithConf), df.o_diagnosis == 'Control')

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
      # print(bmTable[bmTable.final_conf == 1])
      print(df.o_drc_code)
      print(drcCodesWithConf)

      print('df.o_drc_code = CTL\n', df.o_drc_code[df.o_diagnosis == 'Control'])
      print('bmTable[drcCodesWithConf] = CTL\n', bmTable.o_drc_code[bmTable.o_diagnosis == 'Control'])
      # print(adsa)

      print('df[o_diagnosis]', df['o_diagnosis'][indsWithAdConf])

    # print(adsa)

    print('df.shape', df.shape)
    df = df[indsWithAdConf]
    df.reset_index(inplace=True)
    print('df.shape', df.shape)
    print('df[o_diagnosis]', df['o_diagnosis'][df['o_diagnosis'] == 'Control'])
    # print(adsa)

    df[labels] = df[labels].apply(pd.to_numeric, errors='coerce')

    print('df[labels]\n', df[labels])

    df[labels].to_csv('cogDem_bmConf.csv')
    # print(adsa)

    data = df[labels].values
    nrSubj = df.shape[0]
    diagStr = df['o_diagnosis']
    scanTimepts = df['ncf_vid'].values
    partCode = df['o_drc_code'].values
    # partCode = LabelEncoder().fit_transform(partCode)
    ageAtVisit = df['age'].values

    diag = PCA * np.ones(diagStr.shape[0])
    diag[diagStr == 'Control'] = CTL
    diag[diagStr == 'AD'] = AD

    unqPart = np.unique(partCode)
    nrUnqPart = unqPart.shape[0]
    for p in range(nrUnqPart):
      ageCurrPart = ageAtVisit[unqPart[p] == partCode]
      scanTimepts[unqPart[p] == partCode] = np.argsort(np.argsort(ageCurrPart)) + 1

    print('data', data)
    print('nrSubj', nrSubj)
    print('diag', diag)
    print('scanTimepts', scanTimepts)
    print('partCode', partCode)
    print('ageAtVisit', ageAtVisit)


    nrBiomk = data.shape[1]
    for b in range(nrBiomk):
      print('--------', b, labels[b])
      ctlData = data[diag == CTL,b]
      pcaData = data[diag == PCA,b]

      meanCtl = np.nanmean(ctlData)
      stdCtl = np.nanstd(ctlData)

      print('ctlData', ctlData)
      print('meanCtl', meanCtl)
      print('stdCtl', stdCtl)

      data[:,b] = (data[:,b] - meanCtl) / stdCtl

      ctlData = data[diag == CTL,b]
      pcaData = data[diag == PCA,b]

      print('CTL', np.nanmean(ctlData), np.nanstd(ctlData))
      print('PCA', np.nanmean(pcaData), np.nanstd(pcaData))

    biomkProgDir = DECR * np.ones((nrBiomk, 1))
    biomkProgDir[0] = INCR

    data = makeAllSameProgDir(data, biomkProgDir, uniformDir)

    # print(adsas)

    params['data'] = data
    params['diag'] = diag
    params['labels'] = paperLabels
    params['scanTimepts'] = scanTimepts
    params['partCode'] = partCode
    params['ageAtScan'] = ageAtVisit
    params['lengthScaleFactors'] = 2 * np.ones(nrBiomk)
    params['noiseZfactor'] = 1

    # print(data)
    # print(diag)
    # print(data[diag == AD, :])
    # print(adsa)

    nrRows = 3
    nrCols = 5
    assert (nrRows * nrCols >= nrBiomk)

    plotTrajParams['axisPos'] = [0.1, 0.1, 0.9, 0.75]
    plotTrajParams['legendPos'] = (0.5, 1.25)
    plotTrajParams['legendPosSubplotsPcaAd'] = (0.5, 0)
    plotTrajParams['legendCols'] = 3
    plotTrajParams['nrRows'] = nrRows
    plotTrajParams['nrCols'] = nrCols
    plotTrajParams['expName'] = expName
    plotTrajParams['stagingHistNrBins'] = 10
    plotTrajParams['kernelWidthFactor'] = 1.0 / 3

    params['plotTrajParams'] = plotTrajParams

    params['runPartStd'] = ['R', 'R']  # [gaussian fit, aligner]
    params['runPartMain'] = ['R', 'R', 'L']  # std DEM, plotTrajectories, stageSubj
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

    runAllExpFunc = runAllExpDRCCog
    modelNames, res = runModels(params, expName, modelToRun, runAllExpFunc)
    input("Press Enter to continue...")

    if params['masterProcess']:
      printResDRC(modelNames, res)

    print(df.columns.values)


def runAllExpDRCCog(params, expName, dpmBuilder):
  """ runs all experiments"""

  res = {}

  # run if this is the master process or nrProcesses is 1
  unluckyProc = (np.mod(params['currModel'] - 1, params['nrProcesses']) == params['runIndex'] - 1)
  unluckyOrNoParallel = unluckyProc or (params['nrProcesses'] == 1) or params['masterProcess']

  '''PCA DEM'''

  params['patientID'] = PCA
  params['plotTrajParams']['modelCol'] = 'b' # blue
  # params['excludeID'] = [CTL, AD]
  params['excludeID'] = [AD]
  params['excludeXvalidID'] = [AD]
  params['excludeStaging'] = [CTL, AD]
  params['anchorID'] = CTL
  expNamePCA = '%sPCA' % expName
  params['stagingInformPrior'] = False
  if unluckyOrNoParallel:
    dpmObj, res['stdPCA'] = runStdDPM(params, expNamePCA, dpmBuilder, params['runPartMain'])
    pass


  '''AD DEM'''

  params['patientID'] = AD
  params['plotTrajParams']['modelCol'] = 'r'  # blue
  # params['excludeID'] = [CTL, PCA]
  params['excludeID'] = [PCA]
  params['excludeXvalidID'] = [PCA]
  params['excludeStaging'] = [CTL, PCA]
  params['anchorID'] = CTL
  params['stagingInformPrior'] = True
  expNameAD = '%sAD' % expName
  if unluckyOrNoParallel:
    _, res['stdAD'] = runStdDPM(params, expNameAD, dpmBuilder, params['runPartMain'])


  '''Both PCA and AD'''

  # expNameDD = '%sDD' % expName
  # res['diffDiagStats'] = evalDiffDiagDRC(dpmBuilder, expName, params)

  # print(res)
  # print(adad)

  outFolder = 'matfiles/%s' % expNamePCA
  trajSubplotsFigName = '%s/subplotsPcaAd' % outFolder
  # plot subfigures with each biomarker in turn that show two traj: PCA vs AD + bootstraps
  if params['runPartPcaAd'][0] == 'R':
    fig = plotSubfigGroupComp([res['stdPCA'], res['stdAD']], [PCA, AD], params['plotTrajParams'],
                              params['labels'], groupLabels=['PCA', 'AD'])
    fig.savefig(trajSubplotsFigName + '.png', dpi=300)
    fig.savefig(trajSubplotsFigName + '.pdf', dpi=300)

    ttestsGroups([res['stdPCA'], res['stdAD']], params['labels'],
      groupLabels=['PCA', 'AD'])

    ttestsRegions(res['stdPCA'], params['labels'], 'PCA')
    ttestsRegions(res['stdAD'], params['labels'], 'AD')

  # test the EBM on Silvia's subgroups independently
  # runSubgroupsPCA(dpmBuilder, expName, params)
  # perform cross-validation and test the continuum hypothesis
  # runSubgroupsPCA_CV(ebmParams, suffix, fittingFunc, EBMfunc, ebmParams.runPart.subgroupsCV);

  return res


def main():
  print('---------------------------')
  runIndex = int(sys.argv[1])
  nrProcesses = int(sys.argv[2])
  modelToRun = int(sys.argv[3])
  params['runIndex'] = runIndex
  params['nrProcesses'] = nrProcesses
  params['modelToRun'] = modelToRun
  cognitive_dem(runIndex, nrProcesses, modelToRun)

if __name__ == '__main__':
    main()