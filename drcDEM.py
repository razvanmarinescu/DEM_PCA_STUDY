import scipy.io as sio
import sys

# from DisProgBuilder import *
from evaluationFramework import *
from DEM import *
from aux import *
import numpy as np


def runAllExpDRC(params, expName, dpmBuilder):
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
  params['anchorID'] = PCA
  expNamePCA = '%sPCA' % expName
  params['stagingInformPrior'] = False
  if unluckyOrNoParallel:
    dpmObjPCA, res['stdPCA'] = runStdDPM(params, expNamePCA, dpmBuilder, params['runPartMain'])
    # res['outFolder'] = dpmObjPCA.params['outFolder']


  res['dirDiagStatsPCA'] = evalDirDiag(dpmBuilder, expNamePCA, params)
  res['upEqStagesPercPCA'], res['pFUgrBLAllPCA'], res['timeDiffHardPCA'], res['timeDiffSoftPCA'] \
    = evalStaging(dpmBuilder, expNamePCA, params)

  # input("Press Enter to continue...")

  '''AD DEM'''

  params['patientID'] = AD
  params['plotTrajParams']['modelCol'] = 'r'  # blue
  # params['excludeID'] = [CTL, PCA]
  params['excludeID'] = [PCA]
  params['excludeXvalidID'] = [PCA]
  params['excludeStaging'] = [CTL, PCA]
  params['anchorID'] = AD
  params['stagingInformPrior'] = True
  expNameAD = '%sAD' % expName
  if unluckyOrNoParallel:
    dpmObjAD, res['stdAD'] = runStdDPM(params, expNameAD, dpmBuilder, params['runPartMain'])

  res['dirDiagStatsAD'] = evalDirDiag(dpmBuilder, expNameAD, params)
  res['upEqStagesPercAD'], res['pFUgrBLAllAD'], res['timeDiffHardAD'], res['timeDiffSoftAD'] \
    =  evalStaging(dpmBuilder, expNameAD, params)

  # input("Press Enter to continue...")


  # import pdb
  # pdb.set_trace()

  '''Both PCA and AD'''

  expNameDD = '%sDD' % expName
  res['diffDiagStats'] = evalDiffDiagDRC(dpmBuilder, expName, params)

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

  if unluckyOrNoParallel:
    dpmObjPCA.params['plotTrajParams']['xLabel'] = 'Years since $t_0$'


    print(dpmObjPCA.params['labels'])
    # asda
    idxBiomkToPlot = [4,7]
    fig, lgd = plotTrajAlignHistKeir(res['stdPCA']['ts'], res['stdPCA']['xsZ'], dpmObjPCA.params['labels'], dpmObjPCA.params['plotTrajParams'],
      dpmObjPCA.diag, res['stdPCA']['maxLikStages'], idxBiomkToPlot, xLim = dpmObjPCA.params['plotTrajParams']['trajSubfigXlim'], yLim=dpmObjPCA.params['plotTrajParams']['trajSubfigYlim'])
    fig.show()
    fig.savefig(dpmObjPCA.trajAlign + '_PCA_step1.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

    idxBiomkToPlot = [4,7,2,3]
    fig, lgd = plotTrajAlignHistKeir(res['stdPCA']['ts'], res['stdPCA']['xsZ'], dpmObjPCA.params['labels'], dpmObjPCA.params['plotTrajParams'],
      dpmObjPCA.diag, res['stdPCA']['maxLikStages'], idxBiomkToPlot, xLim = dpmObjPCA.params['plotTrajParams']['trajSubfigXlim'], yLim=dpmObjPCA.params['plotTrajParams']['trajSubfigYlim'])
    fig.show()
    fig.savefig(dpmObjPCA.trajAlign + '_PCA_step2.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

    idxBiomkToPlot = [0,1,2,3,4,5,6,7]
    fig, lgd = plotTrajAlignHistKeir(res['stdPCA']['ts'], res['stdPCA']['xsZ'], dpmObjPCA.params['labels'], dpmObjPCA.params['plotTrajParams'],
      dpmObjPCA.diag, res['stdPCA']['maxLikStages'], idxBiomkToPlot, xLim = dpmObjPCA.params['plotTrajParams']['trajSubfigXlim'], yLim=dpmObjPCA.params['plotTrajParams']['trajSubfigYlim'])
    fig.show()
    fig.savefig(dpmObjPCA.trajAlign + '_PCA_step3.png', bbox_extra_artists=(lgd,), bbox_inches='tight')



    print(dpmObjAD.params['labels'])
    # asda
    idxBiomkToPlot = [4,7]
    fig, lgd = plotTrajAlignHistKeir(res['stdAD']['ts'], res['stdAD']['xsZ'], dpmObjAD.params['labels'], dpmObjAD.params['plotTrajParams'],
      dpmObjAD.diag, res['stdAD']['maxLikStages'], idxBiomkToPlot, xLim = dpmObjAD.params['plotTrajParams']['trajSubfigXlim'], yLim=dpmObjAD.params['plotTrajParams']['trajSubfigYlim'])
    fig.show()
    fig.savefig(dpmObjAD.trajAlign + '_AD_step1.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

    idxBiomkToPlot = [4,7,2,3]
    fig, lgd = plotTrajAlignHistKeir(res['stdAD']['ts'], res['stdAD']['xsZ'], dpmObjAD.params['labels'], dpmObjAD.params['plotTrajParams'],
      dpmObjAD.diag, res['stdAD']['maxLikStages'], idxBiomkToPlot, xLim = dpmObjAD.params['plotTrajParams']['trajSubfigXlim'], yLim=dpmObjAD.params['plotTrajParams']['trajSubfigYlim'])
    fig.show()
    fig.savefig(dpmObjAD.trajAlign + '_AD_step2.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

    idxBiomkToPlot = [0,1,2,3,4,5,6,7]
    fig, lgd = plotTrajAlignHistKeir(res['stdAD']['ts'], res['stdAD']['xsZ'], dpmObjAD.params['labels'], dpmObjAD.params['plotTrajParams'],
      dpmObjAD.diag, res['stdAD']['maxLikStages'], idxBiomkToPlot, xLim = dpmObjAD.params['plotTrajParams']['trajSubfigXlim'], yLim=dpmObjAD.params['plotTrajParams']['trajSubfigYlim'])
    fig.show()
    fig.savefig(dpmObjAD.trajAlign + '_AD_step3.png', bbox_extra_artists=(lgd,), bbox_inches='tight')





  # test the EBM on Silvia's subgroups independently
  # runSubgroupsPCA(dpmBuilder, expName, params)
  # perform cross-validation and test the continuum hypothesis
  # runSubgroupsPCA_CV(ebmParams, suffix, fittingFunc, EBMfunc, ebmParams.runPart.subgroupsCV);

  return res

def ttestsRegions(res, labels, diagStr):
  '''
  Performs unpaired t-tests between different brain ROIs, same groups

  '''


  (nrSamples, nrPoints, nrBiomk) = res['xsSamples'].shape

  timepts = [-10, 0, 10]
  nrTimepts = len(timepts)

  # define a cell array
  groupTs = np.empty(nrBiomk, dtype=object)
  groupXs = np.empty(nrBiomk, dtype=object)

  fsPred = np.empty((nrBiomk, nrTimepts), dtype=object)

  tStat = np.zeros((nrBiomk, nrBiomk, nrTimepts), float)
  pVal = np.zeros((nrBiomk, nrBiomk, nrTimepts), float)

  for b in range(nrBiomk):
    # print("biomk %d %s" % (b, labels[b]))

    goodSamples = np.logical_not(res['badSamples'][:, b])
    groupTs[b] = res['tsSamples'][goodSamples,:,b]
    groupXs[b] = res['xsSamples'][goodSamples,:,b]

    assert groupTs[b].shape[0] == groupXs[b].shape[0]

    # print(groupTs[b, g].shape, groupXs[b, g].shape)
    nrGoodSamples = groupTs[b].shape[0]
    fsSamples = [interpolate.interp1d(groupTs[b][s,:], groupXs[b][s,:],
      kind='linear', fill_value='extrapolate') for s in range(nrGoodSamples)]

    for t in range(len(timepts)):
      # for three different timepoints
      fsPred[b,t] = [ fsSamples[s](timepts[t]) for s in range(len(fsSamples))]

  for b1 in range(nrBiomk):
    for b2 in range(nrBiomk):
      for t in range(len(timepts)):
        # for three different timepoints
        tStat[b1, b2, t], pVal[b1, b2, t] = scipy.stats.ttest_ind(
          fsPred[b1, t], fsPred[b2, t])

        # print('timept=%d: tStats=%f, pVal=%f' % (timepts[t], tStat[b, pairIndex, t],
        #   pVal[b, pairIndex, t]))

  createLatexRegions(labels, timepts, tStat, pVal, diagStr)


def createLatexRegions(labels, timepts, tStat, pVal, diagStr):

  text = ''
  nrBiomk, _, nrTimepts = pVal.shape
  alpha = 0.05
  corrTerm = (nrBiomk * (nrBiomk-1))/2


  for t in range(len(timepts)):
    text += '\n\n %% %s T = %d \n\n' % (diagStr, timepts[t])
    text += r'''\begin{table}[H]
\centering
\begin{tabular}{c |''' + ''.join(['c ' for _ in range(nrBiomk)]) + '}\n'

    text += ' & '.join(['Region'] + labels) + '\\\\\n\hline \n'

    for b1 in range(nrBiomk):
      text += labels[b1]

      for b2 in range(nrBiomk):
        if b1 > b2:
          text += ' & %.2e' % (pVal[b1, b2, t])
          if pVal[b1, b2, t] < alpha/corrTerm:
            text += '*'
        else:
          text += ' & -'

      # endline
      text+= '\\\\\n'

    captionText = 'P-values for comparison between the volumes of different brain ' \
                  'regions of %s subjects at timePoint %d. (*) Statistically ' \
                  'significant p-values that survived bonferroni correction.'\
                  % (diagStr, timepts[t])
    # close tabular space

    text += '\\end{tabular} \n\\caption{%s} \n\\end{table}' % captionText


  print(text)


def ttestsGroups(resList, labels, groupLabels):
  '''
  Performs unpaired t-tests between different diagnostic groups on levels of atrophy,
    rate of atrophy and timing of atrophy

  '''

  nrGroups = len(resList)

  (nrSamples, nrPoints, nrBiomk) = resList[0]['xsSamples'].shape

  if nrGroups == 2:
    groupPairs = [[0,1]]
  elif nrGroups == 3:
    groupPairs = [[0, 1], [0,2], [1,2]]
  else:
    raise ValueError('need to define pairs for nrGroups other than 2,3')

  nrPairs = len(groupPairs)

  timepts = [-10, 0, 10]
  nrTimepts = len(timepts)

  # define a cell array
  groupTs = np.empty((nrBiomk, nrGroups), dtype=object)
  groupXs = np.empty((nrBiomk, nrGroups), dtype=object)

  groupPred = np.empty((nrBiomk, nrGroups, nrTimepts), dtype=object)

  tStat = np.zeros((nrBiomk, nrPairs, nrTimepts), float)
  pVal = np.zeros((nrBiomk, nrPairs, nrTimepts), float)

  fs = [0 for g in range(nrGroups)]

  for b in range(nrBiomk):
    print("biomk %d %s" % (b, labels[b]))

    for g in range(nrGroups):
      goodSamples = np.logical_not(resList[g]['badSamples'][:, b])
      groupTs[b, g] = resList[g]['tsSamples'][goodSamples,:,b]
      groupXs[b, g] = resList[g]['xsSamples'][goodSamples,:,b]

      assert groupTs[b, g].shape[0] == groupXs[b, g].shape[0]

      # print(groupTs[b, g].shape, groupXs[b, g].shape)
      nrGoodSamples = groupTs[b, g].shape[0]
      fs[g] = [interpolate.interp1d(groupTs[b, g][s,:], groupXs[b, g][s,:],
        kind='linear', fill_value='extrapolate') for s in range(nrGoodSamples)]

      for t in range(len(timepts)):
        # for three different timepoints
        groupPred[b,g,t] = [fs[g][s](timepts[t]) for s in range(len(fs[g]))]


    for pairIndex in range(len(groupPairs)):
      for t in range(len(timepts)):
        # for three different timepoints
        g1Index = groupPairs[pairIndex][0]
        g2Index = groupPairs[pairIndex][1]
        tStat[b, pairIndex, t], pVal[b, pairIndex, t] = scipy.stats.ttest_ind(
          groupPred[b,g1Index,t], groupPred[b,g2Index,t])

        print('timept=%d: tStats=%f, pVal=%f' % (timepts[t], tStat[b, pairIndex, t],
          pVal[b, pairIndex, t]))

  createLatexGroups(resList[0]['outFolder'], labels, timepts, groupPairs, tStat, pVal)

  # print(adas)

def createLatexGroups(outFolder, labels, timepts, groupPairs, tStat, pVal):

  text = r'''
\documentclass[11pt,a4paper,landscape]{report}
\usepackage{amsmath,amssymb,calc,ifthen}
\usepackage{float}
\usepackage[table,usenames,dvipsnames]{xcolor} % for coloured cells in tables
\usepackage{amsmath,graphicx}

\begin{document}
\belowdisplayskip=12pt plus 3pt minus 9pt
\belowdisplayshortskip=7pt plus 3pt minus 4pt

'''
  nrBiomk = len(labels)
  alpha = 0.05
  corrTerm = nrBiomk * len(timepts)

  for pairIndex in range(len(groupPairs)):

    text += r'''\begin{table}[H]
\centering
\begin{tabular}{c |''' + ''.join(['c ' for x in timepts])  + r'''}
Region & T1 = -10 years &  T2 = 0 years &  T3 = 10 years\\
\hline
'''

    for b in range(nrBiomk):
      text += labels[b]

      for t in range(len(timepts)):
        text += ' & %.2e' % (pVal[b, pairIndex, t])
        if pVal[b, pairIndex, t] < alpha/corrTerm:
          text += '*'

      # endline
      text+= '\\\\\n'

    # close tabular space
    text += r'''
\end{tabular}
\end{table}

'''


  text += r'''
\end{document}

'''

  print(text)

  if outFolder is not None:
    fileName = '%s/pcaAdTtests.tex' % outFolder
    print(fileName)
    with open(fileName, 'w') as f:
      f.write(text)

  ordering = [4, 7, 5, 6, 2, 3, 1]

  print('alpha/corrTerm', alpha/corrTerm)
  alphaThresh = [alpha/corrTerm, 1e-6, 1e-9]

  for pairIndex in range(len(groupPairs)):
    for b in ordering:
      print(labels[b])
      for t in range(len(timepts)):
        text = ''
        if pVal[b, pairIndex, t] > alphaThresh[0]:
          text = '-'
        else:
          if tStat[b, pairIndex, t] < 0:
            text += 'p'
          else:
            text += 'a'

          for a in range(len(alphaThresh)):
            if pVal[b, pairIndex, t] < alphaThresh[a]:
              text += '*'

        print(text)



def runSubgroupsPCA(dpmBuilder, expName, params):

  controlID = CTL
  allIDs = np.unique(params['subgroups'])

  # a. Vision subgroup: greater occipital lobe atrophy and cortical thinning than other subgroups?
  # b. Space subgroup:  greater superior parietal atrophy and cortical thinning than other subgroups?
  # c. Object subgroup: greater inferior temporal atrophy and cortical thinning than other subgroups?

  grIds = [EAR, PER, SPA]
  nrGr = len(grIds)
  suffixList = ['EAR', 'PER', 'SPA']

  res = [0 for i in range(nrGr)]
  expNameSubgroup = [0 for i in range(nrGr)]

  params['diag'] = params['subgroups']

# for g in range(nrGr):
  for g in [0,1,2]:
    params['patientID'] = grIds[g]
    params['plotTrajParams']['modelCol'] = 'b'  # blue
    params['excludeID'] = [CTL] # used in DEM to filter subjects that should not be sed in GP fit.
    params['excludeXvalidID'] = allIDs[np.logical_not(np.in1d(allIDs, [CTL, grIds[g]]))]
    params['anchorID'] = params['patientID']
    expNameSubgroup[g] = '%s%s' % (expName, suffixList[g])

    # params['plotTrajParams']['diagStr'] = ['CTL', suffixList[g]]
    # params['plotTrajParams']['diagColors'] = ['g', 'r']

    dpmObj, res[g] = runStdDPM(params, expNameSubgroup[g], dpmBuilder, params['runPartSubgroupMain'])

  outFolder = 'matfiles/%s' % expNameSubgroup[0]
  trajSubplotsFigName = '%s/subgroupsComparison.png' % outFolder
  # plot subfigures with each biomarker in turn that show two traj: EAR vs SPA vs PER
  if params['runPartSubgroupMain'][3] == 'R':
    fig = plotSubfigGroupComp(res, grIds, params['plotTrajParams'], params['labels'],
      groupLabels=suffixList)
    fig.savefig(trajSubplotsFigName, dpi=100)

def printResDRC(modelNames, res):
  nrModels = len(modelNames)
  upEqStagesPercPCA = np.zeros((nrModels, 2))
  pFUgrBLAllPCA = np.zeros((nrModels, 2))
  upEqStagesPercAD = np.zeros((nrModels, 2))
  pFUgrBLAllAD = np.zeros((nrModels, 2))
  upEqStagesPerc = np.zeros((nrModels, 2))
  pFUgrBLAll = np.zeros((nrModels, 2))
  diagCtlPca = np.zeros((nrModels, 2))
  diagCtlAd = np.zeros((nrModels, 2))
  diagPcaAd = np.zeros((nrModels, 2))

  timeDiffHardPCA = np.zeros((nrModels, 2))
  timeDiffHardAD = np.zeros((nrModels, 2))
  timeDiffSoftPCA = np.zeros((nrModels, 2))
  timeDiffSoftAD = np.zeros((nrModels, 2))
  timeDiffHard = np.zeros((nrModels, 2))
  timeDiffSoft = np.zeros((nrModels, 2))

  for m in range(nrModels):
    upEqStagesPercPCA[m, :] = res[m]['upEqStagesPercPCA']
    pFUgrBLAllPCA[m, :] = res[m]['pFUgrBLAllPCA']
    upEqStagesPercAD[m, :] = res[m]['upEqStagesPercAD']
    pFUgrBLAllAD[m, :] = res[m]['pFUgrBLAllAD']
    upEqStagesPerc[m, :] = (upEqStagesPercPCA[m, :] + upEqStagesPercAD[m, :]) / 2
    pFUgrBLAll[m, :] = (pFUgrBLAllPCA[m, :] + pFUgrBLAllAD[m, :]) / 2

    timeDiffHardPCA[m, :] = res[m]['timeDiffHardPCA']
    timeDiffSoftPCA[m, :] = res[m]['timeDiffSoftPCA']
    timeDiffHardAD[m, :] = res[m]['timeDiffHardAD']
    timeDiffSoftAD[m, :] = res[m]['timeDiffSoftAD']
    timeDiffHard[m, :] = (timeDiffHardPCA[m, :] + timeDiffHardAD[m, :]) / 2
    timeDiffSoft[m, :] = (timeDiffSoftPCA[m, :] + timeDiffSoftAD[m, :]) / 2


    diagCtlPca[m, :] = [res[m]['dirDiagStatsPCA']['mean'][0],
                        res[m]['dirDiagStatsPCA']['std'][0]]
    diagCtlAd[m, :] = [res[m]['dirDiagStatsAD']['mean'][0],
                       res[m]['dirDiagStatsAD']['std'][0]]
    diagPcaAd[m, :] = [res[m]['diffDiagStats']['mean'][0],
                       res[m]['diffDiagStats']['std'][0]]

  np.set_printoptions(precision=4)
  print(modelNames)
  print('upEqStagesPerc', arrayToStrNoBrackets(upEqStagesPerc))
  print('upEqStagesPercPCA', arrayToStrNoBrackets(upEqStagesPercPCA))
  print('upEqStagesPercAD', arrayToStrNoBrackets(upEqStagesPercAD))

  print('pFUgrBLAll', arrayToStrNoBrackets(pFUgrBLAll))
  print('pFUgrBLAllPCA', arrayToStrNoBrackets(pFUgrBLAllPCA))
  print('pFUgrBLAllAD', arrayToStrNoBrackets(pFUgrBLAllAD))

  print('timeDiffHard', arrayToStrNoBrackets(timeDiffHard))
  print('timeDiffHardPCA', arrayToStrNoBrackets(timeDiffHardPCA))
  print('timeDiffHardAD', arrayToStrNoBrackets(timeDiffHardAD))

  print('timeDiffSoft', arrayToStrNoBrackets(timeDiffSoft))
  print('timeDiffSoftPCA', arrayToStrNoBrackets(timeDiffSoftPCA))
  print('timeDiffSoftAD', arrayToStrNoBrackets(timeDiffSoftAD))

  print('diagPcaAd', arrayToStrNoBrackets(diagPcaAd))
  print('diagCtlPca', arrayToStrNoBrackets(diagCtlPca))
  print('diagCtlAd', arrayToStrNoBrackets(diagCtlAd))

  formalLabels = ['DEM - Standard Alignment', 'DEM - Optimised Alignment']

  print('PCA staging')
  for m in range(nrModels):
    print('  %s & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f\\\\' %
          (formalLabels[m], upEqStagesPercPCA[m,0], upEqStagesPercPCA[m,1],
          pFUgrBLAllPCA[m, 0], pFUgrBLAllPCA[m, 1],
          timeDiffHardPCA[m, 0], timeDiffHardPCA[m, 1],
          timeDiffSoftPCA[m, 0], timeDiffSoftPCA[m, 1]))

  print('\nAD staging')
  for m in range(nrModels):
    print('  %s & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f\\\\' %
          (formalLabels[m], upEqStagesPercAD[m, 0], upEqStagesPercAD[m, 1],
           pFUgrBLAllAD[m, 0], pFUgrBLAllAD[m, 1],
           timeDiffHardAD[m, 0], timeDiffHardAD[m, 1],
           timeDiffSoftAD[m, 0], timeDiffSoftAD[m, 1]))

  print('\n DRC diag pred')
  for m in range(nrModels):
    print('  %s & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f\\\\' %
          (formalLabels[m], diagPcaAd[m, 0], diagPcaAd[m, 1],
           diagCtlPca[m, 0], diagCtlPca[m, 1],
           diagCtlAd[m, 0], diagCtlAd[m, 1]))

  pass