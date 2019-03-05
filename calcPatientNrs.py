import numpy as np
import pandas as pd

pd.options.display.max_colwidth = 100
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 1000)

# in this sheet, participants with non-AD CSF data were already filtered
df = pd.read_csv('../data/DRC/visible/180903-final_sheet.csv')
# print(df.s_scan_location)

# print(df[df.o_drc_code == 'DORFR'])

filterScansCodes = ['8362', '3170', '5279', '13946', '14487', '34006', '9763', '30716',
  '16168', '9753']

# print(df.s_studyno)
df.s_studynoStr = df.s_studyno.astype(str)
filterMaskCodes = np.in1d(df.s_studynoStr, filterScansCodes)

#df = df.loc[~filterMaskCodes, :].reset_index(drop=True)
df.loc[filterMaskCodes, 's_date_at_ax' : 's_studyno'] = np.nan
df.loc[filterMaskCodes, 's_studynoStr'] = np.nan


# print(df[df.o_drc_code == 'DORFR'])
# print(adsa)

df['p_date_at_ax'] = pd.to_datetime(df['p_date_at_ax'], errors='coerce')
df['o_dob_coded'] = pd.to_datetime(df['o_dob_coded'], errors='coerce')
df['age'] = (df['p_date_at_ax'] - df['o_dob_coded']).dt.days / 365

df.to_csv('../data/DRC/visible/180903-final_sheet_CsfFilt_MriFilt.csv', index=False)

df['p_date_at_ax'] = pd.to_datetime(df['p_date_at_ax'], errors='coerce')
df['o_dob_coded'] = pd.to_datetime(df['o_dob_coded'], errors='coerce')
df['age'] = (df['p_date_at_ax'] - df['o_dob_coded']).dt.days / 365

df['entriesWithImg'] = np.logical_not(pd.isnull(df.s_studyno))
df['entriesWithPsych'] = np.logical_not(pd.isnull(df.p_date_at_ax))
df.o_drc_code = df.o_drc_code.astype(str)

dfGrCodes = df.groupby(by='o_drc_code')
drcCodes = np.unique(df.o_drc_code)

dfGrDiag = df.groupby(by='o_diagnosis')
dfGrDiagUnq = dfGrDiag['o_drc_code'].nunique()

# print(dfGrDiagUnq)

### total nr of subjects
print('PCA', len(dfGrDiag.get_group('PCA').groupby(by='o_drc_code')))
print('tAD', len(dfGrDiag.get_group('AD').groupby(by='o_drc_code')))
print('Controls', len(dfGrDiag.get_group('Control').groupby(by='o_drc_code')))

tableNrs = pd.DataFrame(index=range(6), columns=['MinVisits',
  'PCA Img n', 'PCA Img A', 'PCA Img VI',
  'PCA Psych n', 'PCA Psych A', 'PCA Psych VI',
  'AD Img n', 'AD Img A', 'AD Img VI',
  'AD Psych n', 'AD Psych A', 'AD Psych VI',
  'Control Img n', 'Control Img A', 'Control Img VI',
  'Control Psych n', 'Control Psych A', 'Control Psych VI'])

minNrVisitsList = [1, 2, 3, 4, 5, 6]

diag = ['PCA', 'AD', 'Control']

for v in range(len(minNrVisitsList)):
  for d in range(len(diag)):
    diagGr = dfGrDiag.get_group(diag[d])
    # print(diagGr)
    # print(diagGr.columns)
    tableNrs.loc[v, 'MinVisits'] = minNrVisitsList[v]

    diagGrImg =  diagGr[diagGr.entriesWithImg]
    diagGrImgDrcGr = diagGrImg.groupby(by='o_drc_code')
    print(diagGrImgDrcGr['age'].min())
    # print(asda)
    tableNrs.loc[v, '%s Img n' % diag[d]] = len([x for x in diagGrImgDrcGr if x[1].shape[0] >= minNrVisitsList[v]])
    # tableNrs.loc[v, '%s Img A' % diag[d]] = len([x['age'] for x in diagGrImgDrcGr if x[1].shape[0] >= minNrVisitsList[v]])

    diagGrPsych =  diagGr[diagGr.entriesWithPsych]
    diagGrPsychDrcGr = diagGrPsych.groupby(by='o_drc_code')
    # print([x for x in diagGrPsychDrcGr][0])
    tableNrs.loc[v, '%s Psych n' % diag[d]] = len([x for x in diagGrPsychDrcGr if x[1].shape[0] >= minNrVisitsList[v]])




print(tableNrs[['MinVisits', 'PCA Img n', 'PCA Psych n', 'AD Img n', 'AD Psych n', 'Control Img n', 'Control Psych n']])