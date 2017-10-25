# XRDExplorer
Simple QT5 UI to play around with combinatorial XRD datasets.
If you are running a standart sypy stack everything should work after these steps ...

# Installation
Install xrayutilities via:
'''
pip install –global-option=”–without-openmp” xrayutilities
'''
you need to change line 14 and 34
'''
sys.path.append('../PythonCompositionPlots-master')
uiFile = '../main_designer.ui'
'''
to the absolute path


