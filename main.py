import sys
#QT imports
from PyQt5.QtWidgets import QApplication, QMainWindow, QInputDialog, QLineEdit, QFileDialog, QErrorMessage, QTableWidget, QTableWidgetItem,QTableWidgetItem
from PyQt5 import QtCore, QtWidgets, uic
from main_designer import Ui_MainWindow
# plotting
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
#import myternaryutility
sys.path.append("../PythonCompositionPlots-master")
from myternaryutility import TernaryPlot
#numerical things
import pandas as pd
import numpy as np
import random
#sklearn imports
from sklearn.cluster import KMeans as kmc
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn import manifold
from sklearn import decomposition
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy import signal
#import xu
import xrayutilities as xu

#make the UI
uiFile = '/Users/helgestein/Documents/PythonScripts/Python3/htAx2/main_designer.ui' # Enter file here.
Ui_MainWindow, QtBaseClass = uic.loadUiType(uiFile)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        #fix the menubar for osx
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        #connect the slots

        #xrdraw
        self.actionComposition.triggered.connect(self.importFcn)
        self.actionPosition.triggered.connect(self.importFcn)
        self.actionXRD.triggered.connect(self.importFcn)
        self.actionAngle.triggered.connect(self.importFcn)
        self.actionFunction.triggered.connect(self.importFcn)
        self.horizontalSliderRaw.valueChanged['int'].connect(self.XRDSliderPositionChange)

        #they are implemented sepearately so that not all figures update
        self.comboBox.currentIndexChanged.connect(self.comboBoxDispChange)
        self.comboBox_3.currentIndexChanged.connect(self.comboBoxNMFChange)
        self.comboBox_4.currentIndexChanged.connect(self.comboBoxQBChange)

        self.comboBox_cut.currentIndexChanged.connect(self.comboBoxQBChange)
        self.comboBox_sort.currentIndexChanged.connect(self.comboBoxQBChange)
        self.doubleSpinBox_val.valueChanged.connect(self.comboBoxQBChange)
        self.doubleSpinBox_tol.valueChanged.connect(self.comboBoxQBChange)
        self.setShoVec.valueChanged.connect(self.showVecChange)

        #clustering
        self.clusterButton.clicked.connect(self.onClickClusterButton)
        self.pdButton.clicked.connect(self.onClickpdButton)
        self.mdsButton.clicked.connect(self.onClickmdsButton)
        self.pushButtonNMF.clicked.connect(self.onClickNMFButton)
        self.comboBoxClusterMethods.currentIndexChanged.connect(self.comboBoxClusterMethodsChange)

        #QBplots
        self.saveQBButton.clicked.connect(self.onClicksaveQBButton)
        self.forgetQBButton.clicked.connect(self.onClickforgetQBButton)

        #make plots ...

        #rawXRD
        self.figureXRD = plt.figure()
        self.canvasXRD = FigureCanvas(self.figureXRD)
        self.toolbarXRD = NavigationToolbar(self.canvasXRD, self)

        self.figureSpec = plt.figure()
        self.canvasSpec = FigureCanvas(self.figureSpec)
        self.toolbarSpec = NavigationToolbar(self.canvasSpec, self)

        #clustering
        self.figureCluRes = plt.figure()
        self.canvasCluRes = FigureCanvas(self.figureCluRes)
        self.toolbarCluRes = NavigationToolbar(self.canvasCluRes, self)

        self.figureCluPDM = plt.figure()
        self.canvasCluPDM = FigureCanvas(self.figureCluPDM)
        self.toolbarCluPDM = NavigationToolbar(self.canvasCluPDM, self)

        self.figureCluMDS = plt.figure()
        self.canvasCluMDS = FigureCanvas(self.figureCluMDS)
        self.toolbarCluMDS = NavigationToolbar(self.canvasCluMDS, self)

        #decomposition
        self.figureDecRes = plt.figure()
        self.canvasDecRes = FigureCanvas(self.figureDecRes)
        self.toolbarDecRes = NavigationToolbar(self.canvasDecRes, self)

        self.figureDecSpec = plt.figure()
        self.canvasDecSpec = FigureCanvas(self.figureDecSpec)
        self.toolbarDecSpec = NavigationToolbar(self.canvasDecSpec, self)

        #QBCuts
        self.figureQBRes = plt.figure()
        self.canvasQBRes = FigureCanvas(self.figureQBRes)
        self.toolbarQBRes = NavigationToolbar(self.canvasQBRes, self)

        self.figureQBPlot = plt.figure()
        self.canvasQBPlot = FigureCanvas(self.figureQBPlot)
        self.toolbarQBPlot = NavigationToolbar(self.canvasQBPlot, self)

        #fitting
        self.figureFitPlot = plt.figure()
        self.canvasFitPlot = FigureCanvas(self.figureFitPlot)
        self.toolbarFitPlot = NavigationToolbar(self.canvasFitPlot, self)

        #add plots as widgets to the existing things

        #rawXRD
        self.XRDLayout.addWidget(self.toolbarXRD)
        self.XRDLayout.addWidget(self.canvasXRD)

        self.XRDLayoutSpectral.addWidget(self.toolbarSpec)
        self.XRDLayoutSpectral.addWidget(self.canvasSpec)

        #clustering
        self.posClusterLayout.addWidget(self.toolbarCluRes)
        self.posClusterLayout.addWidget(self.canvasCluRes)

        self.pdMatrixLayout.addWidget(self.toolbarCluPDM)
        self.pdMatrixLayout.addWidget(self.canvasCluPDM)

        self.mdsLayout.addWidget(self.toolbarCluMDS)
        self.mdsLayout.addWidget(self.canvasCluMDS)

        #fitting
        self.res.addWidget(self.toolbarFitPlot)
        self.res.addWidget(self.canvasFitPlot)

        #decomposition
        self.scatterDecompositionLayout.addWidget(self.toolbarDecRes)
        self.scatterDecompositionLayout.addWidget(self.canvasDecRes)

        self.DecompositionSpectralLayout.addWidget(self.toolbarDecSpec)
        self.DecompositionSpectralLayout.addWidget(self.canvasDecSpec)

        #QBCuts
        self.scatterQBLayout.addWidget(self.toolbarQBRes)
        self.scatterQBLayout.addWidget(self.canvasQBRes)

        self.QBPlotLayout.addWidget(self.toolbarQBPlot)
        self.QBPlotLayout.addWidget(self.canvasQBPlot)

        #make some plots clickable
        self.canvasXRD.mpl_connect('button_press_event', self.onclickXRD)
        self.canvasSpec.mpl_connect('button_press_event', self.onclickSpectra)

        #make the clustering things clickable
        self.canvasCluRes.mpl_connect('button_press_event', self.onclickXRDClusRes)
        self.canvasCluPDM.mpl_connect('button_press_event', self.onclickXRDClusPDM)
        self.canvasCluMDS.mpl_connect('button_press_event', self.onclickXRDClusMDS)

        #make QB plots clickable
        self.canvasQBPlot.mpl_connect('button_press_event', self.onclickQBPlot)

        #make buttons in fitting clickable
        self.pushButtonLoadCifs.clicked.connect(self.onClickpushButtonLoadCifs)
        self.pushButtonFit.clicked.connect(self.onClickpushButtonFit)
        self.pushButtonPlotFit.clicked.connect(self.onClickpushButtonPlotFit)
        #set default plots
        self.plotXRDIntensity = 'Coordinates'
        plt.rcParams['axes.facecolor']='white'
        self.selectedQBSave = []
        self.cwt_min = 1
        self.cwt_max = 100
        self.tolerance = 100
    def importFcn(self, value):
        sender = self.sender()
        self.statusBar().showMessage(sender.text() + ' will be imported ...')

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","CSV Files (*.csv);;All Files (*)", options=options)

        if fileName:
            if sender.text() == 'Composition':
                self.compo = pd.read_csv(fileName,delimiter=',')
                self.compoValues = self.compo.values
                self.elements = [o for o in self.compo.axes[1]]
                self.comboBox_cut.clear()
                self.comboBox_cut.addItems(self.elements)
                self.comboBox_sort.clear()
                self.comboBox_sort.addItems(self.elements)
            elif sender.text() == 'Position':
                self.position = pd.read_csv(fileName,delimiter=',')
                try:
                    self.x = self.position['x'].values
                    self.y = self.position['y'].values
                except:
                    error_dialog = QErrorMessage()
                    error_dialog.showMessage('Your positions file is incorrect! {}'.format(self.position))
                    error_dialog.exec_()
                self.plotpos()
            elif sender.text() == 'XRD/Raman':
                self.XRD = pd.read_csv(fileName,delimiter=',',header=None)
                self.plotXRD()
            elif sender.text() == 'Angle':
                self.angle = pd.read_csv(fileName,header=None)
                self.maxAnlge = np.max(self.angle)
                self.minAnlge = np.min(self.angle)
                self.horizontalSliderRaw.setMaximum(self.maxAnlge*1000)
                self.horizontalSliderRaw.setMinimum(self.minAnlge*1000)
                self.plotXRD()

            elif sender.text() == 'Function':
                self.function = pd.read_csv(fileName,delimiter=',')
            self.statusBar().showMessage(sender.text() + ' was imported.')
        else:
            self.statusBar().showMessage('Nothing was selected...')

    def XRDSliderPositionChange(self):
        self.plotpos()

    def comboBoxDispChange(self):
        self.plotXRDIntensity = self.comboBox.currentText()
        self.plotpos()

    def plotpos(self):
        if hasattr(self, 'XRD') and hasattr(self, 'angle'):
            selAngle = self.horizontalSliderRaw.value()/1000
            self.angleID = np.argmin(np.abs(self.angle.values-selAngle))
            self.figureXRD.clear()
            ax = self.figureXRD.add_subplot(111)
            ax.axis('equal')
            #select one of the plotting styles
            if self.plotXRDIntensity == 'Coordinates':
                self.scatterCoordsPlot(ax,prop='Intensity')
                self.canvasXRD.draw()
            elif self.plotXRDIntensity == 'Ternary':
                self.scatterTernaryPlot(ax,prop='Intensity')
                self.canvasXRD.draw()
            #redraw the XRD plot for when the slider changed it's position
            self.plotXRD()
        elif hasattr(self, 'position'):
            self.figureXRD.clear()
            ax = self.figureXRD.add_subplot(111)
            ax.plot(self.x,self.y.T, '*-')
            self.canvasXRD.draw()

    def plotXRD(self):
        if hasattr(self, 'XRD') and hasattr(self, 'angle'):
            if hasattr(self, 'xselect'):
                self.figureSpec.clear()
                ax = self.figureSpec.add_subplot(111)
                ax.plot(self.angle,self.XRD.values[:,self.selectedXRDID])
                if hasattr(self, 'angleID') and hasattr(self, 'selectedXRDID'):
                    x,y = self.angle.values[self.angleID], self.XRD.values[self.angleID,self.selectedXRDID]
                    ax.scatter(x,y,s=50,color='red')
                self.canvasSpec.draw()
            else:
                self.figureSpec.clear()
                ax = self.figureSpec.add_subplot(111)
                ax.plot(self.angle,self.XRD)
                self.canvasSpec.draw()

    def onclickSpectra(self,event):
        if event.xdata is None:
            print('Clicked outside!')
        else:
            self.xselectSpec, self.yselectSpec = event.xdata, event.ydata
            self.angleID = np.argmin(np.abs(self.angle.values-self.xselectSpec))
            self.horizontalSliderRaw.setValue(self.angle.values[self.angleID]*1000)
            self.plotXRD()
            self.plotpos()

    def onclickXRD(self,event):
        if event.xdata is None:
            print('Clicked outside!')
        else:
            self.xselect, self.yselect = event.xdata, event.ydata
            print('Clicked at x:{}, y:{}'.format(self.xselect, self.yselect))
            #calculate the distances and select the nearest
            if self.plotXRDIntensity == 'Ternary':
                if hasattr(self, 'compo'):
                    self.figureXRD.clear()
                    ax = self.figureXRD.add_subplot(111)
                    stp = TernaryPlot(ax, ellabels=self.elements)
                    cmp = stp.toComp([self.xselect, self.yselect])
                    print('Clicked at {}'.format(cmp))
                    distsCMP = np.sqrt(np.sum((self.compoValues/100-cmp)**2,axis=1))
                    self.selectedXRDID = np.argmin(distsCMP)
                    print('Selected composition {} Selected ID:{}'.format(self.compoValues[self.selectedXRDID,:]/100,self.selectedXRDID))
            elif self.plotXRDIntensity == 'Coordinates':
                distsXY = np.sqrt((self.x-self.xselect)**2+(self.y-self.yselect)**2)
                self.selectedXRDID = np.argmin(distsXY)
                print('Selected I ID:{}'.format(self.selectedXRDID))
            self.plotXRD()
            self.plotpos()
            self.MAInfo.setText('MA ID:')
            self.lcdD.display(self.selectedXRDID+1)

    def comboBoxClusterMethodsChange(self, event):
        #this adjusts the indeces of the metrics box
        self.ClusterMethod = self.comboBoxClusterMethods.currentText()
        if self.ClusterMethod == 'KMeans':
            self.comboBoxMetric.clear()
            self.comboBoxMetric.addItems(['euclidean'])
        elif self.ClusterMethod == 'AgglomWard':
            self.comboBoxMetric.clear()
            self.comboBoxMetric.addItems(['euclidean'])
        elif self.ClusterMethod == 'AgglomAvg':
            self.comboBoxMetric.clear()
            self.comboBoxMetric.addItems(['euclidean','l1','l2','manhattan','cosine','chebyshev','seuclidean','canberra','braycurtis','hamming'])
        elif self.ClusterMethod == 'AgglomComplete':
            self.comboBoxMetric.clear()
            self.comboBoxMetric.addItems(['euclidean','l1','l2','manhattan','cosine','chebyshev','canberra','braycurtis','hamming'])
        elif self.ClusterMethod == 'SpectralClustering':
            self.comboBoxMetric.clear()
            self.comboBoxMetric.addItems(['nearest_neighbors','rbf'])
        elif self.ClusterMethod == 'DBScan':
            self.comboBoxMetric.clear()
            self.comboBoxMetric.addItems(['euclidean','l1','l2','manhattan','cosine','chebyshev','seuclidean','canberra','braycurtis','hamming'])


    def onClickClusterButton(self,event):
        self.ClusterMethod = self.comboBoxClusterMethods.currentText()
        self.Metric = self.comboBoxMetric.currentText()
        if hasattr(self, 'XRD'):
            #all clusterings should more or less set self.ClusterID
            if self.ClusterMethod == 'KMeans':
                print('Clusters: {}'.format(self.setNumberclusters.value()))
                clus = kmc(n_clusters=self.setNumberclusters.value(), random_state=0).fit(self.XRD.values.T)
                self.ClusterID = np.array(clus.labels_)
                #tell the user it was done using euclidean
                self.comboBoxMetric.setCurrentIndex(0)
                #tell the user a second time
                self.statusBar().showMessage('KMeans clustering is currently only possibile using the Euclidean')
            elif self.ClusterMethod == 'AgglomWard':
                print('Clusters: {}'.format(self.setNumberclusters.value()))
                clus = AgglomerativeClustering(n_clusters=self.setNumberclusters.value()).fit(self.XRD.values.T)
                self.ClusterID = np.array(clus.labels_)
                self.comboBoxMetric.setCurrentIndex(0)
            elif self.ClusterMethod == 'AgglomAvg':
                print('Clusters: {}'.format(self.setNumberclusters.value()))
                clus = AgglomerativeClustering(n_clusters=self.setNumberclusters.value(),linkage='average',affinity=self.Metric).fit(self.XRD.values.T)
                self.ClusterID = np.array(clus.labels_)
            elif self.ClusterMethod == 'AgglomComplete':
                print('Clusters: {}'.format(self.setNumberclusters.value()))
                clus = AgglomerativeClustering(n_clusters=self.setNumberclusters.value(),linkage='complete',affinity=self.Metric).fit(self.XRD.values.T)
                self.ClusterID = np.array(clus.labels_)
            elif self.ClusterMethod == 'SpectralClustering':
                clus = SpectralClustering(n_clusters=self.setNumberclusters.value(), eigen_solver='arpack',affinity=self.Metric).fit(self.XRD.values.T)
                self.ClusterID = np.array(clus.labels_)
            elif self.ClusterMethod == 'DBScan':
                clus = DBSCAN(metric=self.Metric,eps=np.float(self.setEps.text()), min_samples=np.int(self.setNMin.text())).fit(self.XRD.values.T)
                self.ClusterID = np.array(clus.labels_)
            self.statusBar().showMessage('{} is using the {} metric'.format(self.ClusterMethod,self.Metric))
            #what follows here is essentially the same as in raw xrd but only with
            #cluster information as color code and a matching colorscale
            self.plotClusRes()

    def plotClusRes(self):
        if hasattr(self, 'position'):
            self.figureCluRes.clear()
            ax = self.figureCluRes.add_subplot(111)
            ax.axis('equal')
            self.plotXRDIntensity = self.comboBox_2.currentText()
            #select one of the plotting styles
            if self.plotXRDIntensity == 'Coordinates':
                self.scatterCoordsPlot(ax,prop='clust')
                self.canvasCluRes.draw()
            elif self.plotXRDIntensity == 'Ternary':
                self.scatterTernaryPlot(ax,prop='clust')
                self.canvasCluRes.draw()

        elif hasattr(self, 'position'):
            self.figureCluRes.clear()
            ax = self.figureCluRes.add_subplot(111)
            ax.plot(self.x,self.y.T, '*-')
            self.canvasCluRes.draw()

    def onClickpdButton(self,event):
            self.Metric = self.comboBoxMetric.currentText()
            if hasattr(self, 'XRD'):
                self.calculatePD()

    def calculatePD(self):
            if self.Metric == 'l1':
                self.pdm = pdist(X=self.XRD.values.T,metric='cityblock')
            elif self.Metric == 'l2':
                self.pdm = pdist(X=self.XRD.values.T,metric='euclidean')
            elif self.Metric == 'euclidean':
                self.pdm = pdist(X=self.XRD.values.T,metric='euclidean')
            elif self.Metric == 'manhattan':
                self.pdm = pdist(X=self.XRD.values.T,metric='citylock')
            elif self.Metric == 'cosine':
                self.pdm = pdist(X=self.XRD.values.T,metric='cosine')
            self.plotPDM()

    def plotPDM(self):
        self.figureCluPDM.clear()
        ax = self.figureCluPDM.add_subplot(111)
        ax.axis('equal')
        if hasattr(self, 'pdm'):
            if hasattr(self, 'ClusterID'):
                ID = np.argsort(self.ClusterID)
                mat = np.array(squareform(self.pdm))
                matarr = mat[ID,:]
                matarr = matarr[:,ID]
                ax.imshow(matarr,origin='lower')
                if hasattr(self, 'selectedXRDID'):
                    ax.plot([0,len(mat[0,:])],[self.selectedXRDID,self.selectedXRDID],color='red')
                    ax.plot([self.selectedXRDID,self.selectedXRDID],[0,len(mat[0,:])],color='red')

            else:
                ax.imshow(np.array(squareform(self.pdm)),origin='lower')
        self.canvasCluPDM.draw()

    def onClickmdsButton(self,event):
        if hasattr(self, 'pdm'):
            self.calculateMDS()

    def calculateMDS(self):
        mdso = manifold.MDS(dissimilarity='precomputed')
        self.pos = mdso.fit(squareform(self.pdm)).embedding_
        self.plotMDS()

    def plotMDS(self):
        if hasattr(self, 'pdm') and hasattr(self, 'pos'):
            self.figureCluMDS.clear()
            ax = self.figureCluMDS.add_subplot(111)
            ax.axis('equal')
            if hasattr(self, 'pdm'):
                if hasattr(self, 'ClusterID'):
                    ax.scatter(self.pos[:,0],self.pos[:,1],c=self.ClusterID)
                    if hasattr(self, 'selectedXRDID'):
                        ax.scatter(self.pos[self.selectedXRDID,0],self.pos[self.selectedXRDID,1],c='red')
            self.canvasCluMDS.draw()

    def onClickNMFButton(self,event):
        self.plotXRDIntensity = self.comboBox_3.currentText()
        print(self.plotXRDIntensity)
        if hasattr(self, 'XRD'):
            #do the NMF with the parameter supplied
            #currently this is the most bare nmf implementation possibile ...
            model = decomposition.NMF(n_components=self.setNoVec.value(), init='random', random_state=0)
            self.W = model.fit_transform(self.XRD.values.T)
            self.H = model.components_
            self.showVecID = np.int(self.setShoVec.value())
            self.plotXRDDecompIntensity()
            self.plotXRDDecompSpec()

    def plotXRDDecompIntensity(self):
        self.figureDecRes.clear()
        ax = self.figureDecRes.add_subplot(111)
        if self.plotXRDIntensity == 'Coordinates':
            self.scatterCoordsPlot(ax,prop='W')
            self.canvasDecRes.draw()
        elif self.plotXRDIntensity == 'Ternary':
            self.scatterTernaryPlot(ax,prop='W')
            self.canvasDecRes.draw()

    def plotXRDDecompSpec(self):
        self.figureDecSpec.clear()
        ax = self.figureDecSpec.add_subplot(111)
        ax.plot(self.angle,self.H[self.showVecID,:])
        self.canvasDecSpec.draw()

    def showVecChange(self):
        selVec = np.abs(np.int(self.setShoVec.value()))
        if selVec<self.setNoVec.value()-1:
            self.showVecID = selVec
        else:
            self.showVecID = 0
            self.statusBar().showMessage('Illegal vector number selected, was set to 0')
        if hasattr(self,'W'):
            self.plotXRDDecompIntensity()
            self.plotXRDDecompSpec()

    def comboBoxNMFChange(self):
        self.plotXRDIntensity = self.comboBox_3.currentText()
        if hasattr(self,'W'):
            self.plotXRDDecompIntensity()
            self.plotXRDDecompSpec()

    def onclickXRDClusRes(self,event):
        #clicked Clustering results
        if event.xdata is None:
            print('Clicked outside!')
        else:
            self.xselect, self.yselect = event.xdata, event.ydata
            #print('Clicked at x:{}, y:{}'.format(self.xselect, self.yselect))
            #calculate the distances and select the nearest
            if self.plotXRDIntensity == 'Ternary':
                self.figureCluRes.clear()
                ax = self.figureCluRes.add_subplot(111)
                stp = TernaryPlot(ax, ellabels=self.elements)
                cmp = stp.toComp([self.xselect, self.yselect])
                print('Clicked at {}'.format(cmp))
                distsCMP = np.sqrt(np.sum((self.compoValues/100-cmp)**2,axis=1))
                self.selectedXRDID = np.argmin(distsCMP)
                #print('Selected composition {} Selected ID:{}'.format(self.compoValues[self.selectedXRDID,:]/100,self.selectedXRDID))
            elif self.plotXRDIntensity == 'Coordinates':
                distsXY = np.sqrt((self.x-self.xselect)**2+(self.y-self.yselect)**2)
                self.selectedXRDID = np.argmin(distsXY)
                #print('Selected I ID:{}'.format(self.selectedXRDID))
            self.plotClusRes()
            self.plotPDM()
            self.plotMDS()
            self.MAInfo.setText('MA ID:')
            self.lcdD.display(self.selectedXRDID+1)

    def onclickXRDClusPDM(self,event):
        #clicked PDM results
        if event.xdata is None:
            print('Clicked outside!')
        else:
            self.xselectPDM, self.yselectPDM = event.xdata, event.ydata
            #print(np.round(self.xselectPDM))
            self.selectedXRDID = np.int(np.round(self.yselectPDM))
            self.plotClusRes()
            self.plotPDM()
            self.plotMDS()

    def onclickXRDClusMDS(self,event):
        #clicked MDS results
        if event.xdata is None:
            print('Clicked outside!')
        else:
            self.xselect, self.yselect = event.xdata, event.ydata
            ax = self.figureCluPDM.add_subplot(111)
            dists = np.sqrt((self.pos[:,0]-self.xselect)**2+(self.pos[:,1]-self.yselect)**2)
            self.selectedXRDID = np.argmin(dists)
            if hasattr(self, 'compoValues'):
                print('Selected composition {} Selected ID:{}'.format(self.compoValues[self.selectedXRDID,:]/100,self.selectedXRDID))
            self.plotClusRes()
            self.plotPDM()
            self.plotMDS()

    def comboBoxQBChange(self):
        if hasattr(self, 'XRD') and hasattr(self, 'angle'):
            if hasattr(self, 'compoValues') and hasattr(self, 'y'):
                self.plotXRDIntensity = self.comboBox_4.currentText()
                self.cutEl = self.comboBox_cut.currentText()
                self.cutConc = self.doubleSpinBox_val.value()
                self.cutTol = self.doubleSpinBox_tol.value()
                self.doubleSpinBox_val.setSingleStep(self.cutTol)
                self.sortEl = self.comboBox_sort.currentText()
                self.QBID = np.where(abs(self.compo[self.cutEl]-np.float(self.cutConc))<np.float(self.cutTol))
                self.plotQBScatter()
                self.plotQBPlot()

    def plotQBScatter(self):
        self.figureQBRes.clear()
        ax = self.figureQBRes.add_subplot(111)
        if self.plotXRDIntensity == 'Coordinates' and hasattr(self, 'x'):
            self.scatterCoordsPlot(ax,prop='Intensity')
        elif self.plotXRDIntensity == 'Ternary' and hasattr(self, 'compo'):
            self.scatterTernaryPlot(ax,prop='Intensity')
        self.canvasQBRes.draw()

    def plotQBPlot(self):
        self.figureQBPlot.clear()
        ax = self.figureQBPlot.add_subplot(111)
        if self.cutEl in self.elements and self.cutEl != self.sortEl:
            if len(self.QBID[0])<2:
                self.QBID[0] = [0,1,2]
            self.sQBID = np.argsort(self.compo[self.sortEl].values[self.QBID])
            self.xg, self.yq = np.meshgrid(self.angle,self.compo[self.sortEl].values[self.QBID])
            self.qbdata = self.XRD.values[:,self.QBID[0][self.sQBID]].T
            plt.pcolormesh(self.xg,self.yq,self.qbdata,shading='gouraud')
            ax.axis('tight')
        self.canvasQBPlot.draw()
        #self.plotQBScatter()

    def onclickQBPlot(self,event):
        if event.xdata is None:
            print('Clicked outside!')
        else:
            if hasattr(self,'QBID'):
                self.xselect, self.yselect = event.xdata, event.ydata

                x = np.array(self.angle)
                y = np.array(self.compo[self.sortEl].values[self.QBID])

                distsx = np.sqrt((x-self.xselect)**2)
                distsy = np.sqrt((y*100-self.yselect)**2)

                self.angleID = np.argmin(distsx)

                distsy = np.sqrt((y-self.yselect)**2)
                self.selectedXRDID = self.QBID[0][np.argmin(distsy)]
                #new variable to use later
                self.selectedFromQB = self.selectedXRDID

                if hasattr(self, 'compoValues'):
                    print('Selected composition {} Selected ID:{}'.format(self.compoValues[self.selectedXRDID,:]/100,self.selectedXRDID))

                self.plotQBScatter()
    def onClicksaveQBButton(self,event):
        print('sane')
        if hasattr(self,'selectedXRDID'):
            self.selectedQBSave.append(self.selectedXRDID)
            print('QBSel: '.format(len(self.selectedQBSave)))
        self.plotQBScatter()

    def onClickforgetQBButton(self,event):
        self.selectedQBSave = []
        self.QBID = []
        self.statusBar().showMessage('Cleared QB results')

    def scatterCoordsPlot(self,ax,prop='Intensity'):
        if prop == 'Intensity':
            if hasattr(self, 'angleID'):
                c = self.XRD.values[self.angleID,:]
                ax.scatter(self.x,self.y, c=c)
        elif prop == 'W':
            if hasattr(self, 'showVecID'):
                c = self.W[:,self.showVecID]
                ax.scatter(self.x,self.y, c=c)
        elif prop=='clust':
            if hasattr(self, 'ClusterID'):
                c = self.ClusterID
                ax.scatter(self.x,self.y, c=c)
        elif prop=='FOM':
            if hasattr(self, 'selectedFOM'):
                c = self.selectedFOM
                ax.scatter(self.x,self.y, c=c)
        #orientation
        if hasattr(self, 'angleID'):
            ax.text(np.min(self.x),np.max(self.y),'{}'.format(self.angle.values[self.angleID]))
        #plot QB results
        if hasattr(self,'selectedQBSave'):
            ax.scatter(self.x[self.selectedQBSave],self.y[self.selectedQBSave], c='red',s=200)
        #if a MA has been selected mark it by a red dot
        if hasattr(self, 'selectedXRDID'):
            ax.scatter(self.x[self.selectedXRDID],self.y[self.selectedXRDID], c='red',s=50)
        if hasattr(self, 'QBID'):
            ax.scatter(self.x[self.QBID],self.y[self.QBID], color = 'grey',s=100, alpha=0.5)
        self.canvasDecRes.draw()

    def scatterTernaryPlot(self,ax,prop='Intensity'):
        #before plotting a ternary plot make sure we have a composition
        #this is for plotting a ternary
        if hasattr(self, 'compo'):
            #indent all
            stp = TernaryPlot(ax, ellabels=self.elements)
            if prop=='Intensity':
                if hasattr(self, 'angleID'):
                    c = self.XRD.values[self.angleID,:]
                    stp.scatter(self.compoValues/100,c=c)
            elif prop=='W':
                if hasattr(self, 'W') and hasattr(self, 'showVecID'):
                    c = self.W[:,self.showVecID]
                    stp.scatter(self.compoValues/100,c=c)
            elif prop=='clust':
                #if hasattr(self, 'ClusterID'):
                c = self.ClusterID
                stp.scatter(self.compoValues/100,c=c)
            elif prop=='FOM':
                if hasattr(self, 'selectedFOM'):
                    c = self.selectedFOM
                    stp.scatter(self.compoValues/100,c=c)
            #orientation
            if hasattr(self, 'QBID'):
                selC = self.compoValues[self.QBID,:]
                stp.scatter(selC[0]/100, c='grey', alpha=0.5, s=100)
            if hasattr(self, 'selectedXRDID'):
                selC = self.compoValues[self.selectedXRDID,:]
                stp.scatter(selC/100, c='red',s=50)
            if len(self.selectedQBSave)>0:
                selC = self.compoValues[self.selectedQBSave,:]
                stp.scatter(selC/100, c='red',s=200)
            ax.text(np.min(self.x),np.max(self.y),'{}'.format(self.angle.values[self.angleID]))
            #if a MA has been selected mark it by a red dot
            stp.label(fontsize=10)
    def onClickpushButtonLoadCifs(self,event):
        #open folderselectiondialogue
        filter = "CIF (*.cif)"
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileNames, _ = QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileName()", "",filter, options=options)

        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(len(fileNames))
        self.tableWidget.setColumnCount(7)
        self.tableWidget.setHorizontalHeaderLabels(['File','Formula', 'System', 'Symmetry Group','Peaks #','Main Peak [deg]','Main Peak [q]'])
        self.gridLayoutTable.addWidget(self.tableWidget)

        self.structures = []
        self.comboBoxStructureSelector.clear()
        for fileName,row in zip(fileNames,range(len(fileNames))):
            a,q,r,props = self.calcAngles(fileName)
            self.structures.append({'name':props[2], 'a':a, 'q':q, 'r':r})
            self.tableWidget.setItem(row,0, QTableWidgetItem(fileName))
            self.tableWidget.setItem(row,1, QTableWidgetItem(props[2]))
            self.tableWidget.setItem(row,2, QTableWidgetItem(props[0]))
            self.tableWidget.setItem(row,3, QTableWidgetItem(props[1]))
            self.tableWidget.setItem(row,4, QTableWidgetItem('{}'.format(len(a))))
            mid = np.argmax(r)
            self.tableWidget.setItem(row,5, QTableWidgetItem('{}'.format(a[mid])))
            self.tableWidget.setItem(row,6, QTableWidgetItem('{}'.format(q[mid])))
            self.comboBoxStructureSelector.addItems([props[2]])
    def onClickpushButtonFit(self,event):
        #find what to fit
        if self.comboBoxFitPatterns.currentText() == 'Imported XRD':
            fitData = self.XRD.values
        else:
            fitData = self.H
        self.cwt_min = self.spinBoxCWTmin.value()
        self.cwt_max = self.spinBoxCWTmax.value()
        self.tolerance = self.doubleSpinBoxTolerance.value()
        self.fom = []
        l = len(self.structures)
        for struct,p in zip(self.structures,range(l)):
            tfom = []
            for i in range(len(fitData[0,:])):
                #change this line accordingly if something else than angles should be fit
                tfom.append(self.calcFOM(self.angle.values,fitData[:,i],struct['a']))
            self.fom.append({'Structure':struct['name'], 'fom':tfom})
            self.progressBarFit.setValue(p/l)
            print('Structure prog {}'.format(p/(l)))

    def calcFOM(self, x, y, angles):
        peakind = signal.find_peaks_cwt(y, np.arange(self.cwt_min,self.cwt_max))
        #find the highest fom
        fom = 0
        for e,ei in zip(x[peakind],range(len(x[peakind]))):
            for t,ti in zip(angles,range(len(angles))):
                temp = np.log(y[peakind[ei]]/np.abs(e-t))
                if np.abs(e-t)<self.tolerance and fom>temp:
                    fom = temp
        return fom

    def calcAngles(self, file):
        try:
            st = xu.materials.Crystal.fromCIF(file)
            pst = xu.simpack.PowderDiffraction(st, tt_cutoff=90)

            unp = list(pst.data.values())
            angles = [unp[i]['ang'] for i in range(len(unp))]
            q = [unp[i]['qpos'] for i in range(len(unp))]
            r = [unp[i]['r'] for i in range(len(unp))]
            props = [st.lattice.crystal_system, st.lattice.name, st.name]
            return angles, q, r, props
        except:
            print('CIF related error!')

    def onClickpushButtonPlotFit(self, file):
        if hasattr(self,'fom'):
         #set the selected structure
         self.figureFitPlot.clear()
         ax = self.figureFitPlot.add_subplot(111)
         self.FOMName = self.comboBoxStructureSelector.currentText()
         for f in self.fom:
             if f['Structure'] == self.FOMName:
                 self.selectedFOM = f['fom']
         if self.comboBoxPlotTypeFit.currentText() == 'Coordinates':
             self.scatterCoordsPlot(ax,prop='FOM')
         elif self.comboBoxPlotTypeFit.currentText() == 'Ternary':
             self.scatterTernaryPlot(ax,prop='FOM')
         self.canvasFitPlot.draw()
        else:
            print('nope!')
def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
