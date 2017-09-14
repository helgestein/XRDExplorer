import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QInputDialog, QLineEdit, QFileDialog, QErrorMessage
from PyQt5 import QtCore, QtWidgets
from main_designer import Ui_MainWindow
# Make sure that we are using QT5
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
#import John's myternaryutility
sys.path.append('/Users/helge/Documents/PythonScripts/Python3/PythonCompositionPlots-master')
from myternaryutility import TernaryPlot

#numerical stuff
import pandas as pd
import numpy as np
import random
import ternary
#sklearn stuff
from sklearn.cluster import KMeans as kmc
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn import manifold
from sklearn import decomposition
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
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
        self.comboBox.currentIndexChanged.connect(self.comboBoxDispChange)
        self.comboBox_3.currentIndexChanged.connect(self.comboBoxNMFChange)

        self.setShoVec.valueChanged.connect(self.showVecChange)

        #clustering
        self.clusterButton.clicked.connect(self.onClickClusterButton)
        self.pdButton.clicked.connect(self.onClickpdButton)
        self.mdsButton.clicked.connect(self.onClickmdsButton)
        self.pushButtonNMF.clicked.connect(self.onClickNMFButton)
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

        #decomposition
        self.scatterDecompositionLayout.addWidget(self.toolbarDecRes)
        self.scatterDecompositionLayout.addWidget(self.canvasDecRes)

        self.DecompositionSpectralLayout.addWidget(self.toolbarDecSpec)
        self.DecompositionSpectralLayout.addWidget(self.canvasDecSpec)

        #make some plots clickable
        self.canvasXRD.mpl_connect('button_press_event', self.onclickXRD)
        self.canvasSpec.mpl_connect('button_press_event', self.onclickSpectra)

        #make the clustering things clickable
        self.canvasCluRes.mpl_connect('button_press_event', self.onclickXRDClusRes)
        self.canvasCluPDM.mpl_connect('button_press_event', self.onclickXRDClusPDM)
        self.canvasCluMDS.mpl_connect('button_press_event', self.onclickXRDClusMDS)

        #set default plots
        self.plotXRDIntensity = 'Coordinates'
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
                ax.scatter(self.x,self.y, c = self.XRD.values[self.angleID,:])
                #if a MA has been selected mark it by a red dot
                if hasattr(self, 'selectedXRDID'):
                    print('fff')
                    self.plotXRD()
                    ax.scatter(self.x[self.selectedXRDID],self.y[self.selectedXRDID], color = 'red',s=50)
            elif self.plotXRDIntensity == 'Ternary':
                #before plotting a ternary plot make sure we have a composition
                if hasattr(self, 'compo'):
                    #this is for plotting a ternary
                    stp = TernaryPlot(ax, ellabels=self.elements)
                    if np.max(self.compoValues[:,1])>1:
                        #plot the color values
                        stp.scatter(self.compoValues/100,c=self.XRD.values[self.angleID,:])
                        #if a MA has been selected mark it by a red dot
                        if hasattr(self, 'selectedXRDID'):
                            x,y = stp.toCart(self.compoValues[self.selectedXRDID]/100)
                    else:
                        stp.scatter(self.compoValues,c=self.XRD.values[self.angleID,:])
                    stp.label(fontsize=10)
                    stp.colorbar()
                    if hasattr(self, 'selectedXRDID'):
                        ax.scatter(x,y, color = 'red',s=50)

                    #pylab.show()
                    # ax.scatter(self.x,self.y, color='green')
            ax.text(np.min(self.x),np.max(self.y),'{}'.format(self.angle.values[self.angleID]))
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
                #find the closest measurement area
                #dists = np.sqrt((self.x-self.xselect)**2+(self.y-self.yselect)**2)
                #self.selectedXRDID = np.argmin(dists)
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
                self.Metric = 'euclidean'
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
                clus = SpectralClustering(n_clusters=self.setNumberclusters.value(), eigen_solver='arpack',affinity='nearest_neighbors').fit(self.XRD.values.T)
                self.ClusterID = np.array(clus.labels_)
            elif self.ClusterMethod == 'DBScan':
                clus = DBSCAN(eps=np.float(self.setEps.text()), min_samples=np.int(self.setNMin.text())).fit(self.XRD.values.T)
                self.ClusterID = np.array(clus.labels_)
            #what follows here is essentially the same as in raw xrd but only with
            #cluster information as color code and a matching colorscale
            self.plotClusRes()

    def plotClusRes(self):
        if hasattr(self, 'position') and hasattr(self, 'compo'):
            self.figureCluRes.clear()
            ax = self.figureCluRes.add_subplot(111)
            ax.axis('equal')
            self.plotXRDIntensity = self.comboBox_2.currentText()
            #select one of the plotting styles
            if self.plotXRDIntensity == 'Coordinates':
                ax.scatter(self.x,self.y, c = self.ClusterID)
                #if a MA has been selected mark it by a red dot
                if hasattr(self, 'selectedXRDID'):
                    self.plotXRD()
                    ax.scatter(self.x[self.selectedXRDID],self.y[self.selectedXRDID], color = 'red',s=50)
            elif self.plotXRDIntensity == 'Ternary':
                #before plotting a ternary plot make sure we have a composition
                if hasattr(self, 'compo'):
                    #this is for plotting a ternary
                    stp = TernaryPlot(ax, ellabels=self.elements)
                    if np.max(self.compoValues[:,1])>1:
                        #plot the color values
                        stp.scatter(self.compoValues/100,c=self.ClusterID)
                        #if a MA has been selected mark it by a red dot
                        if hasattr(self, 'selectedXRDID'):
                            x,y = stp.toCart(self.compoValues[self.selectedXRDID]/100)
                    else:
                        stp.scatter(self.compoValues,c=self.ClusterID)
                    stp.label(fontsize=10)
                    stp.colorbar()
                    if hasattr(self, 'selectedXRDID'):
                        ax.scatter(x,y, color = 'red',s=50)
            ax.text(np.min(self.x),np.max(self.y),'{}'.format(self.angle.values[self.angleID]))
            self.canvasCluRes.draw()
        elif hasattr(self, 'position'):
            self.figureCluRes.clear()
            ax = self.figureCluRes.add_subplot(111)
            ax.plot(self.x,self.y.T, '*-')
            self.canvasCluRes.draw()

    def onClickpdButton(self,event):
        self.Metric = self.comboBoxMetric.currentText()

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
            ax.scatter(self.x,self.y, c = self.W[:,self.showVecID])
            #if a MA has been selected mark it by a red dot
            if hasattr(self, 'selectedXRDID'):
                self.plotXRDDecompSpec()
                ax.scatter(self.x[self.selectedXRDID],self.y[self.selectedXRDID], color = 'red',s=50)
            self.canvasDecRes.draw()
            ax.text(np.min(self.x),np.max(self.y),'{}'.format(self.angle.values[self.angleID]))
        elif self.plotXRDIntensity == 'Ternary':
            print('ja')
            #before plotting a ternary plot make sure we have a composition
            if hasattr(self, 'compo'):
                #this is for plotting a ternary
                stp = TernaryPlot(ax, ellabels=self.elements)
                if np.max(self.compoValues[:,1])>1:
                    #plot the color values
                    stp.scatter(self.compoValues/100,c=self.W[:,self.showVecID])
                    #if a MA has been selected mark it by a red dot
                    if hasattr(self, 'selectedXRDID'):
                        x,y = stp.toCart(self.compoValues[self.selectedXRDID]/100)
                else:
                    stp.scatter(self.compoValues,c=self.W[:,self.showVecID])
                stp.label(fontsize=10)
                stp.colorbar()
                if hasattr(self, 'selectedXRDID'):
                    ax.scatter(x,y, color = 'red',s=50)
            ax.text(np.min(self.x),np.max(self.y),'{}'.format(self.angle.values[self.angleID]))
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
            print('Clicked at x:{}, y:{}'.format(self.xselect, self.yselect))
            #calculate the distances and select the nearest
            if self.plotXRDIntensity == 'Ternary':
                self.figureCluRes.clear()
                ax = self.figureCluRes.add_subplot(111)
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
            print('Selected composition {} Selected ID:{}'.format(self.compoValues[self.selectedXRDID,:]/100,self.selectedXRDID))
            self.plotClusRes()
            self.plotPDM()
            self.plotMDS()

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
