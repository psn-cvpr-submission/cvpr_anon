"""Contains all the base pytorch NN.modules for this project

Functions: 
    create_scatteringExclusive -- creates scattering parameters

Exceptions:
    InvalidInitializationException -- Error thrown when an invalid initialization scheme is passed

Classes: 
    sn_Identity -- computes the identity function in forward pass
    sn_HybridModel -- combinations of a scattering and other nn.modules
    sn_ScatteringBase -- a scattering network
"""
import types

import torch
import cv2

import torch.nn as nn

from kymatio.torch import Scattering2D

from .create_filters import *
from .models_utils import get_filters_visualization, getOneFilter, getAllFilters, compareParams, compareParamsVisualization


class InvalidInitializationException(Exception):
    """Error thrown when an invalid initialization scheme is passed"""
    pass



def _register_single_filter(self, v, n):
    self.register_buffer('tensor' + str(n), v)

    
class sn_Identity(nn.Module):
    """Identity nn.Module for identity"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.n_coefficients = 1

    def forward(self, x):
        return x
        
    def saveFilterGrads(self,scatteringActive):
        pass

    def saveFilterValues(self,scatteringActive):
        pass

    def plotFilterGrad(self):
        pass

    def plotFilterGrads(self):
        pass

    def plotFilterValue(self):
        pass

    def plotFilterValues(self):
        pass

    def checkFilterDistance(self):
        return 0
    
    def setEpoch(self, epoch):
        self.epoch = epoch

    def releaseVideoWriters(self):
        pass
        
    def checkParamDistance(self):
        pass

    def checkDistance(self,compared):
        pass
    



class sn_ScatteringBase(Scattering2D):
    """A learnable scattering nn.module 

    parameters:
        learnable -- should the filters be learnable parameters of this model
        J -- scale of scattering (always 2 for now)
        N -- height of the input image
        M -- width of the input image
        initilization -- the type of init: ['Tight-Frame' or 'Random']
        seed -- the random seed used to initialize the parameters
    """

    def __str__(self):
        tempL = " L" if self.learnable else "NL"
        tempI = "TF" if self.initialization == 'Tight-Frame' else "R"
        return f"{tempI} {tempL}"

    def getFilterViz(self):
        """generates plots of the filters for ['fourier','real', 'imag' ] visualizations"""
        filter_viz = {}
        phi, psi = self.load_filters()
        for mode in ['fourier','real', 'imag' ]: # visualize wavlet filters before training
            f = get_filters_visualization(psi, self.J, 8, mode=mode) 
            filter_viz[mode] = f  
        return filter_viz

    def getOneFilter(self, count, scale, mode):
        return getOneFilter(self.psi, count, scale, mode)

    def getAllFilters(self, totalCount, scale, mode):
        phi, psi = self.load_filters()
        return getAllFilters(psi, totalCount, scale, mode)

    def __init__(self, J, N, M, second_order, initialization, seed, 
                 learnable=True, lr_orientation=0.1, 
                 lr_scattering=0.1, monitor_filters=True,
                 filter_video=False, pixelwise=False):
        """Constructor for the leanable scattering nn.Module
        
        Creates scattering filters and adds them to the nn.parameters if learnable
        
        parameters: 
            J -- scale of scattering (always 2 for now)
            N -- height of the input image
            M -- width of the input image
            second_order -- 
            initilization -- the type of init: ['Tight-Frame' or 'Random']
            seed -- the random seed used to initialize the parameters
            learnable -- should the filters be learnable parameters of this model
            lr_orientation -- learning rate for the orientation of the scattering parameters
            lr_scattering -- learning rate for scattering parameters other than orientation                 
            monitor_filters -- boolean indicating whether to track filter distances from initialization
            filter_video -- whether to create filters from 

        """
        super(sn_ScatteringBase, self).__init__(J=J, shape=(M, N))

        self.second_order = second_order
        self.learnable = learnable
        self.initialization = initialization
        self.lr_scattering = lr_scattering
        self.lr_orientation = lr_orientation
        self.M_coefficient = self.M/(2**self.J)
        self.N_coefficient = self.N/(2**self.J)
        self.monitor_filters = monitor_filters
        self.filter_video = filter_video
        self.epoch = 0
        self.pixelwise = pixelwise

        L = self.L

        if second_order:
            self.n_coefficients =  L*L*J*(J-1)//2
        else: 
            self.n_coefficients =  L*L*J*(J-1)//2 + 1 + L*J  

        if initialization == "Tight-Frame":
            self.params_filters = create_filters_params(J, L, learnable) #kymatio init
        elif initialization == "Random":
            self.params_filters = create_filters_params_random(J*L, learnable) #random init
        else:
            raise InvalidInitializationException
        shape = (self.M_padded, self.N_padded,)
        ranges = [torch.arange(-(s // 2), -(s // 2) + s, dtype=torch.float) for s in shape]
        grid = torch.stack(torch.meshgrid(*ranges), 0)
        wavelets  = morlets(shape, self.params_filters[0], self.params_filters[1],
                self.params_filters[2], self.params_filters[3])


        self.register_single_filter = types.MethodType(_register_single_filter, self)
        self.psi = update_psi(self.J, self.psi, wavelets) #update psi to reflect the new conv filters
        self.register_filters()
        if learnable:
            if pixelwise:
                self.register_parameter(name='scattering_wavelets',
                        param=nn.Parameter(wavelets))
                phi, psi = self.load_filters()
                self.psi = update_psi(self.J, self.psi, self.scattering_wavelets)
                for i in range(0, len(self.params_filters)):
                    self.register_buffer(name='scattering_params_'+str(i), tensor=self.params_filters[i])
            else:
                for i in range(0, len(self.params_filters)):
                    self.params_filters[i] = nn.Parameter(self.params_filters[i])
                    self.register_parameter(name='scattering_params_'+str(i), param=self.params_filters[i])
        else:
            for i in range(0, len(self.params_filters)):
                self.register_buffer(name='scattering_params_'+str(i), tensor=self.params_filters[i])
        self.register_buffer(name='grid', tensor=grid)

        
        def updateFilters_hook(self, ip):
            """if were using learnable scattering, update the filters to reflect 
            the new parameter values obtained from gradient descent"""
            if (self.training or self.scatteringTrain) and self.learnable:
                if not pixelwise:
                    wavelets = morlets(self.grid, self.params_filters[0], 
                                        self.params_filters[1], self.params_filters[2], 
                                        self.params_filters[3])
                else:
                    wavelets = self.scattering_wavelets

                phi, psi = self.load_filters()
                self.psi = update_psi(self.J, psi, wavelets)
                self.register_filters()
                # scatteringTrain lags behind self.training
                self.scatteringTrain = self.training

                self.writeVideoFrame()
        self.register_forward_pre_hook(updateFilters_hook)

        def reshape_hook(self, x, S):
            S = S[:,:, -self.n_coefficients:,:,:]
            S = S.reshape(S.size(0), self.n_coefficients*3, S.size(3), S.size(4))
            return S
        self.register_forward_hook(reshape_hook)

        # visualization code
        self.filterTracker = {'1':[],'2':[],'3':[], 'scale':[], 'angle': []}
        self.filterGradTracker = {'angle': [],'1':[],'2':[],'3':[]}

        self.filters_plots_before = self.getFilterViz()
        self.scatteringTrain = True
                    
        if self.monitor_filters == True:
            compared_params = self.params_filters
            self.compared_params_grouped = torch.cat([x.unsqueeze(1) for x in compared_params[1:]],dim=1)
            self.compared_params_angle = compared_params[0] % (2 * np.pi)
            self.params_history = []

        if self.filter_video:
            self.videoWriters = {}
            self.videoWriters['real'] = cv2.VideoWriter('videos/scatteringFilterProgressionReal{}epochs.avi'.format("--"),
                                              cv2.VideoWriter_fourcc(*'DIVX'), 30, (160,160), isColor=True)
            self.videoWriters['imag'] = cv2.VideoWriter('videos/scatteringFilterProgressionImag{}epochs.avi'.format("--"),
                                              cv2.VideoWriter_fourcc(*'DIVX'), 30, (160,160), isColor=True)
            self.videoWriters['fourier'] = cv2.VideoWriter('videos/scatteringFilterProgressionFourier{}epochs.avi'.format("--"),
                                                 cv2.VideoWriter_fourcc(*'DIVX'), 30, (160,160), isColor=True)

    def writeVideoFrame(self):
        """Writes frames to the appropriate video writer objects"""
        if self.filter_video:
            for vizType in self.videoWriters.keys():
                temp = cv2.applyColorMap(np.array(self.getAllFilters(totalCount=16, scale=0, mode=vizType),dtype=np.uint8),cv2.COLORMAP_TURBO)
                temp = cv2.putText(temp, "Epoch {}".format(self.epoch),(2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                self.videoWriters[vizType].write(temp)

    def releaseVideoWriters(self):
        if self.filter_video:
            for vizType in self.videoWriters.keys():
                self.videoWriters[vizType].release()

    def setEpoch(self, epoch):
        self.epoch = epoch


    def checkParamDistance(self):
        """Method to checking the minimal distance between initialized filters and learned ones
        
        Euclidean distances are calculated between each filter for parameters other than orientations
        for orientations, we calculate the arc between both points on the unit circle. Then, the sum of
        these two distances becomes the distance between two filters. Finally, we use munkre's assignment 
        algorithm to compute the optimal match (I.E. the one that minizes total distance)        

        return: 
            minimal distance
        """
        tempParamsGrouped = torch.cat([x.unsqueeze(1) for x in self.params_filters[1:]],dim=1).cpu()
        tempParamsAngle = (self.params_filters[0] % (2 * np.pi)).cpu()
        self.params_history.append({'params':tempParamsGrouped,'angle':tempParamsAngle})
        return compareParams(
            params1=tempParamsGrouped,
            angles1=tempParamsAngle, 
            params2=self.compared_params_grouped,
            angles2=self.compared_params_angle
        )

    def compareParamsVisualization(self):
        """visualize the matched filters"""
        tempParamsGrouped = torch.cat([x.unsqueeze(1) for x in self.params_filters[1:]],dim=1).cpu()
        tempParamsAngle = (self.params_filters[0] % (2 * np.pi)).cpu()
        self.params_history.append({'params':tempParamsGrouped,'angle':tempParamsAngle})
        return compareParamsVisualization(
            params1=tempParamsGrouped,
            angles1=tempParamsAngle, 
            params2=self.compared_params_grouped,
            angles2=self.compared_params_angle
        )

    def saveFilterValues(self,scatteringActive):
        try:
            if scatteringActive:
                orientations1 = self.params_filters[0].detach().clone()
                self.filterTracker['1'].append(self.params_filters[1].detach().clone())
                self.filterTracker['2'].append(self.params_filters[2].detach().clone()) 
                self.filterTracker['3'].append(self.params_filters[3].detach().clone()) 
                scale = torch.mul(self.params_filters[1].detach().clone(), self.params_filters[2].detach().clone())
                self.filterTracker['scale'].append(scale) 
                self.filterTracker['angle'].append(orientations1) 

            else:
                self.filterGradTracker['angle'].append(torch.zeros(self.params_filters[1].shape[0])) 
                self.filterGradTracker['1'].append(torch.zeros(self.params_filters[1].shape[0])) 
                self.filterGradTracker['2'].append(torch.zeros(self.params_filters[1].shape[0]))
                self.filterGradTracker['3'].append(torch.zeros(self.params_filters[1].shape[0]))
        except Exception:
            pass


    def saveFilterGrads(self,scatteringActive):
        try:
            if scatteringActive:
                self.filterGradTracker['angle'].append(self.params_filters[0].grad.clone()) 
                self.filterGradTracker['1'].append(self.params_filters[1].grad.clone()) 
                self.filterGradTracker['2'].append(self.params_filters[2].grad.clone()) 
                self.filterGradTracker['3'].append(self.params_filters[3].grad.clone()) 
            else:
                self.filterGradTracker['angle'].append(torch.zeros(self.params_filters[1].shape[0])) 
                self.filterGradTracker['1'].append(torch.zeros(self.params_filters[1].shape[0])) 
                self.filterGradTracker['2'].append(torch.zeros(self.params_filters[1].shape[0]))
                self.filterGradTracker['3'].append(torch.zeros(self.params_filters[1].shape[0]))
        except Exception:
            pass



    def plotFilterGrads(self):
        """plots the graph of the filter gradients"""
        filterNum = self.params_filters[1].shape[0]
        col = 8
        row = int(filterNum/col)
        size = (80, 10*row,)

        f, axarr = plt.subplots(row, col, figsize=size) # create plots

        for x in range(filterNum):#iterate over all the filters
            temp = {
                'orientation1': [float(filters[x].cpu().numpy()) for filters in self.filterGradTracker['angle']],
                'xis': [float(filters[x].cpu().numpy())  for filters in self.filterGradTracker['1']],
                'sigmas': [float(filters[x].cpu().numpy())  for filters in self.filterGradTracker['2']],
                'slant': [float(filters[x].cpu().numpy())  for filters in self.filterGradTracker['3']],
            }

            axarr[int(x/col),x%col].plot([x for x in range(len(temp['orientation1']))],temp['orientation1'],color='red', label='orientation1')
            axarr[int(x/col),x%col].plot([x for x in range(len(temp['xis']))],temp['xis'],color='green', label='xis')
            axarr[int(x/col),x%col].plot([x  for x in range(len(temp['sigmas']))],temp['sigmas'],color='yellow', label='sigma')
            axarr[int(x/col),x%col].plot([x for x in range(len(temp['slant']))],temp['slant'],color='orange', label='slant')
            axarr[int(x/col),x%col].legend()

        return f

    
    def plotFilterValues(self):
        """plots the graph of the filter values"""
        filterNum = self.params_filters[1].shape[0]
        col = 8
        row = int(filterNum/col)
        size = (80, 10*row,)

        f, axarr = plt.subplots(row, col, figsize=size) # create plots

        for x in range(filterNum):#iterate over all the filters
            temp = {
                'orientation1': [float(filters[x].cpu().numpy()) for filters in self.filterTracker['angle']],
                'xis': [float(filters[x].cpu().numpy())  for filters in self.filterTracker['1']],
                'sigmas': [float(filters[x].cpu().numpy())  for filters in self.filterTracker['2']],
                'slant': [float(filters[x].cpu().numpy())  for filters in self.filterTracker['3']],
                'scale': [float(filters[x].cpu().numpy())  for filters in self.filterTracker['scale']],
            }

            axarr[int(x/col),x%col].plot([x for x in range(len(temp['orientation1']))],temp['orientation1'],color='red', label='theta')
            axarr[int(x/col),x%col].plot([x for x in range(len(temp['xis']))],temp['xis'],color='green', label='xis')
            axarr[int(x/col),x%col].plot([x for x in range(len(temp['sigmas']))],temp['sigmas'],color='yellow', label='sigma')
            axarr[int(x/col),x%col].plot([x for x in range(len(temp['slant']))],temp['slant'],color='orange', label='slant')
            axarr[int(x/col),x%col].plot([x for x in range(len(temp['scale']))],temp['scale'],color='black', label='scale')
            axarr[int(x/col),x%col].legend()

        return f
        

    def plotParameterValues(self):
        size = (10, 10)
        f, axarr = plt.subplots(2, 2, figsize=size) # create plots
        plt.subplots_adjust(hspace=0.35, wspace=0.35)
        label = ['theta','xis','sigma','slant']

        for idx,param in enumerate(['angle',"1",'2','3']):#iterate over all the parameters
            for idx2,filter in enumerate(torch.stack(self.filterTracker[param]).T):
                filter = filter.cpu().numpy()
                axarr[int(idx/2),idx%2].plot([x for x in range(len(filter))],filter)
            axarr[int(idx/2),idx%2].set_title(label[idx], fontsize=16)
            axarr[int(idx/2),idx%2].set_xlabel('Epoch', fontsize=12) # Or ITERATION to be more precise
            axarr[int(idx/2),idx%2].set_ylabel('Value', fontsize=12)
            

        return f
