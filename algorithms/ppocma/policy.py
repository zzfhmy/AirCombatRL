import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class Policy(nn.Module):
    def __init__(self, stateDim:int, actionDim:int, actionMinLimit:np.array, actionMaxLimit:np.array, mode="PPO-CMA"
                 , entropyLossWeight=0, networkDepth=2, networkUnits=64, networkActivation="lrelu"
                 , networkSkips=False, networkUnitNormInit=True, usePPOLoss=False, separateVarAdapt=False
                 , learningRate=0.001, minSigma=0.01, useSigmaSoftClip=True, PPOepsilon=0.2, piEpsilon=0, nHistory=1
                 , globalVariance=False, trainableGlobalVariance=True, useGradientClipping=False
                 , maxGradientNorm=0.5, negativeAdvantageAvoidanceSigma=0):
        
        super(Policy, self).__init__()
        
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.actionMinLimit = torch.tensor(actionMinLimit, dtype=torch.float32)
        self.actionMaxLimit = torch.tensor(actionMaxLimit, dtype=torch.float32)
        self.mode = mode
        self.entropyLossWeight = entropyLossWeight
        self.networkDepth = networkDepth
        self.networkUnits = networkUnits
        self.networkActivation = networkActivation
        self.networkSkips = networkSkips
        self.networkUnitNormInit = networkUnitNormInit
        self.usePPOLoss = usePPOLoss
        self.separateVarAdapt = separateVarAdapt
        self.learningRate = learningRate
        self.minSigma = minSigma
        self.useSigmaSoftClip = useSigmaSoftClip
        self.PPOepsilon = PPOepsilon
        self.piEpsilon = piEpsilon
        self.nHistory = nHistory
        self.globalVariance = globalVariance
        self.trainableGlobalVariance = trainableGlobalVariance
        self.useGradientClipping = useGradientClipping
        self.maxGradientNorm = maxGradientNorm
        self.negativeAdvantageAvoidanceSigma = negativeAdvantageAvoidanceSigma
        
        maxSigma = 1.0 * (self.actionMaxLimit - self.actionMinLimit)
        
        self.usedSigmaSum = 0
        self.usedSigmaSumCounter = 0
        
        if stateDim == 0:
            self.policyMean = nn.Parameter(torch.zeros(actionDim, dtype=torch.float32))
            self.policyLogVar = nn.Parameter(torch.log(torch.square(0.5 * (self.actionMaxLimit - self.actionMinLimit))) * torch.ones(actionDim, dtype=torch.float32))
            self.globalLogVarVariable = self.policyLogVar
        else:
            self.policyMean, self.policyLogVar = self._build_network(stateDim, actionDim)
        
        if self.useSigmaSoftClip:
            self.maxLogVar = torch.log(maxSigma * maxSigma)
            self.minLogVar = torch.log(self.minSigma * self.minSigma)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.learningRate)
        
        self.history = deque()
        self.initialized = False
        
    def _build_network(self, stateDim, actionDim):
        layers = []
        input_dim = stateDim
        for _ in range(self.networkDepth):
            layers.append(nn.Linear(input_dim, self.networkUnits))
            if self.networkActivation == "lrelu":
                layers.append(nn.LeakyReLU())
            elif self.networkActivation == "relu":
                layers.append(nn.ReLU())
            input_dim = self.networkUnits
        
        if self.separateVarAdapt or self.globalVariance:
            mean_layer = nn.Linear(input_dim, actionDim)
            var_layer = nn.Linear(input_dim, actionDim)
        else:
            output_layer = nn.Linear(input_dim, actionDim * 2)
            mean_layer = output_layer
            var_layer = output_layer
        
        if self.networkUnitNormInit:
            for layer in layers:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight)
        
        self.mean_network = nn.Sequential(*layers, mean_layer)
        self.var_network = nn.Sequential(*layers, var_layer)
        
        return self.mean_network, self.var_network

    def forward(self, state):
        if self.stateDim == 0:
            policyMean = self.policyMean
            policyLogVar = self.policyLogVar
        else:
            policyMean = self.mean_network(state)
            policyLogVar = self.var_network(state)
        
        policyMean = torch.sigmoid(policyMean) * (self.actionMaxLimit - self.actionMinLimit) + self.actionMinLimit
        
        if self.useSigmaSoftClip:
            policyLogVar = torch.sigmoid(policyLogVar) * (self.maxLogVar - self.minLogVar) + self.minLogVar
        
        policyVar = torch.exp(policyLogVar)
        policySigma = torch.sqrt(policyVar)
        
        return policyMean, policyVar, policyLogVar, policySigma
    
    def loss(self, policyMean, policyVar, policyLogVar, actionIn, advantagesIn, oldPolicyMean=None, logPiOldIn=None):
        if self.usePPOLoss:
            logPi = torch.sum(-0.5 * torch.square(actionIn - policyMean) / policyVar - 0.5 * policyLogVar, dim=1)
            if self.piEpsilon == 0:
                r = torch.exp(logPi - logPiOldIn)
            else:
                r = torch.exp(logPi) / (self.piEpsilon + torch.exp(logPiOldIn))
            perSampleLoss = torch.min(r * advantagesIn, torch.clamp(r, 1 - self.PPOepsilon, 1 + self.PPOepsilon) * advantagesIn)
            policyLoss = -torch.mean(perSampleLoss)
            if self.entropyLossWeight > 0:
                policyLoss -= self.entropyLossWeight * 0.5 * torch.mean(torch.sum(policyLogVar, dim=1))
            return policyLoss
        else:
            # Detach gradients for policy mean and variance
            policyNoGrad = policyMean.detach()
            policyVarNoGrad = policyVar.detach()
            policyLogVarNoGrad = policyLogVar.detach()
            
            # Calculate log probability without gradient for mean and variance
            logpNoMeanGrad = -torch.sum(0.5 * torch.square(actionIn - policyNoGrad) / policyVar + 0.5 * policyLogVar, dim=1)
            logpNoVarGrad = -torch.sum(0.5 * torch.square(actionIn - policyMean) / policyVarNoGrad + 0.5 * policyLogVarNoGrad, dim=1)
            
            # Calculate positive advantages
            posAdvantages = torch.relu(advantagesIn)
            
            # Calculate policy losses
            policySigmaLoss = -torch.mean(posAdvantages * logpNoMeanGrad)
            policyMeanLoss = -torch.mean(posAdvantages * logpNoVarGrad)
            
            # If negative advantage avoidance is enabled, compute the mirrored log probability
            if self.negativeAdvantageAvoidanceSigma > 0:
                negAdvantages = torch.relu(-advantagesIn)
                mirroredAction = oldPolicyMean - (actionIn - oldPolicyMean)
                logpNoVarGradMirrored = -torch.sum(0.5 * torch.square(mirroredAction - policyMean) / policyVarNoGrad + 0.5 * policyLogVarNoGrad, dim=1)
                effectiveKernelSqWidth = self.negativeAdvantageAvoidanceSigma ** 2 * policyVarNoGrad
                avoidanceKernel = torch.mean(torch.exp(-0.5 * torch.square(actionIn - oldPolicyMean) / effectiveKernelSqWidth), dim=1)
                policyMeanLoss -= torch.mean((negAdvantages * avoidanceKernel) * logpNoVarGradMirrored)
            
            return policySigmaLoss, policyMeanLoss

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.useGradientClipping:
            nn.utils.clip_grad_norm_(self.parameters(), self.maxGradientNorm)
        self.optimizer.step()

    def init_policy(self, stateMean, stateSd, actionMean, actionSd, nMinibatch=64, nBatches=4000, verbose=True):
        for batchIdx in range(nBatches):
            states = np.random.normal(stateMean, stateSd, size=[nMinibatch, self.stateDim])
            states = torch.tensor(states, dtype=torch.float32)
            actionIn = torch.tensor(actionMean, dtype=torch.float32).view(1, -1).repeat(nMinibatch, 1)
            initSigmaIn = torch.tensor(actionSd, dtype=torch.float32).view(1, -1).repeat(nMinibatch, 1)
            
            policyMean, policyVar, policyLogVar, policySigma = self.forward(states)
            initLoss = torch.mean((actionIn - policyMean)**2) + torch.mean((initSigmaIn - policySigma)**2)
            
            self.optimize(initLoss)
            
            if verbose and (batchIdx % 100 == 0):
                print(f"Initializing policy with random Gaussian data, batch {batchIdx}/{nBatches}, loss {initLoss.item()}")
        
        self.initialized = True

    def train_policy(self, states, actions, advantages, nMinibatch, nEpochs, nBatches=0, stateOffset=0, stateScale=1, verbose=True):
        assert np.all(np.isfinite(states))
        assert np.all(np.isfinite(actions))
        assert np.all(np.isfinite(advantages))
        assert states.shape[0] == actions.shape[0]
        assert states.shape[0] == advantages.shape[0]
        
        if self.initialized:
            assert states.shape[1] == self.stateDim
        
        assert actions.shape[1] == self.actionDim
        assert advantages.shape[1] == 1
        
        nSamples = states.shape[0]
        
        if nBatches == 0:
            nBatches = int(np.ceil(nSamples / nMinibatch))
        
        statesIn = torch.tensor(states, dtype=torch.float32)
        actionIn = torch.tensor(actions, dtype=torch.float32)
        advantagesIn = torch.tensor(advantages, dtype=torch.float32)
        
        oldPolicyMean, _, _, _ = self.forward(statesIn)
        logPiOldIn = -torch.sum(0.5 * (actionIn - oldPolicyMean)**2 / policyVar + 0.5 * policyLogVar, dim=1, keepdim=True)
        
        for epoch in range(nEpochs):
            permutedIndices = torch.randperm(nSamples)
            shuffledStatesIn = statesIn[permutedIndices]
            shuffledActionIn = actionIn[permutedIndices]
            shuffledAdvantagesIn = advantagesIn[permutedIndices]
            
            for batchIdx in range(nBatches):
                startIdx = batchIdx * nMinibatch
                endIdx = min(startIdx + nMinibatch, nSamples)
                
                batchStatesIn = shuffledStatesIn[startIdx:endIdx]
                batchActionIn = shuffledActionIn[startIdx:endIdx]
                batchAdvantagesIn = shuffledAdvantagesIn[startIdx:endIdx]
                
                policyMean, policyVar, policyLogVar, _ = self.forward(batchStatesIn)
                
                if self.usePPOLoss:
                    loss = self.loss(policyMean, policyVar, policyLogVar, batchActionIn, batchAdvantagesIn, oldPolicyMean[batchIdx], logPiOldIn[batchIdx])
                else:
                    sigmaLoss, meanLoss = self.loss(policyMean, policyVar, policyLogVar, batchActionIn, batchAdvantagesIn)
                    loss = sigmaLoss + meanLoss
                
                self.optimize(loss)
                
                if verbose and (batchIdx % 100 == 0):
                    print(f"Training policy, epoch {epoch}/{nEpochs}, batch {batchIdx}/{nBatches}, loss {loss.item()}")
        
        return loss
