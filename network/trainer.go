package network

import (
	"fmt"
	"github.com/vadimlarionov/multilayer-neural-network/utils"
)

type BackpropagationTrainer struct {
	Nn           *NeuralNetwork
	LearningRate float64
}

type layerData struct {
	outputs []float64
	delta   []float64
}

func (tr *BackpropagationTrainer) Train(data [][]float64, maxEpochs int) {
	for i := 0; i < maxEpochs; i++ {
		rightRecognized, epochErr := tr.trainEpoch(data)
		percent := float64(rightRecognized) * 100.0 / float64(len(data))
		fmt.Printf("Epoch: %d; Right recognized: %f; Error: %f\n", i, percent, epochErr)
	}
}

func (tr *BackpropagationTrainer) trainEpoch(data [][]float64) (rightPredicted int, epochErr float64) {
	for _, dataset := range data {
		classNumber := int(dataset[0])
		inputs := dataset[1:]
		if tr.trainDataset(classNumber, inputs) {
			rightPredicted++
		}
	}
	return rightPredicted, 0.0
}

func (tr *BackpropagationTrainer) trainDataset(classNumber int, inputs []float64) (predicted bool) {
	layersData := tr.forwardPropagation(inputs)

	tr.prepareDelta(classNumber, layersData)

	for layerIndex := len(tr.Nn.layers) - 1; layerIndex >= 0; layerIndex-- {
		for neuronIndex, n := range tr.Nn.layers[layerIndex].neurons {
			var neuronInputs []float64
			if layerIndex > 0 {
				neuronInputs = layersData[layerIndex-1].outputs
			} else {
				neuronInputs = inputs
			}

			n.updateWeights(tr.LearningRate, layersData[layerIndex].delta[neuronIndex], neuronInputs)
		}
	}

	return utils.IndexMaxElement(layersData[len(tr.Nn.layers)-1].outputs) == classNumber
}

func (tr *BackpropagationTrainer) prepareDelta(classNumber int, layersData []*layerData) {
	numLayers := len(tr.Nn.layers)
	for layerIndex := numLayers - 1; layerIndex >= 0; layerIndex-- {
		for neronIndex, outValue := range layersData[layerIndex].outputs {
			n := tr.Nn.layers[layerIndex].neurons[neronIndex]
			var delta float64
			if layerIndex < numLayers-1 {
				delta = deltaHiddenLayer(n, neronIndex, tr.Nn.layers[layerIndex+1],
					layersData[layerIndex+1], outValue)
			} else {
				delta = deltaOutputLayer(n, neronIndex, classNumber, outValue)
			}
			layersData[layerIndex].delta[neronIndex] = delta
		}
	}
}

func deltaOutputLayer(n *Neuron, neuronIndex, classNumber int, outValue float64) (delta float64) {
	if neuronIndex != classNumber {
		delta = 0 - outValue
	} else {
		delta = 1 - outValue
	}
	return delta * n.activator.derivative(outValue)
}

func deltaHiddenLayer(n *Neuron, neuronIndex int, nextLayer *Layer,
	nextLayerData *layerData, outValue float64) (delta float64) {

	for i, nextLayerDelta := range nextLayerData.delta {
		delta += nextLayerDelta * nextLayer.neurons[i].weights[neuronIndex]
	}
	return delta * n.activator.derivative(outValue)
}

func (tr *BackpropagationTrainer) forwardPropagation(inputs []float64) (layersData []*layerData) {
	layersData = make([]*layerData, len(tr.Nn.layers))
	for layerIndex, l := range tr.Nn.layers {
		ld := layerData{}
		ld.delta = make([]float64, len(l.neurons))
		ld.outputs = make([]float64, len(l.neurons))

		if layerIndex > 0 {
			ld.outputs = l.Activate(layersData[layerIndex-1].outputs)
		} else {
			ld.outputs = l.Activate(inputs)
		}

		layersData[layerIndex] = &ld
	}
	return layersData
}
