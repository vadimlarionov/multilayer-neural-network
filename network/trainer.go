package network

type BackpropagationTrainer struct {
	nn           *NeuralNetwork
	learningRate float64
}

type layerData struct {
	outputs []float64
	delta   []float64
}

func (tr *BackpropagationTrainer) train(classNumber int, inputs []float64) {
	layersData := tr.forwardPropagation(inputs)

	tr.prepareDelta(classNumber, layersData)

	for layerIndex := len(tr.nn.layers) - 1; layerIndex >= 0; layerIndex-- {
		for neuronIndex, n := range tr.nn.layers[layerIndex].neurons {
			var neuronInputs []float64
			if layerIndex > 0 {
				neuronInputs = layersData[layerIndex-1].outputs
			} else {
				neuronInputs = inputs
			}

			n.updateWeights(tr.learningRate, layersData[layerIndex].delta[neuronIndex], neuronInputs)
		}
	}
}

func (tr *BackpropagationTrainer) prepareDelta(classNumber int, layersData []*layerData) {
	for layerIndex := len(tr.nn.layers) - 1; layerIndex >= 0; layerIndex-- {
		for neronIndex, outValue := range layersData[layerIndex].outputs {
			n := tr.nn.layers[layerIndex].neurons[neronIndex]
			var delta float64
			if layerIndex != len(tr.nn.layers)-1 {
				delta = deltaOutputLayer(n, neronIndex, classNumber, outValue)
			} else {
				delta = deltaHiddenLayer(n, neronIndex, tr.nn.layers[layerIndex+1],
					layersData[layerIndex+1], outValue)
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
	layersData = make([]*layerData, len(tr.nn.layers))
	for layerIndex, l := range tr.nn.layers {
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
