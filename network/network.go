package network

import (
	"errors"
	"github.com/vadimlarionov/multilayer-neural-network/utils"
	"math"
)

type Activator interface {
	activate(x float64) float64
	derivative(out float64) float64
}

type SigmoidActivator struct {
}

func (s *SigmoidActivator) activate(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (s *SigmoidActivator) derivative(out float64) float64 {
	value := s.activate(out)
	return value * (1 - value)
}

type Neuron struct {
	weights   []float64
	bias      float64
	activator Activator
}

func (n *Neuron) Activate(inputs []float64) float64 {
	value := -n.bias
	for i, input := range inputs {
		value += n.weights[i] * input
	}
	return n.activator.activate(value)
}

func (n *Neuron) updateWeights(learningRate, delta float64, inputs []float64) {
	deltaBias := delta * learningRate
	n.bias -= deltaBias
	for i := range n.weights {
		n.weights[i] += deltaBias * inputs[i]
	}
}

func NewNeuron(numInputs int) *Neuron {
	n := Neuron{}
	n.weights = make([]float64, numInputs)
	n.activator = &SigmoidActivator{}
	return &n
}

type Layer struct {
	neurons []*Neuron
}

func (l *Layer) Activate(inputs []float64) []float64 {
	result := make([]float64, len(l.neurons))
	for i, n := range l.neurons {
		result[i] = n.Activate(inputs)
	}
	return result
}

func (l *Layer) updateWeights(neuronIndex int, learningRate, delta float64, inputs []float64) {
	l.neurons[neuronIndex].updateWeights(learningRate, delta, inputs)
}

func NewLayer(numNeurons, numInputs int) *Layer {
	l := Layer{}
	l.neurons = make([]*Neuron, numNeurons)
	for i := 0; i < numNeurons; i++ {
		l.neurons[i] = NewNeuron(numInputs)
	}
	return &l
}

type NeuralNetwork struct {
	numInputs  int
	numOutputs int
	layers     []*Layer
}

func (nn *NeuralNetwork) Recognize(inputs []float64) (classNumber int) {
	result := inputs
	for _, layer := range nn.layers {
		result = layer.Activate(result)
	}

	return utils.IndexMaxElement(result)
}

type Builder struct {
	numInputs    int
	numOutputs   int
	hiddenLayers []int
}

func (b *Builder) AddLayer(numNeurons int) *Builder {
	b.hiddenLayers = append(b.hiddenLayers, numNeurons)
	return b
}

func (b *Builder) Build() (nn *NeuralNetwork, err error) {
	if b.numInputs <= 0 {
		return nil, errors.New("The number of inputs must have a positive value")
	}

	if b.numOutputs <= 0 {
		return nil, errors.New("The number of outputs must have a positive value")
	}

	nn = &NeuralNetwork{numInputs: b.numInputs, numOutputs: b.numOutputs}
	nn.layers = make([]*Layer, len(b.hiddenLayers)+1)

	numInputs := b.numInputs
	for i, numNeurons := range b.hiddenLayers {
		if numNeurons <= 0 {
			return nil, errors.New("The number of hidden neurons <= 0")
		}
		nn.layers[i] = NewLayer(numNeurons, numInputs)
		numInputs = numNeurons
	}

	nn.layers[len(b.hiddenLayers)] = NewLayer(b.numOutputs, numInputs)
	return nn, nil
}

func NewBuilder(numInputs, numOutputs int) *Builder {
	b := Builder{numInputs: numInputs, numOutputs: numOutputs}
	b.hiddenLayers = make([]int, 0, 10)
	return &b
}
