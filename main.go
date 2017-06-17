package main

import (
	"fmt"
	"github.com/vadimlarionov/multilayer-neural-network/network"
	"github.com/vadimlarionov/multilayer-neural-network/utils"
	"os"
)

func main() {
	trainFile := "/home/vadim/personal/digitrecognizer/train.csv"
	trainData, err := utils.ReadDataset(trainFile, -1)
	if err != nil {
		fmt.Printf("Can't read %s: %s\n", trainFile, err)
		os.Exit(1)
	}
	trainData = utils.Normalize(trainData, 255, false)

	b := network.NewBuilder(28*28, 10)
	//b.AddLayer(50).AddLayer(10)
	nn, err := b.Build(true)
	if err != nil {
		fmt.Printf("Can't build neural network: %s\n", err)
		os.Exit(1)
	}

	trainer := network.BackpropagationTrainer{Nn: nn, LearningRate: 0.01}
	trainer.Train(trainData, 15)

	testFile := "/home/vadim/personal/digitrecognizer/test.csv"
	testData, err := utils.ReadDataset(testFile, -1)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	testData = utils.Normalize(testData, 255, true)

	results := recognize(nn, testData)
	utils.WriteResult("out.csv", results)
}

func recognize(nn *network.NeuralNetwork, testData [][]float64) (results []int) {
	results = make([]int, len(testData))
	for i, inputs := range testData {
		results[i] = nn.Recognize(inputs)
	}
	return results
}
