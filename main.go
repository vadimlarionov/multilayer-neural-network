package main

import (
	"fmt"
	"github.com/vadimlarionov/multilayer-neural-network/network"
)

func main() {

	b := network.NewBuilder(28*28, 10)
	nn, err := b.Build()
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Printf("%v\n", nn)
}
