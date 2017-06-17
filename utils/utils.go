package utils

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
)

func ReadDataset(fileName string, limit int) (data [][]float64, err error) {
	f, err := os.Open(fileName)
	if err != nil {
		fmt.Printf("Read error: %s\n", err)
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(bufio.NewReader(f))
	// Skip the header
	if _, err = r.Read(); err == io.EOF {
		fmt.Println("Content is not found")
		return nil, err
	}

	unlimited := false
	if limit == -1 {
		unlimited = true
	}

	readCounter := 0
	for i := 0; i < limit || unlimited; i++ {
		record, err := r.Read()
		if err != nil {
			if err == io.EOF {
				break
			} else {
				fmt.Printf("Read error: %s\n", err)
				return nil, err
			}
		}

		dataset := make([]float64, len(record))
		for j, valueString := range record {
			value, err := strconv.ParseFloat(valueString, 64)
			if err != nil {
				fmt.Printf("Parse input error: %s\n", err)
				return nil, err
			}
			dataset[j] = value
		}

		data = append(data, dataset)
		readCounter++
		if readCounter%1000 == 0 {
			fmt.Printf("Read %d records\n", readCounter)
		}
	}
	return data, nil
}

func Normalize(data [][]float64, maxValue float64) [][]float64 {
	for _, dataset := range data {
		for i, value := range dataset {
			if i != 0 {
				dataset[i] = value / maxValue
			}
		}
	}
	return data
}

func WriteResult(fileName string, results []int) (err error) {
	f, err := os.Create(fileName)
	if err != nil {
		fmt.Printf("Can't create file: %s\n", err)
		return err
	}
	defer f.Close()

	writer := csv.NewWriter(f)
	defer writer.Flush()

	if err = writer.Write([]string{"ImageId", "Label"}); err != nil {
		fmt.Printf("Can't write header: %s\n", err)
		return err
	}

	buf := make([]string, 2)
	for i, value := range results {
		buf[0] = strconv.Itoa(i + 1)
		buf[1] = strconv.Itoa(value)
		if err = writer.Write(buf); err != nil {
			fmt.Printf("Can't write row: %s\n", err)
			return err
		}
	}
	return nil
}

func IndexMaxElement(slice []float64) (index int) {
	if len(slice) == 0 {
		return -1
	}
	maxElement := slice[0]
	for i, element := range slice {
		if element > maxElement {
			maxElement = element
			index = i
		}
	}
	return index
}
