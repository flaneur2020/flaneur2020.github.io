package main

import "fmt"

func add(a int, b int) int {
	return a + b
}

func main() {
	val := add(1, 2)
	fmt.Printf("test")
	fmt.Printf("test: %d", val)
}
