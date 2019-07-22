package main

import "fmt"

const MaxUint = ^uint(0)
const MaxInt = int(MaxUint >> 1)

func dpMinSum(grid [][]int, maxi, maxj int) {
	for i := maxi; i >= 0; i-- {
		for j := maxj; j >= 0; j-- {
			if i < maxi && j < maxj {
				grid[i][j] += minInt(grid[i][j+1], grid[i+1][j])
			} else if j < maxj {
				grid[i][j] += grid[i][j+1]
			} else if i < maxi {
				grid[i][j] += grid[i+1][j]
			}
		}
	}
}

func naiveCalcMinSum(grid [][]int, x, y int) int {
	ns := []int{}
	if x-1 >= 0 {
		n := grid[x][y] + naiveCalcMinSum(grid, x-1, y)
		ns = append(ns, n)
	}
	if y-1 >= 0 {
		n := grid[x][y] + naiveCalcMinSum(grid, x, y-1)
		ns = append(ns, n)
	}
	if len(ns) == 0 {
		return grid[x][y]
	} else if len(ns) == 1 {
		return ns[0]
	} else {
		return minInt(ns[0], ns[1])
	}
}

func minInt(a, b int) int {
	if a > b {
		return b
	}
	return a
}

func main() {
	grid := [][]int{
		{1, 3, 1},
		{1, 5, 1},
		{4, 2, 1},
	}
	dpMinSum(grid, 2, 2)
	fmt.Printf("minpath: %v \nn: %d\n", grid[0][0])
}
