package main

import "fmt"

func exists(board [][]byte, word string) bool {
	return iterFistChars(board, word)
}

func iterFistChars(board [][]byte, word string) bool {
	lenI := len(board)
	for i := 0; i < lenI; i++ {
		lenJ := len(board[i])
		for j := 0; j < lenJ; j++ {
			if word[0] == board[i][j] {
				visited := makeVisited(lenI, lenJ)
				ok := iterTrace(board, i, j, lenI, lenJ, visited, word)
				if ok {
					return true
				}
			}
		}
	}
	return false
}

func iterTrace(board [][]byte, i, j int, lenI, lenJ int, visited [][]bool, word string) bool {
	if len(word) == 0 {
		return true
	}

	if board[i][j] != word[0] {
		return false
	}

	if len(word) == 1 {
		return true
	}

	visited[i][j] = true

	if i+1 < lenI && !visited[i+1][j] {
		ok := iterTrace(board, i+1, j, lenI, lenJ, visited, word[1:])
		if ok {
			return true
		}
	}

	if i-1 >= 0 && !visited[i-1][j] {
		ok := iterTrace(board, i-1, j, lenI, lenJ, visited, word[1:])
		if ok {
			return true
		}
	}

	if j+1 < lenJ && !visited[i][j+1] {
		ok := iterTrace(board, i, j+1, lenI, lenJ, visited, word[1:])
		if ok {
			return true
		}
	}

	if j-1 >= 0 && !visited[i][j-1] {
		ok := iterTrace(board, i, j-1, lenI, lenJ, visited, word[1:])
		if ok {
			return true
		}
	}

	return false
}

func makeVisited(lenI, lenJ int) [][]bool {
	m := make([][]bool, lenI)
	for i := 0; i < lenI; i++ {
		m[i] = make([]bool, lenJ)
	}
	return m
}

func main() {
	board := [][]byte{
		{'A', 'B', 'C', 'E'},
		{'S', 'F', 'C', 'S'},
		{'A', 'D', 'E', 'E'},
	}
	fmt.Printf("%d", exists(board, "ABCB"))
}
