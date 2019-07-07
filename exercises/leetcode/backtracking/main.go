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

	v := copyVisited(visited)
	v[i][j] = true

	if len(word) == 1 {
		return true
	}

	if i+1 < lenI && !v[i+1][j] {
		ok := iterTrace(board, i+1, j, lenI, lenJ, v, word[1:])
		if ok {
			return true
		}
	}

	if i-1 >= 0 && !v[i-1][j] {
		ok := iterTrace(board, i-1, j, lenI, lenJ, v, word[1:])
		if ok {
			return true
		}
	}

	if j+1 < lenJ && !v[i][j+1] {
		ok := iterTrace(board, i, j+1, lenI, lenJ, v, word[1:])
		if ok {
			return true
		}
	}

	if j-1 >= 0 && !v[i][j-1] {
		ok := iterTrace(board, i, j-1, lenI, lenJ, v, word[1:])
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

func copyVisited(visited [][]bool) [][]bool {
	m := make([][]bool, len(visited))
	for i := 0; i < len(visited); i++ {
		m[i] = make([]bool, len(visited[i]))
		for j := 0; j < len(visited[i]); j++ {
			m[i][j] = visited[i][j]
		}
	}
	return m
}

func main() {
	board1 := [][]byte{
		{'A', 'B', 'C', 'E'},
		{'S', 'F', 'C', 'S'},
		{'A', 'D', 'E', 'E'},
	}
	fmt.Printf("%v", board1)

	board2 := [][]byte{
		{'A', 'B', 'C', 'E'},
		{'S', 'F', 'E', 'S'},
		{'A', 'D', 'E', 'E'},
	}
	fmt.Printf("%d", exists(board2, "ABCESEEEFS"))
}
