package main

import (
	"fmt"
)

func permute(nums []int, start int, rs *[][]int) {
	if start >= len(nums) {
		return
	}

	tmp := make([]int, len(nums))
	copy(tmp, nums)
	nums = tmp

	if start == len(nums)-1 {
		*rs = append(*rs, nums)
		return
	}

	for i := start; i < len(nums); i++ {
		tmp := nums[start]
		nums[start] = nums[i]
		nums[i] = tmp
		permute(nums, start+1, rs)
	}
}

func main() {
	rs := [][]int{}
	arr := []int{1, 2, 3}
	permute(arr, 0, &rs)
	fmt.Printf("%v\n", rs)
}
