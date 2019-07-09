package main

import "fmt"

func numsToSpans(nums []int) {
	j := 0
	for i := 0; i < len(nums); i++ {
		for j = i; j < len(nums) && nums[i] == nums[j]; j++ {
		}
		r := j - 1
		fmt.Printf("i:%d j:%d\n", i, r)
		i = r
	}
}

func removeDuplicates(nums []int) int {
	j := 0
	t := 0
	for i := 0; i < len(nums); i++ {
		for j = i; j < len(nums) && nums[i] == nums[j]; j++ {
		}
		r := j - 1
		fmt.Printf("i:%d j:%d\n", i, r)
		if r-i >= 1 {
			nums[t] = nums[i]
			nums[t+1] = nums[i]
			t += 2
		} else if r-i == 0 {
			nums[t] = nums[i]
			t += 1
		}
		i = r
	}
	return t
}

func main() {
	nums := []int{1, 1, 1, 1, 2, 3, 3, 3}
	n := removeDuplicates(nums)
	fmt.Printf("nums: %v n: %d\n", nums[0:n], n)

	nums = []int{1, 1, 1, 2, 2, 3, 4}
	n = removeDuplicates(nums)
	fmt.Printf("nums: %v n: %d\n", nums[0:n], n)

	nums = []int{1, 1, 1, 2, 2, 3, 4}
	n = removeDuplicates(nums)
	fmt.Printf("nums: %v n: %d\n", nums[0:n], n)

	nums = []int{1, 1, 1, 2, 2, 3, 4, 4, 4}
	n = removeDuplicates(nums)
	fmt.Printf("nums: %v n: %d\n", nums[0:n], n)
}
