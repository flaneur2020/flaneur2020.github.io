package hello

class Solution {
    fun threeSum(originNums: List<Int>): List<List<Int>> {
        var nums = originNums.toMutableList()
        var results: MutableSet<List<Int>> = mutableSetOf()
        nums.sort()
        println("sortedNum: ${nums}")
        for (i in 0..nums.count()-1){
            for (j in (nums.count()-1) downTo (i+1)) {
                var numsRange = nums.subList(i+1, j)
                val expectedNum = 0 - (nums[i] + nums[j])
                val idx = numsRange.binarySearch(expectedNum)
                println("numsRange: ${numsRange}, i: ${i}, j: ${j}, expectedNum: ${expectedNum}, idx: ${idx}")
                if (idx < 0) {
                    continue
                }
                results.add(listOf(nums[i], nums[j], expectedNum))
            }
        }
        return results.toList()
    }
}

// kotlinc 29.kt -include-runtime -d /tmp/29.jar && java -jar /tmp/29.jar
fun main() {
    var nums = listOf(8,-15,-2,-13,8,5,6,-3,-9,3,6,-6,8,14,-9,-8,-9,-6,-14,5,-7,3,-10,-14,-12,-11,12,-15,-1,12,8,-8,12,13,-13,-3,-5,0,10,2,-11,-7,3,4,-8,9,3,-10,11,5,10,11,-7,7,12,-12,3,1,11,9,-9,-4,9,-12,-6,11,-7,4,-4,-12,13,-8,-12,2,3,-13,-12,-8,14,14,12,9,10,12,-6,-1,8,4,8,4,-1,14,-15,-7,9,-14,11,9,5,14)
    var solution = Solution()
    var results = solution.threeSum(nums)
    println("results: ${results}")
}
