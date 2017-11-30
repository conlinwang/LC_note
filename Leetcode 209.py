class Solution:
    # @param {integer} s
    # @param {integer[]} nums
    # @return {integer}
    def minSubArrayLen(self, s, nums):
        size = len(nums)
        start, end, num_sum = 0, 0, 0
        bestAns = size + 1
        
        while True:
            if num_sum < s:
                if end >= size:
                    break
                num_sum += nums[end]
                end += 1
            else:
                if start > end:
                    break
                bestAns = min(end - start, bestAns)
                num_sum -= nums[start]
                start += 1
        if bestAns <= size:
            return bestAns
        else:
            return 0
                   
nums = [2,3,1,2,4,3]                    
s =7
my_sol = Solution()
print my_sol.minSubArrayLen(s,nums)
