def maxSubArray(nums):
    a = c = nums[0]
    end = 0
    begin = 0
    for i in range(1, len(nums)):
        if a+nums[i] >= nums[i]:
            a = a + nums[i]
        else:
            a = nums[i]
            begin = i
        if a >= c:
            c = a
            end = i
        else:
            c = c

    result = nums[begin:end+1]
    return result

if __name__ == '__main__':
    print(maxSubArray([-2,1,-3,4,-1,2,1,-5,4]))
