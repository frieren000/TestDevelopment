# 1.两数之和
def twoSum(nums_list, target):
    for i in range(0, len(nums_list)):
        first_num = nums_list[i]
        second_num = target - first_num
        for j in range(i + 1, len(nums_list)):
            if nums_list[j] == second_num:
                return [i, j]

# 1.1 两数之和 -- 哈希表
def twoSumByPointer(nums_list, target):
    num_to_index = {}
    for i, num in enumerate(nums_list):
        complement = target - num
        if complement in num_to_index:
            return [num_to_index[complement], i]
        num_to_index[num] = i
    return []

# 1.3 两数之和 II - 输入有序数组 -- 对撞指针
def twoSumByOrder(nums, target):
    left_point = 0
    right_point = len(nums) - 1
    while left_point < right_point:
        if nums[left_point] + nums[right_point] == target:
            return [left_point + 1, right_point + 1]
        elif nums[left_point] + nums[right_point] < target:
            left_point += 1
        else:
            right_point -= 1
    return []

# 2.回文数
def isPalindrome(int):
    str_int = str(int)
    if str_int == str_int[::-1]:
        print(str_int, str_int[::-1])
        return True
    else:
        return False

# 3.两数相加 -- 待尝试
# def addTwoNumbers(list_1, list_2):
#     return 0
    
# l1 = [9,9,9,9,9,9,9]
# l2 = [9,9,9,9]
# addTwoNumbers(l1, l2)

# 4.删除有序数组中的重复项
def removeDuplicates(nums):
    new_nums_list = []
    for i in range(0, len(nums)):
        if nums[i] not in new_nums_list:
            new_nums_list.append(nums[i])
    nums[:] = new_nums_list
    # nums = new_nums_list 并没有修改原列表,只是重新绑定了变量
    # nums[:] = new_nums_list是一个切片赋值操作,清空原列表的所有元素并重新赋值
    return len(nums)

# 5.三数之和 -- 三个for暴力循环 -- 大数据容易超时
def threeSum(nums_list, target):
    nums = set()
    for i in range(0, len(nums_list)):
        for j in range(i + 1, len(nums_list)):
            for k in range(j + 1, len(nums_list)):
                if nums_list[i] + nums_list[j] + nums_list[k] == target:
                    nums.add(tuple(sorted([nums_list[i], nums_list[j], nums_list[k]])))
    
    result = [list(t) for t in nums]
    return result

# 5.1 三数之和 -- 指针法
def threeSumByOrder(nums_list, target):
        result = []
        temple_result = set()
        nums_list.sort()
        for i in range(0, len(nums_list)):
            left_point = i + 1
            right_point = len(nums_list) - 1
            while left_point < right_point:
                three_sum = nums_list[i] + nums_list[left_point] + nums_list[right_point]
                # 关键点一:求和过程应在while循环内
                if three_sum == target:
                    temple_result.add((nums_list[i], nums_list[left_point], nums_list[right_point]))
                    left_point += 1
                    right_point -= 1
                    # 关键点二:在确认一组解后应移动指针位置
                elif three_sum < target:
                    left_point += 1
                else:
                    right_point -= 1
        result = [list(t) for t in temple_result]
        return result

# 5.2 三数之和 -- 更快的方法
def threeSumByOrderFast(nums_list, target):
    nums_list.sort()
    result = []
    
    for i in range(len(nums_list) - 2):
        if i > 0 and nums_list[i] == nums_list[i - 1]:
            continue  # 跳过重复 i
        
        left, right = i + 1, len(nums_list) - 1
        while left < right:
            total = nums_list[i] + nums_list[left] + nums_list[right]
            if total == target:
                result.append([nums_list[i], nums_list[left], nums_list[right]])
                # 跳过重复值
                while left < right and nums_list[left] == nums_list[left + 1]:
                    left += 1
                while left < right and nums_list[right] == nums_list[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < target:
                left += 1
            else:
                right -= 1
                
    return result

# 6.最大三角形面积
def largestTriangleArea(points):
    len_points = len(points)
    max_area = 0.0
    for i in range(0, len_points):
        for j in range(i, len_points):
            for k in range(j, len_points):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]

                product = (x1 * y2 + x2 * y3 + x3 * y1) - (y1 * x2 + y2 * x3 + y3 * x1)
                area = 0.5 * abs(product)

                if area > max_area:
                    max_area = area
    
    return float(max_area)

# 7.最大三角形周长 -- 暴力解法 -- 大数据容易超时
def largestPerimeter(nums):
    max_perimeter = 0
    nums.sort(reverse=True)
    # 保证越靠前的三个数周长越大
    for i in range(0, len(nums)):
        for j in range(i + 1, len(nums)):
            for k in range(j + 1, len(nums)):
                perimeter = nums[i] + nums[j] + nums[k]
                if (nums[i] < nums[j] + nums[k]) and  perimeter > max_perimeter:
                    max_perimeter = perimeter
    
    return max_perimeter

# 7.1 最大三角形周长 -- 更快的解法
def largestPerimeterByFast(nums):
    max_perimeter = 0
    nums.sort(reverse=True)
    for i in range(0, len(nums) - 2):
        perimeter = nums[i] + nums[i + 1] + nums[i + 2]
        if nums[i + 2] + nums[i + 1] > nums[i]:
            max_perimeter = perimeter
            break
        # 这里可以使用break在找到第一组符合条件的三个数字时直接退出循环
    return max_perimeter

# 8. 最长公共前缀
def longestCommonPrefix(strs):
    # 对字符串列表进行一个升序排序 -- 字典序列
    # 排序的作用: 按字典序排序后,差异最大的两个字符会出现在两端
    strs.sort()
    first_str = strs[0]
    last_str = strs[-1]
    for i in range(len(first_str)):
        if i < len(last_str) and first_str[i] == last_str[i]:
            continue
        else:
            return first_str[:i]
    # 当first_str == last_str时,first_str为最大公共前缀,必须进行显示返回
    return first_str

# 9. 多边形三角剖分的最低得分 -- 区间DP(动态规划)
# TODO 还是得练DP
def minScoreTriangulation(values):
    n = len(values)
    # dp[i][j] 表示从 i 到 j 构成的多边形的最小得分(i < j)
    dp = [[0] * n for _ in range(n)]

    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i + 1, j):
                dp[i][j] = min(
                    dp[i][j],
                    dp[i][k] + dp[k][j] + values[i] * values[k] * values[j]
                )
    return dp[0][n - 1]
        
# 10. 数组的三角和
def triangularSum(nums):
    len_nums = len(nums)
    if len_nums == 1:
        return nums[0]
    else:
        new_nums_list = []
        for i in range(0, len_nums - 1):
            new_nums = (nums[i] + nums[i + 1]) % 10
            new_nums_list.append(new_nums)
    
    return triangularSum(new_nums_list)

# 10.1 一个更快的解法
def triangularSumFast(nums):
    total = len(nums)-1
    now = 1
    answer = 0
    for i,num in enumerate(nums):
        answer += now * num
        now = now * (total - i) // ( i + 1)
        
    return answer % 10

# 11. 换水问题
def numWaterBottles(numBottles, numExchange):
    ans = numBottles + (numBottles - 1) // (numExchange - 1)
    
    return ans

# 12. 换水问题 II
def maxBottlesDrunk( numBottles, numExchange):
    ans = numBottles
    empty = numBottles
    while empty >= numExchange:
        ans += 1
        empty -= numExchange - 1
        numExchange += 1
    return ans