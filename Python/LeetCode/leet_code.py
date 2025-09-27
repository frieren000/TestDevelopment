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

points = [[0,0],[0,1],[1,0],[0,2],[2,0]]
print(largestTriangleArea(points))