# 1.两数之和
def twoSum(nums_list, target):
    for i in range(0, len(nums_list)):
        first_num = nums_list[i]
        second_num = target - first_num
        for j in range(i + 1, len(nums_list)):
            if nums_list[j] == second_num:
                return [i, j]
        
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