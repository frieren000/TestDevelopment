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

# 3.两数相加
def addTwoNumbers(list_1, list_2):
    print(list_1, list_2)
    
l1 = [9,9,9,9,9,9,9]
l2 = [9,9,9,9]
addTwoNumbers(l1, l2)