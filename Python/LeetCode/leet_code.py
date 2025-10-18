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

# 3.两数相加 -- 双指针解法
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
def addTwoNumbers(l1, l2):
        dummy = ListNode() # 虚拟头节点
        head = dummy
        
        carry = 0
        while l1 or l2 or carry:
            if l1:
                carry += l1.val
                l1 = l1.next
            if l2:
                carry += l2.val
                l2 = l2.next
            node = ListNode(carry % 10)
            carry //= 10
            dummy.next = node
            dummy = dummy.next
        
        return head.next

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
        # 使用set()集合可以确保没有重复的解
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
                    # 下面两行的作用是推进搜索
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

# 13.接雨水 I -- 双指针
def trap(height: list[int]) -> int:
    l, r = 0, len(height)-1
    l_max = r_max = 0
    water = 0
    while l <= r: # 当l == r时,中间最后一个柱子也要处理,需要逻辑覆盖
        if l_max < r_max:
            if height[l] > l_max:
                l_max = height[l]
            else:
                water += l_max - height[l]
            l += 1
        else:
            if height[r] > r_max:
                r_max = height[r]
            else:
                water += r_max - height[r]
            r -= 1
            
    return water
    
# 14.接雨水 II -- BFS + 最小堆
import heapq
def trapRainWater(heightMap):
    if not heightMap or not heightMap[0] or len(heightMap) < 3 or len(heightMap[0]) < 3:
        
            return 0

    m, n = len(heightMap), len(heightMap[0])
    visited = [[False] * n for _ in range(m)]
    heap = []

    # 1. 将最外圈加入最小堆
    for i in range(m):
        for j in [0, n - 1]:
            heapq.heappush(heap, (heightMap[i][j], i, j))
            visited[i][j] = True
    for j in range(1, n - 1):
        for i in [0, m - 1]:
            heapq.heappush(heap, (heightMap[i][j], i, j))
            visited[i][j] = True

    # 2. 方向数组
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    total_water = 0

    # 3. BFS + 最小堆
    while heap:
        h, i, j = heapq.heappop(heap)
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and not visited[ni][nj]:
                visited[ni][nj] = True
                # 如果邻居更低，可以存水
                if heightMap[ni][nj] < h:
                    total_water += h - heightMap[ni][nj]
                    # 以当前水位 h 入堆（水填平了）
                    heapq.heappush(heap, (h, ni, nj))
                else:
                    # 邻居更高，不能存水，以其自身高度入堆
                    heapq.heappush(heap, (heightMap[ni][nj], ni, nj))
    
    return total_water

# 15. 盛水最多的容器
def maxArea(height):
    len_height = len(height)
    left_point = 0
    right_point = len_height - 1
    max_area = 0
    while left_point < right_point:
        area = (right_point - left_point) * min(height[left_point], height[right_point])
        if area > max_area:
            max_area = area
        elif height[left_point] < height[right_point]:
            left_point += 1
        else:
            right_point -= 1
            
    return max_area

# 16. 太平洋大西洋水流问题 -- 深度优先
#TODO 可以着手实现深度和广度优先的搜索方式
def pacificAtlantic(heights):
        if not heights or not heights[0]:
            return []
        
        m, n = len(heights), len(heights[0])
        pacific = [[False] * n for _ in range(m)]
        atlantic = [[False] * n for _ in range(m)]
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        def dfs(i, j, visited, prev_height):
            # 终止条件：越界、已访问、或当前高度小于前一个（无法逆流）
            if i < 0 or i >= m or j < 0 or j >= n or visited[i][j] or heights[i][j] < prev_height:
                return
            visited[i][j] = True
            for di, dj in directions:
                dfs(i + di, j + dj, visited, heights[i][j])
        
        # 从太平洋边界出发（左 + 上）
        for i in range(m):
            dfs(i, 0, pacific, heights[i][0])
        for j in range(n):
            dfs(0, j, pacific, heights[0][j])
        
        # 从大西洋边界出发（右 + 下）
        for i in range(m):
            dfs(i, n - 1, atlantic, heights[i][n - 1])
        for j in range(n):
            dfs(m - 1, j, atlantic, heights[m - 1][j])
        
        # 收集交集
        return [[i, j] for i in range(m) for j in range(n) if pacific[i][j] and atlantic[i][j]]

# 17. 二叉树的遍历(前/中/后)
def traversal_unified(root, order='inorder'):
    if not root:
        return []
    
    stack = [(root, 0)]  # (节点, 状态: 0=未访问, 1=已访问)
    result = []
    
    while stack:
        node, state = stack.pop()
        
        if state == 1:
            result.append(node.val)
        else:
            # 根据遍历类型，决定入栈顺序（注意：栈是后进先出，所以要反着压）
            if order == 'postorder':
                # 后序：左 → 右 → 根 → 入栈顺序：根(1), 右(0), 左(0)
                stack.append((node, 1))
                if node.right:
                    stack.append((node.right, 0))
                if node.left:
                    stack.append((node.left, 0))
            elif order == 'inorder':
                # 中序：左 → 根 → 右 → 入栈顺序：右(0), 根(1), 左(0)
                if node.right:
                    stack.append((node.right, 0))
                stack.append((node, 1))
                if node.left:
                    stack.append((node.left, 0))
            elif order == 'preorder':
                # 前序：根 → 左 → 右 → 入栈顺序：右(0), 左(0), 根(1)
                if node.right:
                    stack.append((node.right, 0))
                if node.left:
                    stack.append((node.left, 0))
                stack.append((node, 1))
            else:
                raise ValueError("order must be 'preorder', 'inorder', or 'postorder'")
    
    return result

# 18.各位相加
def addDigits(num):
    if len(str(num)) > 1:
        ans = 0
        num_list = []
        for i in range(0, len(str(num))):
            num_list.append(int(str(num)[i]))
        ans = sum(num_list)
        return addDigits(ans)
        
    else:
        return num

# 18 - 1. 各位相加 -- 更快的方法
def addDigits(num):
        while num >= 10:
            num = sum(int(i) for i in str(num))

        return num
        
# 19. 水位上升的泳池中游泳
def swimInWater(self, grid):
        import heapq
        n = len(grid)
        # 优先队列：(当前路径的最大高度, x, y)
        heap = [(grid[0][0], 0, 0)]
        visited = [[False] * n for _ in range(n)]
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while heap:
            max_h, x, y = heapq.heappop(heap)
            
            if x == n - 1 and y == n - 1:
                return max_h
                
            if visited[x][y]:
                continue
            visited[x][y] = True
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n and not visited[nx][ny]:
                    # 新路径的最大高度 = max(当前最大高度, 下一格高度)
                    new_max = max(max_h, grid[nx][ny])
                    heapq.heappush(heap, (new_max, nx, ny))
        
        return -1

# 20. 重新排列数组
def shuffle(nums, n):
    new_nums_list = []
    for i in range(0, len(nums) // 2):
        new_nums_list.append(nums[i])
        new_nums_list.append(nums[n + i])
    return new_nums_list

# 21. 分割字符的最大得分
def maxScore(s):
        if s:
            left_list = []
            right_list = []
            score_list = []
            for i in range(1, len(s)):
                left_list.append(s[:i])
                right_list.append(s[i:])
            for m in range(0, len(left_list)):
                sum_left_list = left_list[m].count('0')
                sum_right_list = right_list[m].count('1')
                score_list.append(sum_left_list + sum_right_list)
            
            return max(score_list)
        else:
            return 0
    
# 22. 统计范围内的元音字符串数
def vowelStrings(words, left, right):
    count = 0
    vowel_str_list = ["a", "e", "i", "o", "u"]
    for i in range(left, right + 1):
        if (words[i][0] in vowel_str_list) and (words[i][-1] in vowel_str_list):
            count += 1
    return count

# 23. 山脉数组的峰顶索引
def peakIndexInMountainArray(arr):
    left_point = 0
    right_point = len(arr) - 1

    while left_point <= right_point:
        # 从left_point开始,向右走一半距离,来源于(right - left) >> 1的经典bug
        # 可以保证不会出现溢出的情况
        mid = left_point + (right_point - left_point) // 2
        if arr[mid] < arr[mid + 1]:
            left_point = mid + 1
        else:
            right_point = mid - 1
        
    return left_point

# 24. 避免洪水泛滥
def avoidFlood(rains):
    from bisect import bisect_right, insort
    n = len(rains)
    ans = [-1] * n  # 默认下雨天为 -1
    
    # 记录每个湖泊上一次下雨的日期
    last_rain = {}
    
    # 用有序列表保存所有晴天的索引（日期）
    dry_days = []  # sorted list of indices where rains[i] == 0
    
    for i in range(n):
        lake = rains[i]
        
        if lake == 0:
            # 晴天：先记录，后面再决定抽哪个湖
            ans[i] = 1
            insort(dry_days, i)  # 保持有序
        else:
            # 下雨天
            if lake in last_rain:
                # 湖已经满了！必须在上次下雨后找一个晴天抽干
                last_day = last_rain[lake]
                
                # 在 dry_days 中找第一个 > last_day 的晴天
                pos = bisect_right(dry_days, last_day)
                
                if pos == len(dry_days):
                    # 没有可用晴天 → 洪水！
                    return []
                
                # 使用这个晴天抽干 lake
                dry_day = dry_days[pos]
                ans[dry_day] = lake  # 记录那天抽的是 lake
                dry_days.pop(pos)    # 这个晴天已被使用
            
            # 更新该湖的最后下雨时间
            last_rain[lake] = i
    
    return ans

# 25. 咒语和药水的成功对数
import bisect
def successfulPairs(spells, potions, success):
    ans_list = []
    result = [[a * b for b in potions] for a in spells]
    for i in range(0, len(result)):
        ans = 0
        temp_result = sorted(result[i])
        for j in range(0, len(temp_result)):
            if temp_result[j] >= success:
                ans += 1
        ans_list.append(ans)
    return ans_list

# 25. 咒语和药水的成功对数 -- 更快的方法
def successfulPairs(spells, potions, success):
    potions.sort()
    return [len(potions) - bisect.bisect_right(potions, (success - 1) // i) for i in spells]

# 26. 酿造药水需要的最少总时间
from itertools import accumulate, pairwise
def minTime(skill, mana):
        n = len(skill)
        s = list(accumulate(skill, initial=0))

        suf_record = [n - 1]
        for i in range(n - 2, -1, -1):
            if skill[i] > skill[suf_record[-1]]:
                suf_record.append(i)

        pre_record = [0]
        for i in range(1, n):
            if skill[i] > skill[pre_record[-1]]:
                pre_record.append(i)

        start = 0
        for pre, cur in pairwise(mana):
            record = pre_record if pre < cur else suf_record
            start += max(pre * s[i + 1] - cur * s[i] for i in record)
        return start + mana[-1] * s[-1]
    
# 27. 从魔法师身上吸取的最大能量
def maximumEnergy(energy, k):
    n = len(energy)
    for i in range(n - 1 - k, -1, -1):
        energy[i] += energy[i + k]
    return max(energy)

# 28. 施咒的最大总伤害
from functools import cache
from collections import Counter
def maximumTotalDamage(power):
        cnt = Counter(power)
        a = sorted(cnt)

        @cache
        def dfs(i):
            if i < 0:
                return 0
            x = a[i]
            j = i
            while j and a[j - 1] >= x - 2:
                j -= 1
            return max(dfs(i - 1), dfs(j - 1) + x * cnt[x])

        return dfs(len(a) - 1)

# 29. 定长子串中元音的最大数目 -- 固定长度的滑动窗口练习
def maxVowels(s, k):
    vowels = set("aeiou")
    
    # 初始化第一个窗口[0, k - 1]
    current_vowel = 0
    for i in range(k):
        if s[i] in vowels:
            current_vowel += 1
    max_vowels = current_vowel # 初始化最大值
    
    # 滑动窗口,右端点从k到len(s) - 1
    for i in range(k, len(s)):
        # 移除左边字符s[i - k]
        if s[i - k] in vowels:
            current_vowel -= 1
        
        # 加入右边字符s[i]
        if s[i] in vowels:
            current_vowel += 1

        max_vowels = max(max_vowels, current_vowel)
        
        if max_vowels == k:
            break
        
    return max_vowels
    
# 30. 子数组最大平均数 -- 固定长度滑动窗口
def findMaxAverage(nums, k):
    # 初始化第一个窗口
    current_sums = sum(nums[:k])
    max_sum = current_sums
    
    # 滑动窗口 -- 减左 + 加右
    for i in range(k, len(nums)):
        current_sums = current_sums - nums[i - k] + nums[i]
        max_sum = max(current_sums, max_sum)
        
    return max_sum / k

# 31. 大小为K且平均值大于等于阈值的子数组数目
def numOfSubarrays(arr, k, threshold):
    # 初始化第一个窗口
    substr_num = 0
    current_sums = sum(arr[:k])
    # current_sums >= threshold * k 与 (current_sums / k) >= threshold的比较
    # current_sums >= threshold * k避免了精度的问题
    if current_sums >= threshold * k:
        substr_num += 1
         
    # 滑动窗口 -- 减左 + 加右
    for i in range(k, len(arr)):
        current_sums = current_sums -arr[i - k] + arr[i]
        if current_sums >= threshold * k:
            substr_num += 1
    
    return substr_num

# 32. 魔法序列的数组乘积之和 -- 有点过于抽象
MOD = 1_000_000_007
MX = 31

fac = [0] * MX  # fac[i] = i!
fac[0] = 1
for i in range(1, MX):
    fac[i] = fac[i - 1] * i % MOD

inv_f = [0] * MX  # inv_f[i] = i!^-1
inv_f[-1] = pow(fac[-1], -1, MOD)
for i in range(MX - 1, 0, -1):
    inv_f[i - 1] = inv_f[i] * i % MOD

def magicalSum(m, k, nums):
    n = len(nums)
    pow_v = [[1] * (m + 1) for _ in range(n)]
    for i, v in enumerate(nums):
        for j in range(1, m + 1):
            pow_v[i][j] = pow_v[i][j - 1] * v % MOD

    @cache
    def dfs(i, left_m, x, left_k):
        c1 = x.bit_count()
        if c1 + left_m < left_k:  # 可行性剪枝
            return 0
        if i == n or left_m == 0 or left_k == 0:  # 无法继续选数字
            return 1 if left_m == 0 and c1 == left_k else 0
        res = 0
        for j in range(left_m + 1):  # 枚举 I 中有 j 个下标 i
            # 这 j 个下标 i 对 S 的贡献是 j * pow(2, i)
            # 由于 x = S >> i,转化成对 x 的贡献是 j
            bit = (x + j) & 1  # 取最低位,提前从 left_k 中减去,其余进位到 x 中
            r = dfs(i + 1, left_m - j, (x + j) >> 1, left_k - bit)
            res += r * pow_v[i][j] * inv_f[j]
        return res % MOD

    return dfs(0, m, 0, k) * fac[m] % MOD

# 33. 得到K个黑块的最少涂色次数 -- 滑动窗口取最小值
def minimumRecolors(blocks: str, k: int) -> int:
    # 初始化第一个窗口[0, k - 1]
    w_count = 0
    for i in range(k):
        if 'W' == blocks[i]:
            w_count += 1
    min_ops = w_count
    
    # 滑动窗口,右端点从k到len(blocks) - 1
    for i in range(k, len(blocks)):
        # 移除左边的字符
        if 'W' == blocks[i - k]:
            w_count -= 1
        
        if 'W' == blocks[i]:
            w_count += 1
            
        min_ops = min(min_ops, w_count)
        
    return min_ops

# 34. 几乎唯一子数组的最大和 -- 滑动窗口练习
from collections import defaultdict
def maxSum(nums: list[int], m: int, k: int) -> int:
    ans = s = 0
    cnt = defaultdict(int)
    for i, x in enumerate(nums):
        # 1. 进入窗口
        s += x
        cnt[x] += 1

        left = i - k + 1
        if left < 0:  # 窗口大小不足 k
            continue

        # 2. 更新答案
        if len(cnt) >= m:
            ans = max(ans, s)

        # 3. 离开窗口
        out = nums[left]
        s -= out
        cnt[out] -= 1
        if cnt[out] == 0:
            del cnt[out]

    return ans

# 35. 移除字母异位词后的结果数组
def removeAnagrams(words: list[str]) -> list[str]:
    res = [words[0]]
    for i in range(1, len(words)):
        if sorted(words[i]) != sorted(words[i - 1]):
            res.append(words[i])
    
    return res

# 35.1. 移除字母异位词后的结果数组 -- 双指针法
def removeAnagramsByPoints(words: list[str]) -> list[str]:
    # 慢指针
    slow_point = 0
    
    for i in range(1, len(words)):
        if sorted(words[i]) != sorted(words[i + 1]):
            slow_point += 1
            words[slow_point] = words[i]
            
    return words[:slow_point + 1]

# 36. 检测相邻递增子数组 I
def hasIncreasingSubarrays(nums: list[int], k: int) -> bool:
    len_nums = len(nums)
    for i in range(len_nums):
        j = i + k
        flag = True
        cnt = 1
        m = i
        while cnt < k:
            if (m + 1 >= len_nums or 
                j + 1 >= len_nums or 
                nums[m] >= nums[m + 1] or 
                nums[j] >= nums[j + 1]):
                flag = False
                break
            m += 1
            j += 1
            cnt += 1
        if flag and cnt == k:
            return True
    return False

# 37. 半径为k的子数组平均值
def getAverages(nums: list[int], k: int) -> list[int]:
    len_nums = len(nums)
    win_size = 2 * k + 1
    ans_list = [-1] * len_nums
    
    if win_size > len_nums:
        return ans_list
    
    #初始化第一个窗口
    win_sum_nums = sum(nums[:win_size])
    average_num = win_sum_nums // win_size
    ans_list[k] = average_num
    
    for i in range(win_size, len_nums):
        win_sum_nums = win_sum_nums - nums[i - win_size] + nums[i]
        average_num = win_sum_nums // win_size
        ans_list[i - k] = average_num
        
    return ans_list

# 38. 长度为K子数组中的最大和
def maximumSubarraySum(nums: list[int], k: int) -> int:
    len_nums = len(nums)
    
    if k > len_nums:
        return 0
    
    # 初始化第一个窗口的和并对max_sum赋初值
    max_sum = 0
    current_sum = sum(nums[:k])
    
    # 检验第一个窗口的合理性
    if len(nums[:k]) == len(set(nums[:k])):
        max_sum = current_sum
    
    for i in range(k, len_nums):
        # 始终更新窗口和
        current_sum = current_sum - nums[i - k] + nums[i]
        
        if len(set(nums[i - k + 1 : i + 1])) == k:
            max_sum = max(max_sum, current_sum)
    
    
    return max_sum
            
# 38.1 长度为K子数组中的最大和 -- 更快的方法 -- 使用频次哈希表
# defaultdict -- 创建带有默认值的字典,默认值为可调用对象
def maximumSubarraySumFaster(nums: list[int], k: int) -> int:
    from collections import defaultdict
    
    n = len(nums)
    if k > n:
        return 0
    
    freq = defaultdict(int)
    current_sum = 0
    
    # 初始化第一个窗口
    for i in range(k):
        freq[nums[i]] += 1
        current_sum += nums[i]
    
    max_sum = current_sum if len(freq) == k else 0

    # 滑动窗口
    for i in range(k, n):
        left = nums[i - k]
        right = nums[i]
        
        # 移除 left
        freq[left] -= 1
        if freq[left] == 0:
            del freq[left]
        current_sum -= left
        
        # 添加 right
        freq[right] += 1
        current_sum += right
        
        # 检查是否所有元素唯一
        if len(freq) == k:
            max_sum = max(max_sum, current_sum)
    
    return max_sum

# 39. 执行操作后的最大MEX
def findSmallestInteger(nums: list[int], value: int) -> int:
    cnt = [0] * value
    for i in nums:
        r = i % value
        cnt[r] += 1
    
    t = 0
    while True:
        r = t % value
        if 0 == cnt[r]:
            return t
        cnt[r] -= 1
        t += 1
        
# 40. 可获得的最大点数 -- 滑动窗口练习
def maxScore(cardPoints: list[int], k: int) -> int:
    len_card = len(cardPoints)
    all_sum_card = sum(cardPoints)
    
    if len_card == k:
        return all_sum_card
    
    win_size = len_card - k
    current_num = sum(cardPoints[:win_size])
    min_num = current_num
    
    for i in range(win_size, len_card):
        current_num += cardPoints[i] - cardPoints[i - win_size]
        min_num = min(min_num, current_num)
        
    return all_sum_card - min_num

# 41. 执行操作后不同元素的最大数量
def maxDistinctElements(nums: list[int], k: int) -> int:
    nums.sort()
    last = float('-inf')
    count = 0
    
    for x in nums:
        low = x - k
        high = x + k
        candidate = max(low, last + 1)
        if candidate <= high:
            count += 1
            last = candidate
    return count

# 42. 使库存平衡的最少丢弃数 -- 固定长度滑动窗口进阶练习
def minArrivalsToDiscard(arrivals: list[int], w: int, m: int) -> int:
    ans = 0
    cnt = [0] * (max(arrivals) + 1)
    
    for i , x in enumerate(arrivals):
        if m == cnt[x]:
            arrivals[i] = 0
            ans += 1
        else:
            cnt[x] += 1
        
        left = i - w + 1
        if left >= 0:
            cnt[arrivals[left]] -= 1
    
    return ans
