import heapq
import bisect
from typing import List, Optional, Tuple, Set, Dict
from collections import defaultdict, Counter, deque
from itertools import accumulate, count, pairwise


# 1.两数之和
def twoSum(nums_list: List[int], target: int) -> List[int]:
    for i in range(0, len(nums_list)):
        first_num = nums_list[i]
        second_num = target - first_num
        for j in range(i + 1, len(nums_list)):
            if nums_list[j] == second_num:
                return [i, j]
    return []

# 1.1 两数之和 -- 哈希表
def twoSumByPointer(nums_list: List[int], target: int) -> List[int]:
    num_to_index: Dict[int, int] = {}
    for i, num in enumerate(nums_list):
        complement = target - num
        if complement in num_to_index:
            return [num_to_index[complement], i]
        num_to_index[num] = i
    return []

# 1.3 两数之和 II - 输入有序数组 -- 对撞指针
def twoSumByOrder(nums: List[int], target: int) -> List[int]:
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
def isPalindrome(x: int) -> bool:
    str_int = str(x)
    if str_int == str_int[::-1]:
        print(str_int, str_int[::-1])
        return True
    else:
        return False

# 3.两数相加 -- 双指针解法
class ListNode:
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next

def addTwoNumbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode()  # 虚拟头节点
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
def removeDuplicates(nums: List[int]) -> int:
    new_nums_list: List[int] = []
    for i in range(0, len(nums)):
        if nums[i] not in new_nums_list:
            new_nums_list.append(nums[i])
    nums[:] = new_nums_list
    return len(nums)

# 5.三数之和 -- 三个for暴力循环 -- 大数据容易超时
def threeSum(nums_list: List[int], target: int) -> List[List[int]]:
    nums: Set[Tuple[int, int, int]] = set()
    for i in range(0, len(nums_list)):
        for j in range(i + 1, len(nums_list)):
            for k in range(j + 1, len(nums_list)):
                if nums_list[i] + nums_list[j] + nums_list[k] == target:
                    nums.add(tuple(sorted([nums_list[i], nums_list[j], nums_list[k]])))
    result = [list(t) for t in nums]
    return result

# 5.1 三数之和 -- 指针法
def threeSumByOrder(nums_list: List[int], target: int) -> List[List[int]]:
    result: List[List[int]] = []
    temple_result: Set[Tuple[int, int, int]] = set()
    nums_list.sort()
    for i in range(0, len(nums_list)):
        left_point = i + 1
        right_point = len(nums_list) - 1
        while left_point < right_point:
            three_sum = nums_list[i] + nums_list[left_point] + nums_list[right_point]
            if three_sum == target:
                temple_result.add((nums_list[i], nums_list[left_point], nums_list[right_point]))
                left_point += 1
                right_point -= 1
            elif three_sum < target:
                left_point += 1
            else:
                right_point -= 1
    result = [list(t) for t in temple_result]
    return result

# 5.2 三数之和 -- 更快的方法
def threeSumByOrderFast(nums_list: List[int], target: int) -> List[List[int]]:
    nums_list.sort()
    result: List[List[int]] = []
    for i in range(len(nums_list) - 2):
        if i > 0 and nums_list[i] == nums_list[i - 1]:
            continue
        left, right = i + 1, len(nums_list) - 1
        while left < right:
            total = nums_list[i] + nums_list[left] + nums_list[right]
            if total == target:
                result.append([nums_list[i], nums_list[left], nums_list[right]])
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
def largestTriangleArea(points: List[List[int]]) -> float:
    len_points = len(points)
    max_area = 0.0
    for i in range(0, len_points):
        for j in range(i + 1, len_points):
            for k in range(j + 1, len_points):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                product = (x1 * y2 + x2 * y3 + x3 * y1) - (y1 * x2 + y2 * x3 + y3 * x1)
                area = 0.5 * abs(product)
                if area > max_area:
                    max_area = area
    return float(max_area)

# 7.最大三角形周长 -- 暴力解法 -- 大数据容易超时
def largestPerimeter(nums: List[int]) -> int:
    max_perimeter = 0
    nums.sort(reverse=True)
    for i in range(0, len(nums)):
        for j in range(i + 1, len(nums)):
            for k in range(j + 1, len(nums)):
                perimeter = nums[i] + nums[j] + nums[k]
                if (nums[i] < nums[j] + nums[k]) and perimeter > max_perimeter:
                    max_perimeter = perimeter
    return max_perimeter

# 7.1 最大三角形周长 -- 更快的解法
def largestPerimeterByFast(nums: List[int]) -> int:
    max_perimeter = 0
    nums.sort(reverse=True)
    for i in range(0, len(nums) - 2):
        perimeter = nums[i] + nums[i + 1] + nums[i + 2]
        if nums[i + 2] + nums[i + 1] > nums[i]:
            max_perimeter = perimeter
            break
    return max_perimeter

# 8. 最长公共前缀
def longestCommonPrefix(strs: List[str]) -> str:
    strs.sort()
    first_str = strs[0]
    last_str = strs[-1]
    for i in range(len(first_str)):
        if i < len(last_str) and first_str[i] == last_str[i]:
            continue
        else:
            return first_str[:i]
    return first_str

# 9. 多边形三角剖分的最低得分 -- 区间DP(动态规划)
def minScoreTriangulation(values: List[int]) -> int:
    n = len(values)
    dp: List[List[int]] = [[0] * n for _ in range(n)]
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
def triangularSum(nums: List[int]) -> int:
    len_nums = len(nums)
    if len_nums == 1:
        return nums[0]
    else:
        new_nums_list: List[int] = []
        for i in range(0, len_nums - 1):
            new_nums = (nums[i] + nums[i + 1]) % 10
            new_nums_list.append(new_nums)
        return triangularSum(new_nums_list)

# 10.1 一个更快的解法
def triangularSumFast(nums: List[int]) -> int:
    total = len(nums) - 1
    now = 1
    answer = 0
    for i, num in enumerate(nums):
        answer += now * num
        now = now * (total - i) // (i + 1)
    return answer % 10

# 11. 换水问题
def numWaterBottles(numBottles: int, numExchange: int) -> int:
    ans = numBottles + (numBottles - 1) // (numExchange - 1)
    return ans

# 12. 换水问题 II
def maxBottlesDrunk(numBottles: int, numExchange: int) -> int:
    ans = numBottles
    empty = numBottles
    while empty >= numExchange:
        ans += 1
        empty -= numExchange - 1
        numExchange += 1
    return ans

# 13.接雨水 I -- 双指针
def trap(height: List[int]) -> int:
    l, r = 0, len(height) - 1
    l_max = r_max = 0
    water = 0
    while l <= r:
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
def trapRainWater(heightMap: List[List[int]]) -> int:
    if not heightMap or not heightMap[0] or len(heightMap) < 3 or len(heightMap[0]) < 3:
        return 0
    m, n = len(heightMap), len(heightMap[0])
    visited: List[List[bool]] = [[False] * n for _ in range(m)]
    heap: List[Tuple[int, int, int]] = []
    for i in range(m):
        for j in [0, n - 1]:
            heapq.heappush(heap, (heightMap[i][j], i, j))
            visited[i][j] = True
    for j in range(1, n - 1):
        for i in [0, m - 1]:
            heapq.heappush(heap, (heightMap[i][j], i, j))
            visited[i][j] = True
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    total_water = 0
    while heap:
        h, i, j = heapq.heappop(heap)
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and not visited[ni][nj]:
                visited[ni][nj] = True
                if heightMap[ni][nj] < h:
                    total_water += h - heightMap[ni][nj]
                    heapq.heappush(heap, (h, ni, nj))
                else:
                    heapq.heappush(heap, (heightMap[ni][nj], ni, nj))
    return total_water

# 15. 盛水最多的容器
def maxArea(height: List[int]) -> int:
    len_height = len(height)
    left_point = 0
    right_point = len_height - 1
    max_area = 0
    while left_point < right_point:
        area = (right_point - left_point) * min(height[left_point], height[right_point])
        if area > max_area:
            max_area = area
        if height[left_point] < height[right_point]:
            left_point += 1
        else:
            right_point -= 1
    return max_area

# 16. 太平洋大西洋水流问题 -- 深度优先
def pacificAtlantic(heights: List[List[int]]) -> List[List[int]]:
    if not heights or not heights[0]:
        return []
    m, n = len(heights), len(heights[0])
    pacific: List[List[bool]] = [[False] * n for _ in range(m)]
    atlantic: List[List[bool]] = [[False] * n for _ in range(m)]
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    def dfs(i: int, j: int, visited: List[List[bool]], prev_height: int) -> None:
        if i < 0 or i >= m or j < 0 or j >= n or visited[i][j] or heights[i][j] < prev_height:
            return
        visited[i][j] = True
        for di, dj in directions:
            dfs(i + di, j + dj, visited, heights[i][j])
    for i in range(m):
        dfs(i, 0, pacific, heights[i][0])
    for j in range(n):
        dfs(0, j, pacific, heights[0][j])
    for i in range(m):
        dfs(i, n - 1, atlantic, heights[i][n - 1])
    for j in range(n):
        dfs(m - 1, j, atlantic, heights[m - 1][j])
    return [[i, j] for i in range(m) for j in range(n) if pacific[i][j] and atlantic[i][j]]

# 17. 二叉树的遍历(前/中/后)
class TreeNode:
    def __init__(self, val: int = 0, left: Optional['TreeNode'] = None, right: Optional['TreeNode'] = None):
        self.val = val
        self.left = left
        self.right = right

def traversal_unified(root: Optional[TreeNode], order: str = 'inorder') -> List[int]:
    if not root:
        return []
    stack: List[Tuple[TreeNode, int]] = [(root, 0)]
    result: List[int] = []
    while stack:
        node, state = stack.pop()
        if state == 1:
            result.append(node.val)
        else:
            if order == 'postorder':
                stack.append((node, 1))
                if node.right:
                    stack.append((node.right, 0))
                if node.left:
                    stack.append((node.left, 0))
            elif order == 'inorder':
                if node.right:
                    stack.append((node.right, 0))
                stack.append((node, 1))
                if node.left:
                    stack.append((node.left, 0))
            elif order == 'preorder':
                if node.right:
                    stack.append((node.right, 0))
                if node.left:
                    stack.append((node.left, 0))
                stack.append((node, 1))
            else:
                raise ValueError("order must be 'preorder', 'inorder', or 'postorder'")
    return result

# 18.各位相加
def addDigits(num: int) -> int:
    if len(str(num)) > 1:
        ans = 0
        num_list = [int(d) for d in str(num)]
        ans = sum(num_list)
        return addDigits(ans)
    else:
        return num

# 18 - 1. 各位相加 -- 更快的方法
def addDigitsFaster(num: int) -> int:
    while num >= 10:
        num = sum(int(i) for i in str(num))
    return num

# 19. 水位上升的泳池中游泳
def swimInWater(grid: List[List[int]]) -> int:
    import heapq
    n = len(grid)
    heap: List[Tuple[int, int, int]] = [(grid[0][0], 0, 0)]
    visited: List[List[bool]] = [[False] * n for _ in range(n)]
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
                new_max = max(max_h, grid[nx][ny])
                heapq.heappush(heap, (new_max, nx, ny))
    return -1

# 20. 重新排列数组
def shuffle(nums: List[int], n: int) -> List[int]:
    new_nums_list: List[int] = []
    for i in range(0, len(nums) // 2):
        new_nums_list.append(nums[i])
        new_nums_list.append(nums[n + i])
    return new_nums_list

# 21. 分割字符的最大得分
def maxScore(s: str) -> int:
    if s:
        left_list: List[str] = []
        right_list: List[str] = []
        score_list: List[int] = []
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
def vowelStrings(words: List[str], left: int, right: int) -> int:
    count = 0
    vowel_str_list = ["a", "e", "i", "o", "u"]
    for i in range(left, right + 1):
        if (words[i][0] in vowel_str_list) and (words[i][-1] in vowel_str_list):
            count += 1
    return count

# 23. 山脉数组的峰顶索引
def peakIndexInMountainArray(arr: List[int]) -> int:
    left_point = 0
    right_point = len(arr) - 1
    while left_point <= right_point:
        mid = left_point + (right_point - left_point) // 2
        if arr[mid] < arr[mid + 1]:
            left_point = mid + 1
        else:
            right_point = mid - 1
    return left_point

# 24. 避免洪水泛滥
def avoidFlood(rains: List[int]) -> List[int]:
    from bisect import bisect_right, insort
    n = len(rains)
    ans = [-1] * n
    last_rain: Dict[int, int] = {}
    dry_days: List[int] = []
    for i in range(n):
        lake = rains[i]
        if lake == 0:
            ans[i] = 1
            insort(dry_days, i)
        else:
            if lake in last_rain:
                last_day = last_rain[lake]
                pos = bisect_right(dry_days, last_day)
                if pos == len(dry_days):
                    return []
                dry_day = dry_days[pos]
                ans[dry_day] = lake
                dry_days.pop(pos)
            last_rain[lake] = i
    return ans

# 25. 咒语和药水的成功对数
def successfulPairs(spells: List[int], potions: List[int], success: int) -> List[int]:
    potions.sort()
    return [len(potions) - bisect.bisect_right(potions, (success - 1) // i) for i in spells]

# 26. 酿造药水需要的最少总时间
def minTime(skill: List[int], mana: List[int]) -> int:
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
def maximumEnergy(energy: List[int], k: int) -> int:
    n = len(energy)
    for i in range(n - 1 - k, -1, -1):
        energy[i] += energy[i + k]
    return max(energy)

# 28. 施咒的最大总伤害
def maximumTotalDamage(power: List[int]) -> int:
    cnt = Counter(power)
    a = sorted(cnt)
    from functools import cache
    @cache
    def dfs(i: int) -> int:
        if i < 0:
            return 0
        x = a[i]
        j = i
        while j and a[j - 1] >= x - 2:
            j -= 1
        return max(dfs(i - 1), dfs(j - 1) + x * cnt[x])
    return dfs(len(a) - 1)

# 29. 定长子串中元音的最大数目 -- 固定长度的滑动窗口练习
def maxVowels(s: str, k: int) -> int:
    vowels = set("aeiou")
    current_vowel = 0
    for i in range(k):
        if s[i] in vowels:
            current_vowel += 1
    max_vowels = current_vowel
    for i in range(k, len(s)):
        if s[i - k] in vowels:
            current_vowel -= 1
        if s[i] in vowels:
            current_vowel += 1
        max_vowels = max(max_vowels, current_vowel)
        if max_vowels == k:
            break
    return max_vowels

# 30. 子数组最大平均数 -- 固定长度滑动窗口
def findMaxAverage(nums: List[int], k: int) -> float:
    current_sums = sum(nums[:k])
    max_sum = current_sums
    for i in range(k, len(nums)):
        current_sums = current_sums - nums[i - k] + nums[i]
        max_sum = max(current_sums, max_sum)
    return max_sum / k

# 31. 大小为K且平均值大于等于阈值的子数组数目
def numOfSubarrays(arr: List[int], k: int, threshold: int) -> int:
    substr_num = 0
    current_sums = sum(arr[:k])
    if current_sums >= threshold * k:
        substr_num += 1
    for i in range(k, len(arr)):
        current_sums = current_sums - arr[i - k] + arr[i]
        if current_sums >= threshold * k:
            substr_num += 1
    return substr_num

# 32. 魔法序列的数组乘积之和 -- 有点过于抽象
MOD = 1_000_000_007
MX = 31

fac = [0] * MX
fac[0] = 1
for i in range(1, MX):
    fac[i] = fac[i - 1] * i % MOD

inv_f = [0] * MX
inv_f[-1] = pow(fac[-1], -1, MOD)
for i in range(MX - 1, 0, -1):
    inv_f[i - 1] = inv_f[i] * i % MOD

def magicalSum(m: int, k: int, nums: List[int]) -> int:
    n = len(nums)
    pow_v = [[1] * (m + 1) for _ in range(n)]
    for i, v in enumerate(nums):
        for j in range(1, m + 1):
            pow_v[i][j] = pow_v[i][j - 1] * v % MOD

    from functools import cache
    @cache
    def dfs(i: int, left_m: int, x: int, left_k: int) -> int:
        c1 = x.bit_count()
        if c1 + left_m < left_k:
            return 0
        if i == n or left_m == 0 or left_k == 0:
            return 1 if left_m == 0 and c1 == left_k else 0
        res = 0
        for j in range(left_m + 1):
            bit = (x + j) & 1
            r = dfs(i + 1, left_m - j, (x + j) >> 1, left_k - bit)
            res += r * pow_v[i][j] * inv_f[j]
        return res % MOD

    return dfs(0, m, 0, k) * fac[m] % MOD

# 33. 得到K个黑块的最少涂色次数 -- 滑动窗口取最小值
def minimumRecolors(blocks: str, k: int) -> int:
    w_count = 0
    for i in range(k):
        if 'W' == blocks[i]:
            w_count += 1
    min_ops = w_count
    for i in range(k, len(blocks)):
        if 'W' == blocks[i - k]:
            w_count -= 1
        if 'W' == blocks[i]:
            w_count += 1
        min_ops = min(min_ops, w_count)
    return min_ops

# 34. 几乎唯一子数组的最大和 -- 滑动窗口练习
def maxSum(nums: List[int], m: int, k: int) -> int:
    ans = s = 0
    cnt: Dict[int, int] = defaultdict(int)
    for i, x in enumerate(nums):
        s += x
        cnt[x] += 1
        left = i - k + 1
        if left < 0:
            continue
        if len(cnt) >= m:
            ans = max(ans, s)
        out = nums[left]
        s -= out
        cnt[out] -= 1
        if cnt[out] == 0:
            del cnt[out]
    return ans

# 35. 移除字母异位词后的结果数组
def removeAnagrams(words: List[str]) -> List[str]:
    res = [words[0]]
    for i in range(1, len(words)):
        if sorted(words[i]) != sorted(words[i - 1]):
            res.append(words[i])
    return res

# 35.1. 移除字母异位词后的结果数组 -- 双指针法
def removeAnagramsByPoints(words: List[str]) -> List[str]:
    slow_point = 0
    for i in range(1, len(words)):
        if sorted(words[i]) != sorted(words[i - 1]):  # 注意：原代码有 bug，应为 words[i] vs words[i-1]
            slow_point += 1
            words[slow_point] = words[i]
    return words[:slow_point + 1]

# 36. 检测相邻递增子数组 I
def hasIncreasingSubarrays(nums: List[int], k: int) -> bool:
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
def getAverages(nums: List[int], k: int) -> List[int]:
    len_nums = len(nums)
    win_size = 2 * k + 1
    ans_list = [-1] * len_nums
    if win_size > len_nums:
        return ans_list
    win_sum_nums = sum(nums[:win_size])
    average_num = win_sum_nums // win_size
    ans_list[k] = average_num
    for i in range(win_size, len_nums):
        win_sum_nums = win_sum_nums - nums[i - win_size] + nums[i]
        average_num = win_sum_nums // win_size
        ans_list[i - k] = average_num
    return ans_list

# 38. 长度为K子数组中的最大和
def maximumSubarraySum(nums: List[int], k: int) -> int:
    len_nums = len(nums)
    if k > len_nums:
        return 0
    max_sum = 0
    current_sum = sum(nums[:k])
    if len(nums[:k]) == len(set(nums[:k])):
        max_sum = current_sum
    for i in range(k, len_nums):
        current_sum = current_sum - nums[i - k] + nums[i]
        if len(set(nums[i - k + 1 : i + 1])) == k:
            max_sum = max(max_sum, current_sum)
    return max_sum

# 38.1 长度为K子数组中的最大和 -- 更快的方法 -- 使用频次哈希表
def maximumSubarraySumFaster(nums: List[int], k: int) -> int:
    from collections import defaultdict
    n = len(nums)
    if k > n:
        return 0
    freq: Dict[int, int] = defaultdict(int)
    current_sum = 0
    for i in range(k):
        freq[nums[i]] += 1
        current_sum += nums[i]
    max_sum = current_sum if len(freq) == k else 0
    for i in range(k, n):
        left = nums[i - k]
        right = nums[i]
        freq[left] -= 1
        if freq[left] == 0:
            del freq[left]
        current_sum -= left
        freq[right] += 1
        current_sum += right
        if len(freq) == k:
            max_sum = max(max_sum, current_sum)
    return max_sum

# 39. 执行操作后的最大MEX
def findSmallestInteger(nums: List[int], value: int) -> int:
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
def maxScore(cardPoints: List[int], k: int) -> int:
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
def maxDistinctElements(nums: List[int], k: int) -> int:
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
def minArrivalsToDiscard(arrivals: List[int], w: int, m: int) -> int:
    ans = 0
    max_val = max(arrivals) if arrivals else 0
    cnt = [0] * (max_val + 1)
    for i, x in enumerate(arrivals):
        if m == cnt[x]:
            arrivals[i] = 0
            ans += 1
        else:
            cnt[x] += 1
        left = i - w + 1
        if left >= 0:
            cnt[arrivals[left]] -= 1
    return ans

# 43. 执行操作后字典序最小的字符串
def findLexSmallestString(s: str, a: int, b: int) -> str:
    def accumulate(s: str, a: int) -> str:
        s_list = list(s)
        for i in range(1, len(s_list), 2):
            digit = int(s_list[i])
            s_list[i] = str((digit + a) % 10)
        return ''.join(s_list)
    def rotation(s: str, b: int) -> str:
        if not s:
            return s
        b = b % len(s)
        return s[-b:] + s[:-b]
    visited: Set[str] = set()
    queue = deque([s])
    visited.add(s)
    while queue:
        curr = queue.popleft()
        s1 = accumulate(curr, a)
        if s1 not in visited:
            visited.add(s1)
            queue.append(s1)
        s2 = rotation(curr, b)
        if s2 not in visited:
            visited.add(s2)
            queue.append(s2)
    return min(visited)

# 44. 爱生气的书店老板 -- 滑动窗口进阶
def maxSatisfied(customers: List[int], grumpy: List[int], minutes: int) -> int:
    n = len(customers)
    total = sum(customers[i] for i in range(n) if grumpy[i] == 0)
    max_extra = 0
    current_extra = 0
    for i in range(n):
        if grumpy[i] == 1:
            current_extra += customers[i]
        if i >= minutes and grumpy[i - minutes] == 1:
            current_extra -= customers[i - minutes]
        max_extra = max(max_extra, current_extra)
    return total + max_extra

# 45. 按策略买卖股票的最佳时机 -- 滑动窗口进阶
def maxProfit(prices: List[int], strategy: List[int], k: int) -> int:
    n = len(prices)
    original_profit = sum(s * p for s, p in zip(strategy, prices))
    if k > n:
        return original_profit

    half = k // 2
    # 初始窗口 [0, k-1]
    orig_win = sum(strategy[i] * prices[i] for i in range(k))
    new_win = sum(prices[i] for i in range(half, k))
    
    max_profit = original_profit - orig_win + new_win

    # 滑动窗口
    for i in range(1, n - k + 1):
        orig_win += -strategy[i - 1] * prices[i - 1] + strategy[i + k - 1] * prices[i + k - 1]
        new_win += -prices[i + half - 1] + prices[i + k - 1]
        current = original_profit - orig_win + new_win
        if current > max_profit:
            max_profit = current

    return max(max_profit, original_profit)

# 46. 执行操作后的变量值
def finalValueAfterOperations(operations: List[str]) -> int:
    x = 0
    for op in operations:
        if "--" in op:
            x -= 1
        else:
            x += 1
    
    return x

# 47. 重新安排会议得到最多空余时间 I -- 滑动窗口进阶练习
def maxFreeTime(eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
    n = len(startTime)
    free = [0] * (n + 1)
    free[0] = startTime[0]  # 最左边的空余时间段
    for i in range(1, n):
        free[i] = startTime[i] - endTime[i - 1]
    free[n] = eventTime - endTime[-1]

    # 套定长滑窗模板（窗口长为 k+1)
    ans = s = 0
    for i, f in enumerate(free):
        s += f
        if i < k:
            continue
        ans = max(ans, s)
        s -= free[i - k]
    return ans

# 48. 执行操作后元素的最高频率 I
def maxFrequency(nums: list[int], k: int, numOperations: int) -> int:
    counter = Counter(nums)
    nums.sort()
    
    # 初始化
    left = 0
    right = 0
    ans = 0
    
    # 枚举潜在的目标值
    for target in range(nums[0], nums[-1] + 1):
        # 无法转换成 target 的数从窗口左侧移出
        while nums[left] < target - k:
            left += 1
        
        # 可以转换成 target 的数都包含进窗口右侧
        while right < len(nums) and nums[right] <= target + k:
            right += 1
        
        # 元素总数
        total = right - left
        # 初始频率
        initial_freq = counter.get(target, 0)
        
        # 可以用来转换的数量
        convert = total - initial_freq
        # 实际能转换的数量
        op_use = min(numOperations, convert)
        
        # 初始频率 + 新转换过来的数量
        total_freq = initial_freq + op_use
        ans = max(ans, total_freq)

    return ans

# 49. 最少交换次数来组合所有的 1 II -- 滑动窗口进阶练习
def minSwaps(nums: List[int]) -> int:
    len_nums = len(nums)
    total_ones = sum(nums)
    
    if total_ones == 0 or total_ones == len_nums:
        return 0
    
    # 对数组进行环形处理
    extened = nums + nums
    
    # 初始化第一个窗口
    current_ones = sum(extened[:total_ones])
    max_ones = current_ones
    
    # 滑动窗口的精髓 -- 减左 + 加右
    for i in range(1, len_nums):
        current_ones -= extened[i - 1]
        current_ones += extened[i + total_ones - 1]
        max_ones = max(max_ones, current_ones)
    
    return total_ones - max_ones

# 49. 执行操作后元素的最高频率 II
def maxFrequencyII(nums: list[int], k: int, numOperations: int) -> int:
    nums.sort()
    n = len(nums)
    ans = cnt = left = right = 0
    for i,x in enumerate(nums):
        cnt += 1
        #循环直到连续相同段的末尾,这样可以统计出x的出现次数
        if i < n - 1 and x == nums[i + 1]:
            continue
        while nums[left] < x - k:
            left += 1
        while right < n and nums[right] <= x + k:
            right += 1
        ans = max(ans,min(right - left,cnt + numOperations))
        cnt = 0
    
    if ans >= numOperations:
        return ans
    
    left = 0
    for right,x in enumerate(nums):
        while nums[left] < x - k * 2:
            left += 1
        ans  = max(ans,right - left + 1)
    return min(ans,numOperations)

# 50. 拆炸弹 -- 滑动窗口进阶练习
def decrypt(code: list[int], k: int) -> List[int]:
    n = len(code)
    if k == 0:
        return [0] * n
    
    res = [0] * n
    # 统一处理:计算每个位置的窗口和
    for i in range(n):
        total = 0
        if k > 0:
            for j in range(1, k + 1):
                total += code[(i + j) % n]
        else:
            for j in range(k, 0):  # k is negative
                total += code[(i + j) % n]
        res[i] = total
    return res
        
# 51. 判断操作后字符串中的数字是否相等 I
def hasSameDigits(s: str) -> bool:
    while len(s) > 2:
        s = ''.join(
            str((int(s[i]) + int(s[i + 1])) % 10)
            for i in range(len(s) - 1)
        )
    
    return s[0] == s[1]

# 52. 下一个更大的数值平衡数
def nextBeautifulNumber(n: int) -> int:
    res = n + 1
    
    while True:
        flag = True
        cnt_dict = Counter(str(res))
        for cnt_keys, cnt_values in cnt_dict.items():
            if (0 == int(cnt_keys) or
                int(cnt_keys) != cnt_values
                ):
                flag = False
                break
        
        if flag:
            return res
        
        res += 1    

# 53. 计算力扣银行的钱 -- 等差数列
def totalMoney(n: int) -> int:
    a = n // 7
    b = n % 7
    money = (49 + 7 * a) * a / 2 + (2 * a + b + 1) * b / 2
    
    return int(money)

# 53.1. 计算力扣银行的钱 -- 模拟
def totalMoneyBySimulate(n: int) -> int:
    sum_money = 0
    if n <= 7:
        for i in range(1, n + 1):
            sum_money += i
    else:
        week_num = n // 7
        day_num = n % 7
        week_money = (49 + 7 * week_num) * week_num / 2
        day_money = (2 * week_num + day_num + 1) * day_num / 2
        sum_money = int(week_money + day_money)
    
    return sum_money

# 54. 简易银行系统
class Bank:
    def __init__(self, balance: List[int]):
        self.balance = balance
        self.len_balance = len(balance)
    # 转账
    def transfer(self, account1: int, account2: int, money: int) -> bool:
        if (
            (self.len_balance < account1) or
            (self.len_balance < account2) or
            (self.balance[account1 - 1] < money)
            ):
            
            return False
        else:
            self.balance[account1 - 1] -= money
            self.balance[account2 - 1] += money
            
            return True
    # 存款
    def deposit(self, account: int, money: int) -> bool:
        if self.len_balance < account:
            
            return False
        else:
            self.balance[account - 1] += money
            
            return True
    # 取款
    def withdraw(self, account: int, money: int) -> bool:
        if ((self.len_balance < account) or 
            (self.balance[account - 1] < money)):
            
            return False
        else:
            self.balance[account - 1] -= money
            
            return True
        
# 55. 银行中的激光束数量
def numberOfBeams(bank: List[str]) -> int:
    total = 0
    prev = 0
    for row in bank:
        curr = row.count("1")
        if curr:
            total += prev * curr
            prev = curr
    return total

# 56. 仅含置位位的最小整数
def smallestNumber(n: int) -> int:
    res = 1
    while res <= n:
        res = (res <<  1) + 1
    
    return res

# 57. 形成目标数组的子数组最少增加次数 -- 差分
def minNumberOperations(target: List[int]) -> int:
        return target[0] + sum(max(0, target[i] - target[i - 1]) for i in range(1, len(target)))

# 58. 数字小镇中的捣蛋鬼
def getSneakyNumbers(nums: List[int]) -> List[int]:
    ans_list = []
    nums_dict = defaultdict(int)
    for i in nums:
        nums_dict[i] += 1
    for nums_keys in nums_dict.keys():
        if nums_dict[nums_keys] > 1:
            ans_list.append(nums_keys)
    
    return ans_list

# 59. 从链表中移除在数组中存在的节点
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
def modifiedList(nums: List[int], head: Optional[ListNode]) -> Optional[ListNode]:
    nums_set = set(nums)
    dummy = ListNode(0)
    dummy.next = head

    prev = dummy
    curr = head

    while curr:
        if curr.val in nums_set:
            prev.next = curr.next
        else:
            prev = curr
        curr = curr.next
    
    return dummy.next

# 60. 统计网格图中没有被保卫的格子数
def countUnguarded(m: int, n: int, guards: List[List[int]], walls: List[List[int]]) -> int:
        # 左右上下
        DIRS = (0, -1), (0, 1), (-1, 0), (1, 0)
        guarded = [[0] * n for _ in range(m)]

        # 标记警卫格子,墙格子
        for x, y in guards:
            guarded[x][y] = -1
        for x, y in walls:
            guarded[x][y] = -1

        # 遍历警卫
        for x0, y0 in guards:
            # 遍历视线方向(左右上下)
            for dx, dy in DIRS:
                # 视线所及之处,被保卫
                x, y = x0 + dx, y0 + dy
                while 0 <= x < m and 0 <= y < n and guarded[x][y] != -1:
                    guarded[x][y] = 1  # 被保卫
                    x += dx
                    y += dy

        # 统计没被保卫(值为 0)的格子数
        return sum(row.count(0) for row in guarded)

# 61. 使绳子变成彩色的最短时间
def minCost(colors: str, neededTime: List[int]) -> int:
        ans = max_t = 0
        for i, t in enumerate(neededTime):
            ans += t
            if t > max_t:
                max_t = t
            if i == len(colors) - 1 or colors[i] != colors[i + 1]:
                # 遍历到了连续同色段的末尾
                ans -= max_t  # 保留耗时最大的气球
                max_t = 0  # 准备计算下一段的最大耗时
        return ans

# 62. 计算子数组的 x-sum I
def findXSum(nums: List[int], k: int, x: int) -> List[int]:
    n = len(nums)
    # 存储当前窗口内元素的频率
    counts = {}
    res = []

    # 初始化第一个窗口
    for i in range(k):
        value = nums[i]
        counts[value] = counts.get(value, 0) + 1

    # 计算当前窗口的 x-sum
    def cal_x_sum():
        # 降序
        sorted_items = sorted(counts.items(), key=lambda item: (-item[1], -item[0]))
        # 处理唯一元素少于 x 的情况
        l = len(sorted_items)
        limit = x if x < l else l
        
        total_sum = 0
        # 累加前 limit 个元素的和
        for j in range(limit):
            value, count = sorted_items[j]
            total_sum += value * count
        return total_sum

    # 第一个窗口
    res.append(cal_x_sum())

    # 遍历所有长度为 k 的子数组
    for i in range(n - k):
        # 确定移出和移入窗口的元素值
        out_value = nums[i]
        in_value = nums[i + k]

        # 移除最左侧的元素
        count = counts.get(out_value, 0)
        if count <= 1:
            if out_value in counts:
                del counts[out_value]
        else:
            counts[out_value] -= 1

        # 将新元素添加到窗口
        counts[in_value] = counts.get(in_value, 0) + 1
        # 计算新窗口的 x-sum
        res.append(cal_x_sum())

    return res

# 63. 电网维护
class Solution:
    def processQueries(self, n, connections, queries):

        onlines = [True] * (n + 1)
        parents = [value for value in range(n + 1)]

        def find(x):
            if parents[x] == x:
                return x
            parents[x] = find(parents[x])
            return parents[x]

        def union(x1, x2):
            y1, y2 = find(x1), find(x2)
            if y1 != y2:
                parents[y1] = y2

        for x1, x2 in connections:
            union(x1, x2)

        listMap = defaultdict(list)
        for x in range(1, n + 1):
            listMap[find(x)].append(x)

        for stationList in listMap.values():
            stationList.sort(reverse=True)

        def minStation(x):
            stationList = listMap[find(x)]
            while stationList and not onlines[stationList[-1]]:
                stationList.pop()
            return stationList[-1] if stationList else -1

        resultList = []
        for operation, x in queries:
            if operation == 2:
                onlines[x] = False
            elif onlines[x]:
                resultList.append(x)
            else:
                resultList.append(minStation(x))
        return resultList

# 64. 最大化城市的最小电量
def maxPower(stations: List[int], r: int, k: int) -> int:
        n = len(stations)
        # 滑动窗口
        s = sum(stations[:r])  # 先计算 [0, r-1] 的发电量，为第一个窗口做准备
        power = [0] * n
        for i in range(n):
            # 右边进
            if (right := i + r) < n:
                s += stations[right]
            # 左边出
            if (left := i - r - 1) >= 0:
                s -= stations[left]
            power[i] = s

        def check(low: int) -> bool:
            diff = [0] * n  # 差分数组
            sum_d = built = 0
            for i, p in enumerate(power):
                sum_d += diff[i]  # 累加差分值
                m = low - (p + sum_d)
                if m <= 0:
                    continue
                # 需要在 i+r 额外建造 m 个供电站
                built += m
                if built > k:  # 不满足要求
                    return False
                # 把区间 [i, i+2r] 增加 m
                sum_d += m  # 由于 diff[i] 后面不会再访问，我们直接加到 sum_d 中
                if (right := i + r * 2 + 1) < n:
                    diff[right] -= m
            return True

        # 开区间二分
        mn = min(power)
        left, right = mn + k // n, mn + k + 1
        while left + 1 < right:
            mid = (left + right) // 2
            if check(mid):
                left = mid
            else:
                right = mid
        return left

# 65. 使整数变为 0 的最少操作次数
def minimumOneBitOperations(n: int) -> int:
    ans = 0
    while n:
        ans ^= n
        n //= 2
    return ans

# 66. 得到0的操作数
def countOperations(num1: int, num2: int) -> int:
        count = 0
        while num1 != 0 and num2 != 0:
            if num1 >= num2:
                num1 -= num2
            else:
                num2 -= num1
            count += 1
            
        return count

# 67. 将所有元素变为 0 的最少操作次数
def minOperations(nums: List[int]) -> int:
        ans = 0
        top = -1  # 栈顶下标

        for x in nums:
            # 小于栈顶，说明需要增加一次操作
            while top >= 0 and x < nums[top]:
                top -= 1  # 出栈
                ans += 1
            # 栈空，说明这是一个新块的开始
            # 比栈顶大的新值，是 “上坡” 的新台阶，压入等待处理
            if top < 0 or x != nums[top]:
                top += 1
                nums[top] = x  # 入栈
        
        return ans + top + (nums[0] > 0)

# 68. 一和零
def findMaxForm(strs: List[str], m: int, n: int) -> int:
    f  = [[0] * (n + 1) for _ in range(m + 1)]
    sum0 = sum1 = 0
    for s in strs:
        cnt0 = s.count('0')
        cnt1 = len(s) - cnt0
        sum0 = min(sum0 + cnt0, m)
        sum1 = min(sum1 + cnt1, n)
        for j in range(sum0, cnt0 - 1, -1):
            for k in range(sum1, cnt1 - 1, -1):
                v = f[j - cnt0][k - cnt1] + 1
                if v > f[j][k]:
                    f[j][k] = v
    return max(map(max, f))