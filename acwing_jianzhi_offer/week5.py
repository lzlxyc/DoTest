"""
57. 数字序列中某一位的数字
数字以 0123456789101112131415…的格式序列化到一个字符序列中。
在这个序列中，第 5位（从 0开始计数）是 ，第 13位是 ，第 19位是 4，等等。
请写一个函数求任意位对应的数字。
数据范围0≤输入数字 ≤2147483647
样例
输入：13
输出：1
"""
class Solution(object):
    def digitAtIndex(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n < 10: return n
        k = 1
        tot = 9*k*10**(k-1)
        while n > tot:
            n -= tot
            k += 1
            tot = 9*k*10**(k-1)
        p = 10**(k-1) + int(n/k)
        d = n%k
        if not d: return int(str(p-1)[-1])
        return int(str(p)[d-1])

"""
58. 把数组排成最小的数
输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
例如输入数组 [3,32,321]，则打印出这 3个数字能排成的最小数字 321323。
数据范围
数组长度 [0,500]。
样例
输入：[3, 32, 321]
输出：321323
"""
from functools import cmp_to_key

def cmp(a,b):
    return int(str(a) + str(b)) -  int(str(b) +str(a))

class Solution(object):
    def printMinNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        if not nums: return ''
        nums = [str(n) for n in nums]
        nums.sort(key=cmp_to_key(cmp))
        return ''.join(nums)

"""
59. 把数字翻译成字符串
给定一个数字，我们按照如下规则把它翻译为字符串：0翻译成 a，1翻译成 b，……，11翻译成 l，……，25翻译成 z。
一个数字可能有多个翻译。
例如 12258有 5种不同的翻译，它们分别是 bccfi、bwfi、bczi、mcfi 和 mzi。
请编程实现一个函数用来计算一个数字有多少种不同的翻译方法。
数据范围
输入数字位数 [1,101]。
样例
输入："12258"
输出：5
"""
class Solution:
    def getTranslationCount(self, s):
        """
        :type s: str
        :rtype: int
        """
        if len(s)==1: return 1
        nums = list(s)
        p, q = nums.pop(), nums.pop()
        f1, f2 = 1, 1
        if int(q+p) <= 25: f2 += 1
        while nums:
            v = nums.pop()
            if int(v)==0 or int(v+q) > 25:
                f1 = f2
            else:
                tot = f1+f2
                f1 = f2
                f2 = tot
            q = v
        return f2

"""
60. 礼物的最大价值
在一个 m×n的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。
你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格直到到达棋盘的右下角。
给定一个棋盘及其上面的礼物，请计算你最多能拿到多少价值的礼物？
注意：m,n>0m×n≤1350
样例：
输入：
[
  [2,3,1],
  [1,7,1],
  [4,6,1]
]
输出：19
"""
class Solution(object):
    def getMaxValue(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        if not grid or not grid[0]: return 0
        n, m = len(grid), len(grid[0])
        tmp, tot = [], 0
        for v in grid[0]:
            tot += v
            tmp.append(tot)
        for y in range(1,n):
            tmp[0] += grid[y][0]
            for x in range(1,m):
                tmp[x] = max(tmp[x],tmp[x-1]) + grid[y][x]
        return tmp[-1]

"""
61. 最长不含重复字符的子字符串
请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。
假设字符串中只包含从 a 到 z 的字符。
数据范围
输入字符串长度 [0,1000]。
样例
输入："abcabc"
输出：3
"""


class Solution:
    def longestSubstringWithoutDuplication(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s: return 0
        tmp = ''
        max_len = 1
        for char in s:
            if char not in tmp:
                tmp += char
            else:
                idx = tmp.find(char)
                max_len = max(max_len, len(tmp))
                tmp = tmp[tmp.find(char) + 1:] + char

        return max(max_len, len(tmp))

"""
62. 丑数
我们把只包含质因子 2、3和 5的数称作丑数（Ugly Number）。
例如 6、8都是丑数，但 14不是，因为它包含质因子 7。
求第 n个丑数的值。
数据范围
1≤n≤1000
样例
输入：5
输出：5
注意：习惯上我们把 1当做第一个丑数
"""
class Solution(object):
    def getUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 1: return n
        i, j, k = 0, 0, 0
        res = [1]
        while n:
            v = min(2*res[i],3*res[j],5*res[k])
            if v == 2*res[i]: i += 1
            if v == 3*res[j]: j += 1
            if v == 5*res[k]: k += 1
            res.append(v)
            n -= 1
        return res[-2]

"""
63. 字符串中第一个只出现一次的字符
在字符串中找出第一个只出现一次的字符。
如输入"abaccdeff"，则输出b。
如果字符串中不存在只出现一次的字符，返回 # 字符。
数据范围
输入字符串长度 [0,1000]。
样例：
输入："abaccdeff"
输出：'b'
"""
class Solution:
    def firstNotRepeatingChar(self, s):
        """
        :type s: str
        :rtype: str
        """
        word_dict = {}
        for char in s:
            if char not in word_dict:
                word_dict[char] = 1
            else: word_dict[char] += 1
        for k, v in word_dict.items():
            if v==1: return k
        return '#'

"""https://www.acwing.com/problem/content/60/
64. 字符流中第一个只出现一次的字符(困难)
请实现一个函数用来找出字符流中第一个只出现一次的字符。
例如，当从字符流中只读出前两个字符 go 时，第一个只出现一次的字符是 g。
当从该字符流中读出前六个字符 google 时，第一个只出现一次的字符是 l。
如果当前字符流没有存在出现一次的字符，返回 # 字符。
数据范围
字符流读入字符数量 [0,1000]。
样例
输入："google"
输出："ggg#ll"
解释：每当字符流读入一个字符，就进行一次判断并输出当前的第一个只出现一次的字符。
"""

"""https://www.acwing.com/problem/content/61/
65. 数组中的逆序对(困难）
在数组中的两个数字如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。
输入一个数组，求出这个数组中的逆序对的总数。
数据范围
给定数组的长度 [0,100]。
样例
输入：[1,2,3,4,5,6,0]
输出：6
"""

"""
66. 两个链表的第一个公共结点
输入两个链表，找出它们的第一个公共结点。
当不存在公共节点时，返回空节点。
数据范围
链表长度 [1,2000]。
保证两个链表不完全相同，即两链表的头结点不相同。
样例
给出两个链表如下所示：
A：        a1 → a2
                   ↘
                     c1 → c2 → c3
                   ↗            
B:     b1 → b2 → b3
输出第一个公共节点c1
"""


# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def findFirstCommonNode(self, headA, headB):
        """
        :type headA, headB: ListNode
        :rtype: ListNode
        """
        # 方法1
        # note_dict = {}
        # p = headA
        # while p:
        #     note_dict[p] = True
        #     p = p.next
        #
        # p = headB
        # while p:
        #     if p in note_dict:
        #         return p
        #     p = p.next
        # return None

        # 方法2
        p, q = headA, headB
        while p!=q:
            if p: p = p.next
            else: p = headB
            if q: q = q.next
            else: q = headA
        return q

"""
67. 数字在排序数组中出现的次数
统计一个数字在排序数组中出现的次数。
例如输入排序数组 [1,2,3,3,3,3,4,5]和数字 3，由于 3在这个数组中出现了 4次，因此输出 4。
数据范围
数组长度 [0,1000]。
样例
输入：[1, 2, 3, 3, 3, 3, 4, 5] ,  3
输出：4
"""


class Solution(object):
    def getNumberOfK(self, nums, k):
        """
        :type nums: list[int]
        :type k: int
        :rtype: int
        """
        l, r = 0, len(nums) - 1
        while l < r:
            mid = int(l + r >> 1)
            if nums[mid] == k: break
            if nums[mid] < k:
                l = mid + 1
            else:
                r = mid - 1

        c = 0
        for i in range(l, r + 1):
            c += int(nums[i] == k)
        return c 