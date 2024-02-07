"""https://www.acwing.com/problem/content/75/
79. 滑动窗口的最大值(困难）
给定一个数组和滑动窗口的大小，请找出所有滑动窗口里的最大值。
例如，如果输入数组 [2,3,4,2,6,2,5,1]及滑动窗口的大小 ，那么一共存在 6个滑动窗口，它们的最大值分别为 [4,4,6,6,6,5]。
注意：
数据保证 k大于 0，且 k小于等于数组长度。
数据范围
数组长度 [1,1000]。
样例
输入：[2, 3, 4, 2, 6, 2, 5, 1] , k=3
输出: [4, 4, 6, 6, 6, 5]
"""


"""
80. 骰子的点数
请求出投掷 n次，掷出 n∼6n点分别有多少种掷法。
数据范围
1≤n≤10
样例1
输入：n=1
输出：[1, 1, 1, 1, 1, 1]
解释：投掷1次，可能出现的点数为1-6，共计6种。每种点数都只有1种掷法。所以输出[1, 1, 1, 1, 1, 1]。
输入：n=2
输出：[1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
解释：投掷2次，可能出现的点数为2-12，共计11种。每种点数可能掷法数目分别为1,2,3,4,5,6,5,4,3,2,1。
所以输出[1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]。
"""


class Solution(object):
    def numberOfDice(self, n):
        """
        :type n: int
        :rtype: List[int]
        每一行利用上一行的数据，进行累计，累加区间为6个长度
        """
        res = [1, 1, 1, 1, 1, 1]
        for i in range(1, n):
            res += [0] * 5
            res = [sum(res[max(0, k - 5):k + 1]) for k in range(len(res))]
        return res

"""
81. 扑克牌的顺子
从扑克牌中随机抽 5张牌，判断是不是一个顺子，即这 5张牌是不是连续的。
2∼10为数字本身，A为 1，J 为 11，Q 为 12，K 为 13，大小王可以看做任意数字。
为了方便，大小王均以 0来表示，并且假设这副牌中大小王均有两张。
样例1
输入：[8,9,10,11,12]
输出：true
"""


class Solution(object):
    def isContinuous(self, numbers):
        """
        :type numbers: List[int]
        :rtype: bool
        """
        nums = [x for x in numbers if x]
        return max(nums) - min(nums) < 5 if len(nums) == len(set(nums)) and nums else False

"""
82. 圆圈中最后剩下的数字
0,1,…,n−1这 n个数字 (n>0)排成一个圆圈，从数字 0开始每次从这个圆圈里删除第 m个数字。
求出这个圆圈里剩下的最后一个数字。
数据范围
1≤n,m≤4000
样例
输入：n=5 , m=3
输出：3
"""
class Solution(object):
    def lastRemaining(self, n, m):
        """
        :type n: int
        :type m: int
        :rtype: int
        """
        if n == 1: return 0
        nums = list(range(n))
        idx = 0
        while n > 1:
            idx = (idx + m - 1) % n
            nums.remove(nums[idx])
            n -= 1
        return nums[0]

"""
83. 股票的最大利润
假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖 一次 该股票可能获得的利润是多少？
例如一只股票在某些时间节点的价格为 [9,11,8,5,7,12,16,14]。
如果我们能在价格为 5的时候买入并在价格为 16时卖出，则能收获最大的利润 11。
数据范围
输入数组长度 [0,500]。
样例
输入：[9, 11, 8, 5, 7, 12, 16, 14]
输出：11
"""

class Solution(object):
    def maxDiff(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        枚举每一天作为卖出的日子，能获得多少钱，然后更新diff，并且更新最小值，为第二天的计算做准备
        """
        if not nums: return 0
        min_ = nums[0]
        diff = 0
        for n in nums:
            diff = max(diff, n - min_)
            min_ = min(min_, n)
        return diff