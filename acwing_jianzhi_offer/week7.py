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

"""
84. 求1+2+…+n
求 1+2+…+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句 (A?B:C)。
1≤n≤50000。
样例
输入：10
输出：55
"""
class Solution(object):
    def getSum(self, n):
        """
        :type n: int
        :rtype
        nt
        """
        return sum(range(n+1))

"""https://www.acwing.com/solution/content/44616/(题解）
85. 不用加减乘除做加法
写一个函数，求两个整数之和，要求在函数体内不得使用 ＋、－、×、÷四则运算符号。
数据范围
输入和输出都在int范围内。
样例
输入：num1 = 1 , num2 = 2
输出：3
"""
class Solution(object):
    def add(self, num1, num2):
        """
        :type num1: int
        :type num2: int
        :rtype: int
        """
        return sum([num1, num2])

"""
86. 构建乘积数组
给定一个数组A[0, 1, …, n-1]，请构建一个数组B[0, 1, …, n-1]，其中B中的元素B[i]=A[0]×A[1]×… ×A[i-1]×A[i+1]×…×A[n-1]。不能使用除法。
数据范围
输入数组长度 [0,20]。
样例
输入：[1, 2, 3, 4, 5]
输出：[120, 60, 40, 30, 24]
思考题：
能不能只使用常数空间？（除了输出的数组之外）
"""


class Solution(object):
    def multiply(self, A):
        """
        :type A: List[int]
        :rtype: List[int]
        先乘前面，再乘后面
        """
        len_ = len(A)
        tmp = 1
        B = [1] * len_
        for i in range(1, len_):
            tmp *= A[i - 1]
            B[i] *= tmp

        tmp = 1
        for i in range(len_ - 2, -1, -1):
            tmp *= A[i + 1]
            B[i] *= tmp

        return B

"""
87. 把字符串转换成整数
请你写一个函数 StrToInt，实现把字符串转换成整数这个功能。当然，不能使用 atoi 或者其他类似的库函数。
数据范围
输入字符串长度 [0,20]。
样例
输入："123"
注意:
你的函数应满足下列条件：
1)忽略所有行首空格，找到第一个非空格字符，可以是 ‘+/−’ 表示是正数或者负数，紧随其后找到最长的一串连续数字，将其解析成一个整数；
2)整数后可能有任意非数字字符，请将其忽略；
3)如果整数长度为 0，则返回 0；
4)如果整数大于 INT_MAX(2^31−1)，请返回 INT_MAX；如果整数小于INT_MIN(−2^31) ，请返回 INT_MIN；
"""


class Solution(object):
    def strToInt(self, str):
        """
        :type str: str
        :rtype: int
        """
        str = str.replace(' ', '')
        f = -1 if str and str[0] == '-' else 1
        str = str.replace('+', '').replace('-', '')

        INT_MAX = 2 ** 31 - 1
        INT_MIN = -2 ** 31
        string = '0123456789'
        str2num_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
        s = 0
        for char in str:
            if char not in string:
                break
            s = 10 * s + f * str2num_dict[char]
            if s >= INT_MAX:
                return INT_MAX
            if s <= INT_MIN:
                return INT_MIN
        return s

"""
88. 树中两个结点的最低公共祖先
给出一个二叉树，输入两个树节点，求它们的最低公共祖先。一个树节点的祖先节点包括它本身。
注意：
输入的二叉树不为空；输入的两个节点一定不为空，且是二叉树中的节点；
数据范围
树中节点数量 [0,500]。
样例
二叉树[8, 12, 2, null, null, 6, 4, null, null, null, null]如下图所示：
    8
   / \
  12  2
     / \
    6   4
1. 如果输入的树节点为2和12，则输出的最低公共祖先为树节点8。
2. 如果输入的树节点为2和6，则输出的最低公共祖先为树节点2。
"""
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if not root.right and not root.left:
            return root
        self.p, self.q = p, q
        self.dfs(root)
        return self.res

    def dfs(self, root):
        if not root: return False
        left = self.dfs(root.left) if root.left else False
        right = self.dfs(root.right) if root.right else False
        head = root == self.p or root == self.q
        if head and (right or left) or (right and left):
            self.res = root
        return head or left or right
