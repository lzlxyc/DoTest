"""
68. 0到n-1中缺失的数字
一个长度为 n−1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围 0到 n−1之内。
在范围 0到 n−1的 n个数字中有且只有一个数字不在该数组中，请找出这个数字。
数据范围
1≤n≤1000
样例
输入：[0,1,2,4]
输出：3
"""
class Solution(object):
    def getMissingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums: return 0
        n = len(nums)
        return int(n*(n+1)/2) - sum(nums)

"""
69. 数组中数值和下标相等的元素
假设一个单调递增的数组里的每个元素都是整数并且是唯一的。
请编程实现一个函数找出数组中任意一个数值等于其下标的元素。
例如，在数组 [−3,−1,1,3,5]中，数字 3和它的下标相等。
数据范围
数组长度 [1,100]。
样例
输入：[-3, -1, 1, 3, 5]
输出：3
注意:如果不存在，则返回-1。
"""
class Solution(object):
    def getNumberSameAsIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        if x < idx: x-- < idx--
        if x > idx: x++ > idx++
        """
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = int(l + r >> 1)
            if nums[mid] == mid: return mid
            if nums[mid] < mid: l = mid + 1
            else: r = mid - 1
        return -1

"""
70. 二叉搜索树的第k个结点
给定一棵二叉搜索树，请找出其中的第 k小的结点。
你可以假设树和 k都存在，并且 1≤k≤树的总结点数。
数据范围
树中节点数量 [1,500]。
样例
输入：root = [2, 1, 3, null, null, null, null] ，k = 3
    2
   / \
  1   3
输出：3
"""
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution(object):
    def kthNode(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: TreeNode
        就是求中序遍历后输出第k个值
        """
        res, tmp = [], []
        while root or tmp:
            while root:
                tmp.append(root)
                root = root.left
            root = tmp.pop()
            res.append(root)
            root = root.right
        return res[k-1]

"""
71. 二叉树的深度
输入一棵二叉树的根结点，求该树的深度。
从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。
数据范围
树中节点数量 [0,500]。
样例
输入：二叉树[8, 12, 2, null, null, 6, 4, null, null, null, null]如下图所示：
    8
   / \
  12  2
     / \
    6   4
输出：3
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # 方法1
    # def treeDepth(self, root):
    #     """
    #     :type root: TreeNode
    #     :rtype: int
    #     """
    #     if not root: return 0
    #     self.max_ = 1
    #     self.dfs(root, 0)
    #     return self.max_
    #
    # def dfs(self, root, c):
    #     if not root:
    #         self.max_ = max(self.max_, c)
    #         return
    #     self.dfs(root.left, c + 1)
    #     self.dfs(root.right, c + 1)
    #     return

    # 方法2：
    def treeDepth(self, root):
        return 1 + max(self.treeDepth(root.left),self.treeDepth(root.right)) if root else 0

"""
72. 平衡二叉树
如果某二叉树中任意结点的左右子树的深度相差不超过 1，那么它就是一棵平衡二叉树。
注意：
规定空树也是一棵平衡二叉树。
数据范围
树中节点数量 [0,500]。
样例
输入：二叉树[5,7,11,null,null,12,9,null,null,null,null]如下所示，
    5
   / \
  7  11
    /  \
   12   9
"""


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if self.dfs(root) < 0: return False
        return True

    def dfs(self, root):
        if not root: return 0
        left = self.dfs(root.left)
        if left < 0: return - 1
        right = self.dfs(root.right)
        if right < 0: return - 1
        if abs(left - right) > 1: return -1
        return 1 + max(self.dfs(root.left), self.dfs(root.right))

"""
73. 数组中只出现一次的两个数字
一个整型数组里除了两个数字之外，其他的数字都出现了两次。
请写程序找出这两个只出现一次的数字。
你可以假设这两个数字一定存在。
数据范围
数组长度 [1,1000]。
样例
输入：[1,2,3,3,4,4]
输出：[1,2]
"""


class Solution(object):
    def findNumsAppearOnce(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # num_dict = {}
        # res = []
        # for n in nums:
        #     if n not in num_dict:
        #         num_dict[n] = 1
        #     else:
        #         num_dict[n] = 2

        # for k,v in num_dict.items():
        #     if v == 1:
        #         res.append(k)

        # return res

        a_b = 0
        for n in nums:
            a_b ^= n
            # 得到ab异或的结果
        k = 0
        while not (a_b >> k & 1):
            k += 1

        a = 0
        for n in nums:
            if n >> k & 1:
                a ^= n
        b = a_b ^ a
        return [a, b]

"""https://www.acwing.com/problem/content/70/
74. 数组中唯一只出现一次的数字(困难）
在一个数组中除了一个数字只出现一次之外，其他数字都出现了三次。
请找出那个只出现一次的数字。
你可以假设满足条件的数字一定存在。
思考题：
如果要求只使用 O(n)的时间和额外 O(1)的空间，该怎么做呢？
数据范围
数组长度 [1,1500]。
数组内元素取值范围 [0,1000]。
样例
输入：[1,1,1,2,2,2,3,4,4,4]
输出：3
"""


"""
75. 和为S的两个数字
输入一个数组和一个数字 s，在数组中查找两个数，使得它们的和正好是 s。
如果有多对数字的和等于 s，输出任意一对即可。
你可以认为每组输入中都至少含有一组满足条件的输出。
数据范围
数 组长度 [1,1002]。
样例
输入：[1,2,3,4] , sum=7
输出：[3,4]
"""


class Solution(object):
    def findNumbersWithSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        nums.sort()
        l, r = 0, len(nums) - 1
        while l < r:
            s = nums[l] + nums[r]
            if s == target:
                return [nums[l], nums[r]]
            if s < target:
                l += 1
            else:
                r -= 1

"""
76. 和为S的连续正数序列
输入一个非负整数 S，打印出所有和为 S的连续正数序列（至少含有两个数）。
例如输入 15，由于 1+2+3+4+5=4+5+6=7+8=15，所以结果打印出 3个连续序列 1∼5、4∼6和 7∼8。
数据范围
0≤S≤1000
样例
输入：15
输出：[[1,2,3,4,5],[4,5,6],[7,8]]
"""


class Solution(object):
    def findContinuousSequence(self, sum):
        """
        :type sum: int
        :rtype: List[List[int]]
        """
        if sum < 3: return []
        k, l, r = 1, 1, int(sum >> 1) + 1
        tot, res = 0, []
        while l <= r + 1:
            if tot == sum:
                res.append(list(range(k, l)))
                tot -= k
                k += 1
            while tot > sum:
                tot -= k
                k += 1

            if tot < sum:
                tot += l
                l += 1

        return res





"""
77. 翻转单词顺序
输入一个英文句子，单词之间用一个空格隔开，且句首和句尾没有多余空格。
翻转句子中单词的顺序，但单词内字符的顺序不变。
为简单起见，标点符号和普通字母一样处理。
例如输入字符串"I am a student."，则输出"student. a am I"。
数据范围
输入字符串长度 [0,1000]。
样例
输入："I am a student."
输出："student. a am I"
"""
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        t = s.split()
        t.reverse()
        return ' '.join(t)

"""
78. 左旋转字符串
字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。
请定义一个函数实现字符串左旋转操作的功能。
比如输入字符串"abcdefg"和数字 2，该函数将返回左旋转 2位得到的结果"cdefgab"。
注意：
数据保证 n小于等于输入字符串的长度。
数据范围
输入字符串长度 [0,1000]。
样例
输入："abcdefg" , n=2
输出："cdefgab"
"""
class Solution(object):
    def leftRotateString(self, s, n):
        """
        :type s: str
        :type n: int
        :rtype: str
        """
        return s[n:] + s[:n]